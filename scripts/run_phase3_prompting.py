from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from semantic_bypass import (
    get_run_log_path,
    log_dict,
    log_line,
    log_section,
    setup_logger,
)
from semantic_bypass.checkers import ConstraintSuite
from semantic_bypass.llm import LLMClient, LLMConfig
from semantic_bypass.prompting import PromptVariant, list_prompt_variants, render_schema_hint
from semantic_bypass.schema import SchemaCatalog
from semantic_bypass.spider_utils import (
    create_stratified_subset,
    load_spider_examples,
    load_spider_schemas,
)

CORE_METRICS = ("SHR", "POR", "DIVR")


@dataclass(frozen=True)
class AblationExample:
    example_id: str
    dataset: str
    question: str
    schema: dict[str, dict[str, str]]
    source: str
    reference_sql: str | None = None


@dataclass(frozen=True)
class DatasetLoadResult:
    records: list[AblationExample]
    requested_size: int
    selected_size: int
    candidate_size: int
    source: str
    warnings: list[str]


def _stable_pick(seed_text: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def _type_domain(column_type: str) -> str:
    normalized = column_type.lower()
    if any(token in normalized for token in ("int", "numeric", "decimal", "real", "double", "float")):
        return "numeric"
    if any(token in normalized for token in ("char", "text", "string", "uuid")):
        return "text"
    if any(token in normalized for token in ("date", "time")):
        return "date"
    if "bool" in normalized:
        return "boolean"
    return "other"


def _sample_deterministically(
    records: list[AblationExample],
    sample_size: int,
    seed: int,
) -> list[AblationExample]:
    if sample_size <= 0 or not records:
        return []

    ordered = sorted(records, key=lambda row: row.example_id)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return ordered[: min(sample_size, len(ordered))]


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL row in {path} at line {line_number}: {error}") from error
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_spider_candidates(
    spider_subset_path: Path,
    spider_dir: Path,
    schemas_by_db: dict[str, dict[str, dict[str, str]]],
) -> tuple[list[AblationExample], str, list[str]]:
    warnings: list[str] = []

    if spider_subset_path.exists():
        source = f"subset_jsonl:{spider_subset_path}"
        raw_rows = _load_jsonl_records(spider_subset_path)
        candidates: list[AblationExample] = []
        for index, row in enumerate(raw_rows, start=1):
            db_id = str(row.get("db_id", "")).strip()
            question = str(row.get("question", "")).strip()
            example_id = str(row.get("example_id", "")).strip() or f"subset:{index:05d}"
            if not db_id or not question:
                continue
            schema = schemas_by_db.get(db_id)
            if schema is None:
                warnings.append(f"Spider subset row skipped due to missing schema: db_id={db_id}")
                continue
            candidates.append(
                AblationExample(
                    example_id=f"spider:{example_id}",
                    dataset="spider",
                    question=question,
                    schema=schema,
                    source=str(row.get("source_file", "phase2_spider_subset.jsonl")),
                    reference_sql=str(row.get("sql", "")).strip() or None,
                )
            )
        return candidates, source, warnings

    source = f"spider_raw:{spider_dir}"
    spider_examples, source_counts, spider_warnings = load_spider_examples(spider_dir)
    warnings.extend(spider_warnings)
    warnings.append(f"Spider source counts: {source_counts}")

    filtered_examples = [
        example
        for example in spider_examples
        if example.question.strip() and example.db_id in schemas_by_db
    ]

    if not filtered_examples:
        return [], source, warnings

    practical_pool_size = min(max(300, len(filtered_examples) // 3), len(filtered_examples))
    subset = create_stratified_subset(
        examples=filtered_examples,
        sample_size=practical_pool_size,
        seed=17,
    )

    candidates = [
        AblationExample(
            example_id=f"spider:{example.example_id}",
            dataset="spider",
            question=example.question,
            schema=schemas_by_db[example.db_id],
            source=example.source_file,
            reference_sql=example.sql,
        )
        for example in subset
    ]
    return candidates, source, warnings


def _load_spider_dataset(
    spider_subset_path: Path,
    spider_dir: Path,
    schemas_by_db: dict[str, dict[str, dict[str, str]]],
    sample_size: int,
    seed: int,
) -> DatasetLoadResult:
    candidates, source, warnings = _load_spider_candidates(
        spider_subset_path=spider_subset_path,
        spider_dir=spider_dir,
        schemas_by_db=schemas_by_db,
    )
    selected = _sample_deterministically(candidates, sample_size, seed)
    return DatasetLoadResult(
        records=selected,
        requested_size=sample_size,
        selected_size=len(selected),
        candidate_size=len(candidates),
        source=source,
        warnings=warnings,
    )


def _load_synthetic_dataset(
    synthetic_dataset_path: Path,
    sample_size: int,
    seed: int,
) -> DatasetLoadResult:
    warnings: list[str] = []
    if sample_size <= 0:
        return DatasetLoadResult(
            records=[],
            requested_size=sample_size,
            selected_size=0,
            candidate_size=0,
            source="disabled",
            warnings=warnings,
        )

    if not synthetic_dataset_path.exists():
        warnings.append(f"Synthetic dataset not found: {synthetic_dataset_path}")
        return DatasetLoadResult(
            records=[],
            requested_size=sample_size,
            selected_size=0,
            candidate_size=0,
            source=str(synthetic_dataset_path),
            warnings=warnings,
        )

    raw_rows = _load_jsonl_records(synthetic_dataset_path)
    candidates: list[AblationExample] = []
    for index, row in enumerate(raw_rows, start=1):
        question = str(row.get("question", "")).strip()
        schema = row.get("schema")
        if not question or not isinstance(schema, dict):
            continue

        normalized_schema: dict[str, dict[str, str]] = {}
        for table, columns in schema.items():
            if not isinstance(table, str) or not isinstance(columns, dict):
                continue
            normalized_schema[table] = {
                str(column): str(column_type)
                for column, column_type in columns.items()
                if isinstance(column, str)
            }

        if not normalized_schema:
            continue

        record_id = str(row.get("id", "")).strip() or f"synthetic:{index:05d}"
        candidates.append(
            AblationExample(
                example_id=f"synthetic:{record_id}",
                dataset="synthetic",
                question=question,
                schema=normalized_schema,
                source=str(synthetic_dataset_path.name),
                reference_sql=str(row.get("sql", "")).strip() or None,
            )
        )

    selected = _sample_deterministically(candidates, sample_size, seed)
    return DatasetLoadResult(
        records=selected,
        requested_size=sample_size,
        selected_size=len(selected),
        candidate_size=len(candidates),
        source=str(synthetic_dataset_path),
        warnings=warnings,
    )


def _pick_table_and_columns(schema: dict[str, dict[str, str]]) -> tuple[str, dict[str, str]]:
    if not schema:
        return "unknown_table", {}
    table_name = sorted(schema.keys())[0]
    columns = schema.get(table_name, {})
    return table_name, dict(sorted(columns.items(), key=lambda item: item[0]))


def _first_column_by_domain(columns: dict[str, str], domain: str) -> str | None:
    for column, column_type in columns.items():
        if _type_domain(column_type) == domain:
            return column
    return None


def _mock_sql(example: AblationExample, variant: PromptVariant) -> str:
    table_name, columns = _pick_table_and_columns(example.schema)
    if not columns:
        return "SELECT 1;"

    text_column = _first_column_by_domain(columns, "text") or next(iter(columns))
    numeric_column = _first_column_by_domain(columns, "numeric")
    boolean_column = _first_column_by_domain(columns, "boolean")
    date_column = _first_column_by_domain(columns, "date")
    any_column = next(iter(columns))
    alias = "t"

    if variant.prompt_id == "baseline_nl2sql":
        mode = _stable_pick(f"{example.example_id}:{example.question}", 3)
        if mode == 0:
            if numeric_column:
                return (
                    f"SELECT {alias}.{text_column} FROM {table_name} {alias} "
                    f"WHERE {alias}.{numeric_column} = 7;"
                )
            return (
                f"SELECT {alias}.{text_column} FROM {table_name} {alias} "
                "WHERE "
                f"{alias}.{text_column} = 'sample';"
            )
        if mode == 1 and numeric_column:
            return (
                f"SELECT {alias}.{text_column} FROM {table_name} {alias} "
                f"WHERE {alias}.{numeric_column} = 'high';"
            )
        fabricated_column = f"fabricated_{_stable_pick(example.example_id, 97) + 1}"
        return f"SELECT {alias}.{fabricated_column} FROM {table_name} {alias};"

    if variant.prompt_id == "constraint_explicit":
        filter_column = numeric_column or boolean_column or date_column or text_column or any_column
        return (
            f"SELECT {alias}.{text_column} FROM {table_name} {alias} "
            f"WHERE {alias}.{filter_column} = :{filter_column}_value;"
        )

    if numeric_column:
        return (
            f"SELECT {alias}.{text_column} FROM {table_name} {alias} "
            f"WHERE {alias}.{numeric_column} >= :min_{numeric_column} "
            f"ORDER BY {alias}.{numeric_column} DESC;"
        )
    if date_column:
        return (
            f"SELECT {alias}.{text_column} FROM {table_name} {alias} "
            f"WHERE {alias}.{date_column} >= :start_{date_column};"
        )
    return (
        f"SELECT {alias}.{text_column} FROM {table_name} {alias} "
        f"WHERE {alias}.{any_column} = :{any_column}_value;"
    )


def _empty_variant_bucket() -> dict[str, Any]:
    return {
        "examples_evaluated": 0,
        "any_violation": {"count": 0, "rate": 0.0},
        "metrics": {
            metric: {"detected": 0, "detection_rate": 0.0, "total": 0}
            for metric in CORE_METRICS
        },
        "by_dataset": {},
    }


def _ensure_dataset_bucket(container: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    dataset_bucket = container["by_dataset"].setdefault(
        dataset_name,
        {
            "examples_evaluated": 0,
            "any_violation": {"count": 0, "rate": 0.0},
            "metrics": {
                metric: {"detected": 0, "detection_rate": 0.0, "total": 0}
                for metric in CORE_METRICS
            },
        },
    )
    return dataset_bucket


def _finalize_bucket(bucket: dict[str, Any]) -> None:
    total = bucket["examples_evaluated"]
    bucket["any_violation"]["rate"] = (
        bucket["any_violation"]["count"] / total if total else 0.0
    )
    for metric in CORE_METRICS:
        metric_stats = bucket["metrics"][metric]
        metric_total = metric_stats["total"]
        metric_stats["detection_rate"] = (
            metric_stats["detected"] / metric_total if metric_total else 0.0
        )


def _record_metrics(
    container: dict[str, Any],
    dataset_name: str,
    flags: dict[str, bool],
) -> None:
    any_violation = any(flags.values())

    container["examples_evaluated"] += 1
    container["any_violation"]["count"] += int(any_violation)
    for metric in CORE_METRICS:
        container["metrics"][metric]["total"] += 1
        container["metrics"][metric]["detected"] += int(flags[metric])

    dataset_bucket = _ensure_dataset_bucket(container, dataset_name)
    dataset_bucket["examples_evaluated"] += 1
    dataset_bucket["any_violation"]["count"] += int(any_violation)
    for metric in CORE_METRICS:
        dataset_bucket["metrics"][metric]["total"] += 1
        dataset_bucket["metrics"][metric]["detected"] += int(flags[metric])


def _resolve_examples(
    spider_dataset: DatasetLoadResult,
    synthetic_dataset: DatasetLoadResult,
) -> list[AblationExample]:
    return [*spider_dataset.records, *synthetic_dataset.records]


def _bounded_errors(errors: list[dict[str, str]], new_error: dict[str, str], limit: int = 20) -> None:
    if len(errors) < limit:
        errors.append(new_error)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 prompt-engineering ablation for semantic bypass metrics."
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "anthropic", "ollama", "openrouter", "mock"],
        default="openrouter",
        help="LLM provider to use for SQL generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic dataset sampling.",
    )
    parser.add_argument(
        "--spider-dir",
        type=Path,
        default=ROOT.parent / "spider_data",
        help="Path to local Spider dataset directory.",
    )
    parser.add_argument(
        "--spider-subset",
        type=Path,
        default=ROOT / "data" / "spider_subset" / "phase2_spider_subset.jsonl",
        help="Preferred Spider subset JSONL (falls back to raw Spider files if missing).",
    )
    parser.add_argument(
        "--spider-sample-size",
        type=int,
        default=50,
        help="Number of Spider examples to include.",
    )
    parser.add_argument(
        "--synthetic-dataset",
        type=Path,
        default=ROOT / "data" / "synthetic" / "phase1_synthetic.jsonl",
        help="Optional synthetic JSONL for additional coverage.",
    )
    parser.add_argument(
        "--synthetic-sample-size",
        type=int,
        default=30,
        help="Number of synthetic examples to include (0 disables synthetic sampling).",
    )
    parser.add_argument(
        "--sample-predictions",
        type=int,
        default=5,
        help="Number of prediction samples to keep per prompt variant in output JSON.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=ROOT / "results" / "phase3_prompting.json",
        help="Output JSON path for phase-3 ablation results.",
    )
    parser.add_argument(
        "--no-fallback-to-mock",
        action="store_true",
        help="Disable fallback to deterministic mock SQL when API calls fail.",
    )
    args = parser.parse_args()

    log_path = get_run_log_path(phase="phase3")
    logger = setup_logger("phase3", log_file=log_path)

    log_section(logger, "Phase 3: Prompt Ablation")
    log_dict(logger, {
        "provider": args.provider,
        "seed": args.seed,
        "spider_sample_size": args.spider_sample_size,
        "synthetic_sample_size": args.synthetic_sample_size,
    })
    log_line(logger, f"Log output: {log_path}")

    schemas_by_db, schema_warnings = load_spider_schemas(args.spider_dir)
    log_dict(logger, {"schemas_loaded": len(schemas_by_db)})

    spider_dataset = _load_spider_dataset(
        spider_subset_path=args.spider_subset,
        spider_dir=args.spider_dir,
        schemas_by_db=schemas_by_db,
        sample_size=args.spider_sample_size,
        seed=args.seed,
    )
    synthetic_dataset = _load_synthetic_dataset(
        synthetic_dataset_path=args.synthetic_dataset,
        sample_size=args.synthetic_sample_size,
        seed=args.seed + 1,
    )

    examples = _resolve_examples(spider_dataset, synthetic_dataset)
    if not examples:
        raise RuntimeError(
            "No ablation examples available after loading Spider/synthetic sources."
        )
    log_dict(logger, {"total_examples": len(examples)})

    llm_client = LLMClient(config=LLMConfig(provider=args.provider))
    resolved_provider = llm_client.resolved_provider()
    fallback_to_mock = not args.no_fallback_to_mock
    suite = ConstraintSuite()

    log_line(logger, f"Running prompt ablation with {len(list_prompt_variants())} variants on {len(examples)} examples...")

    variant_results: dict[str, dict[str, Any]] = {
        variant.prompt_id: _empty_variant_bucket() for variant in list_prompt_variants()
    }
    sample_predictions: dict[str, list[dict[str, Any]]] = {
        variant.prompt_id: [] for variant in list_prompt_variants()
    }
    provider_fallback_errors: list[dict[str, str]] = []

    for variant in list_prompt_variants():
        variant_bucket = variant_results[variant.prompt_id]

        for example in examples:
            schema_hint = render_schema_hint(example.schema)
            user_prompt = variant.build_user_prompt(question=example.question, schema_hint=schema_hint)
            provider_used = resolved_provider

            logger.debug(f"[INFERENCE] example_id={example.example_id}, variant={variant.prompt_id}, question={example.question[:50]}...")

            if resolved_provider == "mock":
                generated_sql = _mock_sql(example, variant)
            else:
                try:
                    generated_sql = llm_client.generate_sql(
                        question=example.question,
                        schema_hint=schema_hint,
                        system_prompt=variant.system_prompt,
                        user_prompt=user_prompt,
                    )
                    logger.debug(f"[INFERENCE] generated_sql={generated_sql[:100]}...")
                except Exception as error:
                    if not fallback_to_mock:
                        raise
                    generated_sql = _mock_sql(example, variant)
                    provider_used = "mock_fallback"
                    _bounded_errors(
                        provider_fallback_errors,
                        {
                            "example_id": example.example_id,
                            "variant": variant.prompt_id,
                            "error": str(error)[:300],
                        },
                    )

            checks = suite.evaluate(generated_sql, SchemaCatalog.from_mapping(example.schema))
            core_flags = {metric: checks[metric].detected for metric in CORE_METRICS}
            logger.debug(f"[RESULT] core_flags={core_flags}")
            _record_metrics(variant_bucket, example.dataset, core_flags)

            if len(sample_predictions[variant.prompt_id]) < max(args.sample_predictions, 0):
                sample_predictions[variant.prompt_id].append(
                    {
                        "example_id": example.example_id,
                        "dataset": example.dataset,
                        "provider_used": provider_used,
                        "question": example.question,
                        "generated_sql": generated_sql,
                        "reference_sql": example.reference_sql,
                        "core_flags": core_flags,
                    }
                )

    for bucket in variant_results.values():
        _finalize_bucket(bucket)
        for dataset_bucket in bucket["by_dataset"].values():
            _finalize_bucket(dataset_bucket)

    dataset_warnings = [
        *schema_warnings,
        *spider_dataset.warnings,
        *synthetic_dataset.warnings,
    ]

    output: dict[str, Any] = {
        "phase": "phase3_prompting",
        "json_schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "summary": "Prompt ablation using checker-detected SHR/POR/DIVR rates only.",
            "guardrail": (
                "Results are detector-based and scoped to sampled Spider/synthetic inputs. "
                "They should not be interpreted as execution accuracy or causal proof."
            ),
        },
        "config": {
            "seed": args.seed,
            "provider_requested": args.provider,
            "provider_resolved": resolved_provider,
            "fallback_to_mock_on_error": fallback_to_mock,
            "spider_dir": str(args.spider_dir),
            "spider_subset": str(args.spider_subset),
            "spider_sample_size": args.spider_sample_size,
            "synthetic_dataset": str(args.synthetic_dataset),
            "synthetic_sample_size": args.synthetic_sample_size,
            "sample_predictions": args.sample_predictions,
        },
        "prompt_variants": [
            {
                "prompt_id": variant.prompt_id,
                "title": variant.title,
                "description": variant.description,
                "system_prompt": variant.system_prompt,
                "user_instruction": variant.user_instruction,
            }
            for variant in list_prompt_variants()
        ],
        "dataset": {
            "selected_total": len(examples),
            "components": {
                "spider": {
                    "requested": spider_dataset.requested_size,
                    "selected": spider_dataset.selected_size,
                    "candidates": spider_dataset.candidate_size,
                    "source": spider_dataset.source,
                },
                "synthetic": {
                    "requested": synthetic_dataset.requested_size,
                    "selected": synthetic_dataset.selected_size,
                    "candidates": synthetic_dataset.candidate_size,
                    "source": synthetic_dataset.source,
                },
            },
            "warnings": dataset_warnings,
        },
        "provider": {
            "resolved": resolved_provider,
            "fallback_error_count": len(provider_fallback_errors),
            "fallback_errors": provider_fallback_errors,
        },
        "analysis": {
            "mandatory_metrics": list(CORE_METRICS),
            "variant_results": variant_results,
        },
        "samples": {
            "max_per_variant": max(args.sample_predictions, 0),
            "predictions": sample_predictions,
        },
    }

    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    args.results_output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")

    log_section(logger, "Results Summary")
    log_dict(logger, {
        "total_examples": len(examples),
        "provider": resolved_provider,
        "fallback_errors": len(provider_fallback_errors),
    })
    log_line(logger, "variant | any_violation | SHR | POR | DIVR")
    for variant in list_prompt_variants():
        metrics = variant_results[variant.prompt_id]
        line = (
            f"{variant.prompt_id} | "
            f"{metrics['any_violation']['rate']:.3f} | "
            f"{metrics['metrics']['SHR']['detection_rate']:.3f} | "
            f"{metrics['metrics']['POR']['detection_rate']:.3f} | "
            f"{metrics['metrics']['DIVR']['detection_rate']:.3f}"
        )
        log_line(logger, line)
    log_line(logger, f"Results saved to {args.results_output}")
    logger.info("Phase 3 complete")

    print(f"Phase 3 prompting examples: {len(examples)}")
    print(f"Resolved provider: {resolved_provider}")
    print("variant\tany_violation\tSHR\tPOR\tDIVR")
    for variant in list_prompt_variants():
        metrics = variant_results[variant.prompt_id]
        print(
            f"{variant.prompt_id}\t"
            f"{metrics['any_violation']['rate']:.3f}\t"
            f"{metrics['metrics']['SHR']['detection_rate']:.3f}\t"
            f"{metrics['metrics']['POR']['detection_rate']:.3f}\t"
            f"{metrics['metrics']['DIVR']['detection_rate']:.3f}"
        )
    print(f"Fallback errors: {len(provider_fallback_errors)}")
    print(f"Phase 3 results: {args.results_output}")


if __name__ == "__main__":
    main()
