from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
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
from semantic_bypass.pipeline import NL2SQLPipeline
from semantic_bypass.scale_tiers import (
    DeterministicTierProxyClient,
    ModelTierConfig,
    TIER_ORDER,
    build_tier_llm_client,
    load_model_tier_configs,
    provider_available,
)
from semantic_bypass.spider_utils import (
    create_stratified_subset,
    load_spider_examples,
    load_spider_schemas,
)

ALL_METRICS = ("SHR", "POR", "DIVR", "SJR", "FDVR")
CORE_METRICS = ("SHR", "POR", "DIVR")


@dataclass
class TierRuntime:
    config: ModelTierConfig
    client: Any
    provider_used: str
    is_proxy: bool
    note: str | None = None


def _empty_metric_summary() -> dict[str, Any]:
    return {
        "total": 0,
        "detected": 0,
        "detection_rate": 0.0,
        "implemented_count": 0,
        "not_implemented_count": 0,
        "notes": [],
    }


def _load_subset_jsonl(subset_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    records: list[dict[str, Any]] = []
    with subset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                warnings.append(f"Skipped invalid JSONL row at line {line_number} in {subset_path.name}")
                continue
            if not isinstance(payload, dict):
                warnings.append(f"Skipped non-object row at line {line_number} in {subset_path.name}")
                continue
            db_id = str(payload.get("db_id", "")).strip()
            if not db_id:
                warnings.append(f"Skipped row missing db_id at line {line_number} in {subset_path.name}")
                continue
            records.append(payload)
    return records, warnings


def _build_runtime(config: ModelTierConfig, seed: int) -> TierRuntime:
    provider = config.provider.strip().lower()
    if provider == "mock":
        return TierRuntime(
            config=config,
            client=DeterministicTierProxyClient(config=config, seed=seed),
            provider_used="proxy",
            is_proxy=True,
            note="Provider explicitly set to mock; running deterministic proxy tier simulation.",
        )

    if not provider_available(provider):
        return TierRuntime(
            config=config,
            client=DeterministicTierProxyClient(config=config, seed=seed),
            provider_used="proxy",
            is_proxy=True,
            note=(
                f"Provider '{provider}' unavailable (missing credentials or unsupported provider); "
                "using deterministic proxy tier simulation."
            ),
        )

    return TierRuntime(
        config=config,
        client=build_tier_llm_client(config),
        provider_used=provider,
        is_proxy=False,
    )


def _fallback_runtime(runtime: TierRuntime, seed: int, reason: str) -> TierRuntime:
    return TierRuntime(
        config=runtime.config,
        client=DeterministicTierProxyClient(config=runtime.config, seed=seed),
        provider_used="proxy",
        is_proxy=True,
        note=f"Fell back to deterministic proxy tier simulation after runtime error: {reason}",
    )


def _select_records(records: list[dict[str, Any]], max_examples: int, seed: int) -> list[dict[str, Any]]:
    normalized = sorted(records, key=lambda row: str(row.get("example_id", "")))
    rng = random.Random(seed)
    rng.shuffle(normalized)
    if max_examples <= 0:
        return normalized
    return normalized[: min(max_examples, len(normalized))]


def _pearson_correlation(values_x: list[float], values_y: list[float]) -> float:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return 0.0
    x_mean = sum(values_x) / len(values_x)
    y_mean = sum(values_y) / len(values_y)
    centered_x = [x - x_mean for x in values_x]
    centered_y = [y - y_mean for y in values_y]
    numerator = sum(x * y for x, y in zip(centered_x, centered_y))
    denominator_x = sum(x * x for x in centered_x) ** 0.5
    denominator_y = sum(y * y for y in centered_y) ** 0.5
    if denominator_x == 0 or denominator_y == 0:
        return 0.0
    return numerator / (denominator_x * denominator_y)


def _slope(values_x: list[float], values_y: list[float]) -> float:
    if len(values_x) != len(values_y) or len(values_x) < 2:
        return 0.0
    x_mean = sum(values_x) / len(values_x)
    y_mean = sum(values_y) / len(values_y)
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(values_x, values_y))
    variance = sum((x - x_mean) ** 2 for x in values_x)
    if variance == 0:
        return 0.0
    return covariance / variance


def _trend_label(values: list[float]) -> str:
    if len(values) < 2:
        return "insufficient-data"
    if all(a > b for a, b in zip(values, values[1:])):
        return "decreasing"
    if all(a < b for a, b in zip(values, values[1:])):
        return "increasing"
    if all(a == values[0] for a in values):
        return "flat"
    return "non-monotonic"


def _collect_optional_status(metric_summary: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for metric in ("SJR", "FDVR"):
        metric_data = metric_summary[metric]
        implemented = metric_data["implemented_count"] > 0
        notes = [note for note in metric_data["notes"] if note]
        status[metric] = {
            "implemented": implemented,
            "note_samples": notes[:3],
            "not_implemented_count": metric_data["not_implemented_count"],
            "total": metric_data["total"],
        }
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 model-scale checker analysis.")
    parser.add_argument(
        "--spider-dir",
        type=Path,
        default=ROOT.parent / "spider_data",
        help="Path to local Spider dataset directory.",
    )
    parser.add_argument(
        "--subset-jsonl",
        type=Path,
        default=ROOT / "data" / "spider_subset" / "phase2_spider_subset.jsonl",
        help="Preferred Spider subset artifact (JSONL) from Phase 2.",
    )
    parser.add_argument(
        "--subset-size-fallback",
        type=int,
        default=300,
        help="Fallback subset size when the preferred artifact is unavailable.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=180,
        help="Maximum examples to evaluate per tier.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic record selection and proxy generation.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "anthropic", "ollama", "openrouter", "mock"],
        default=None,
        help="Optional provider override for all tiers.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=ROOT / "results" / "phase4_scale.json",
        help="Output JSON file for Phase 4 scale results.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="Number of prediction samples to keep per tier.",
    )
    args = parser.parse_args()

    log_path = get_run_log_path(phase="phase4")
    logger = setup_logger("phase4", log_file=log_path)

    log_section(logger, "Phase 4: Scale Analysis")
    log_dict(logger, {
        "seed": args.seed,
        "max_examples": args.max_examples,
        "provider": args.provider,
    })
    log_line(logger, f"Log output: {log_path}")

    schemas_by_db, schema_warnings = load_spider_schemas(args.spider_dir)
    if not schemas_by_db:
        raise RuntimeError(
            f"No Spider schemas available in {args.spider_dir}. Warnings: {schema_warnings}"
        )
    log_dict(logger, {"schemas_loaded": len(schemas_by_db)})

    subset_source = "phase2_artifact"
    subset_warnings: list[str] = []
    if args.subset_jsonl.exists():
        source_records, subset_warnings = _load_subset_jsonl(args.subset_jsonl)
    else:
        subset_source = "generated_fallback_subset"
        examples, _, example_warnings = load_spider_examples(args.spider_dir)
        subset_warnings.extend(example_warnings)
        examples_with_schema = [example for example in examples if example.db_id in schemas_by_db]
        fallback_subset = create_stratified_subset(
            examples=examples_with_schema,
            sample_size=args.subset_size_fallback,
            seed=args.seed,
        )
        source_records = [example.to_record() for example in fallback_subset]
        subset_warnings.append(
            "Preferred phase2 subset artifact was unavailable; generated deterministic fallback subset."
        )

    if not source_records:
        raise RuntimeError("No subset records available for Phase 4 scale analysis.")

    selected_records = _select_records(source_records, max_examples=args.max_examples, seed=args.seed)
    if not selected_records:
        raise RuntimeError("Record selection returned zero examples.")
    log_dict(logger, {"selected_records": len(selected_records), "subset_source": subset_source})

    tier_configs = load_model_tier_configs(provider_override=args.provider)
    log_line(logger, f"Running scale analysis across {len(tier_configs)} tiers...")
    tiers_output: dict[str, Any] = {}
    tier_rates: dict[str, dict[str, float]] = {}
    runtime_notes: list[dict[str, str]] = []

    for tier_config in tier_configs:
        runtime = _build_runtime(tier_config, seed=args.seed)
        pipeline = NL2SQLPipeline(llm_client=runtime.client)

        metric_summary = {metric: _empty_metric_summary() for metric in ALL_METRICS}
        evaluated_examples = 0
        skipped_missing_schema = 0
        error_count = 0
        sample_predictions: list[dict[str, Any]] = []
        core_any_detected = 0

        for row in selected_records:
            db_id = str(row.get("db_id", "")).strip()
            schema_mapping = schemas_by_db.get(db_id)
            if schema_mapping is None:
                skipped_missing_schema += 1
                continue

            question = str(row.get("question", "")).strip()
            if not question:
                question = f"List sample rows from {db_id}."

            try:
                run = pipeline.run(question=question, schema=schema_mapping)
            except Exception as error:
                if not runtime.is_proxy:
                    runtime = _fallback_runtime(runtime, seed=args.seed, reason=str(error)[:300])
                    runtime_notes.append({"tier": tier_config.tier, "note": runtime.note or ""})
                    pipeline = NL2SQLPipeline(llm_client=runtime.client)
                    run = pipeline.run(question=question, schema=schema_mapping)
                else:
                    error_count += 1
                    continue

            evaluated_examples += 1
            if any(run.core_flags[metric] for metric in CORE_METRICS):
                core_any_detected += 1

            for metric, result in run.checks.items():
                if metric not in metric_summary:
                    continue
                metric_counts = metric_summary[metric]
                metric_counts["total"] += 1
                metric_counts["detected"] += int(result.detected)
                if result.implemented:
                    metric_counts["implemented_count"] += 1
                else:
                    metric_counts["not_implemented_count"] += 1
                    if result.notes:
                        metric_counts["notes"].append(result.notes)

            if len(sample_predictions) < args.sample_limit:
                sample_predictions.append(
                    {
                        "example_id": row.get("example_id"),
                        "db_id": db_id,
                        "question": question,
                        "generated_sql": run.sql,
                        "core_flags": run.core_flags,
                    }
                )

        for metric in ALL_METRICS:
            metric_counts = metric_summary[metric]
            total = metric_counts["total"]
            metric_counts["detection_rate"] = (metric_counts["detected"] / total) if total else 0.0
            metric_counts["notes"] = sorted(set(metric_counts["notes"]))[:5]

        tier_rates[tier_config.tier] = {
            metric: metric_summary[metric]["detection_rate"] for metric in CORE_METRICS
        }

        if runtime.note:
            runtime_notes.append({"tier": tier_config.tier, "note": runtime.note})

        tiers_output[tier_config.tier] = {
            "descriptor": tier_config.descriptor,
            "provider_requested": tier_config.provider,
            "provider_used": runtime.provider_used,
            "is_proxy": runtime.is_proxy,
            "runtime_note": runtime.note,
            "evaluated_examples": evaluated_examples,
            "skipped_missing_schema_count": skipped_missing_schema,
            "generation_error_count": error_count,
            "metrics": metric_summary,
            "mandatory_metrics": {metric: metric_summary[metric] for metric in CORE_METRICS},
            "optional_metrics_status": _collect_optional_status(metric_summary),
            "any_core_violation_rate": (core_any_detected / evaluated_examples) if evaluated_examples else 0.0,
            "sample_predictions": sample_predictions,
        }

    x_values = [float(index) for index, _ in enumerate(TIER_ORDER, start=1)]
    trend_summary: dict[str, Any] = {
        "tier_order": list(TIER_ORDER),
        "core_metric_rates_by_tier": {},
        "correlation": {},
    }

    for metric in CORE_METRICS:
        per_tier_rates = {tier: tier_rates.get(tier, {}).get(metric, 0.0) for tier in TIER_ORDER}
        ordered_rates = [per_tier_rates[tier] for tier in TIER_ORDER]
        trend_summary["core_metric_rates_by_tier"][metric] = per_tier_rates
        trend_summary["correlation"][metric] = {
            "pearson": _pearson_correlation(x_values, ordered_rates),
            "slope_per_tier_step": _slope(x_values, ordered_rates),
            "trend": _trend_label(ordered_rates),
        }

    proxy_tier_count = sum(1 for tier in tiers_output.values() if tier["is_proxy"])
    if proxy_tier_count == len(tiers_output):
        execution_mode = "proxy-only"
    elif proxy_tier_count == 0:
        execution_mode = "provider-only"
    else:
        execution_mode = "mixed"

    results: dict[str, Any] = {
        "phase": "phase4_scale",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "seed": args.seed,
            "execution_mode": execution_mode,
            "proxy_annotation": (
                "Deterministic proxy tier simulation was used for one or more tiers; "
                "treat rates as local proxy diagnostics, not real model leaderboard claims."
            ),
            "claims_conservative_note": (
                "This phase estimates checker-sensitive scale trends under available local execution only. "
                "When provider/model access is missing, deterministic proxy behavior is substituted."
            ),
            "spider_dir": str(args.spider_dir),
            "subset_source": subset_source,
            "subset_jsonl": str(args.subset_jsonl),
            "selected_example_count": len(selected_records),
            "subset_warning_count": len(schema_warnings + subset_warnings),
        },
        "inputs": {
            "schema_warnings": schema_warnings,
            "subset_warnings": subset_warnings,
        },
        "tiers": tiers_output,
        "trend_summary": trend_summary,
        "runtime_notes": runtime_notes,
    }

    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    args.results_output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    log_section(logger, "Results Summary")
    log_dict(logger, {"execution_mode": execution_mode, "tier_count": len(tiers_output)})
    log_line(logger, "metric | small | medium | large")
    for metric in CORE_METRICS:
        per_tier = trend_summary["core_metric_rates_by_tier"][metric]
        line = f"{metric} | {per_tier['small']:.3f} | {per_tier['medium']:.3f} | {per_tier['large']:.3f}"
        log_line(logger, line)
    log_line(logger, f"Results saved to {args.results_output}")
    logger.info("Phase 4 complete")

    print(f"Subset source: {subset_source}")
    print(f"Selected examples: {len(selected_records)}")
    print(f"Execution mode: {execution_mode}")
    for metric in CORE_METRICS:
        per_tier = trend_summary["core_metric_rates_by_tier"][metric]
        print(
            f"{metric}: "
            f"small={per_tier['small']:.3f}, medium={per_tier['medium']:.3f}, large={per_tier['large']:.3f}"
        )
    print(f"Saved Phase 4 results to {args.results_output}")


if __name__ == "__main__":
    main()
