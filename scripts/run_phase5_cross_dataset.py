from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_bypass import (
    get_run_log_path,
    log_dict,
    log_line,
    log_section,
    setup_logger,
)
from semantic_bypass.cross_dataset import (
    CORE_METRICS,
    DatasetRecord,
    deterministic_sample,
    evaluate_records,
    load_rows,
    normalize_schema,
    write_records_jsonl,
)
from semantic_bypass.spider_utils import load_spider_schemas


@dataclass
class DatasetEntry:
    dataset_id: str
    title: str
    availability: dict[str, Any]
    analysis: dict[str, Any] | None


def _missing_dataset_entry(
    dataset_id: str,
    title: str,
    reason: str,
    *,
    source_paths: list[str],
    warnings: list[str] | None = None,
    proxy_kind: str = "placeholder",
) -> DatasetEntry:
    return DatasetEntry(
        dataset_id=dataset_id,
        title=title,
        availability={
            "status": "missing",
            "available_locally": False,
            "uses_proxy": True,
            "proxy_kind": proxy_kind,
            "reason": reason,
            "source_paths": source_paths,
            "warnings": warnings or [],
        },
        analysis=None,
    )



def _analyzed_dataset_entry(
    dataset_id: str,
    title: str,
    records: list[DatasetRecord],
    *,
    seed: int,
    max_examples: int,
    artifact_path: Path,
    source_paths: list[str],
    warnings: list[str],
    claim_scope: str,
    sample_limit: int,
    availability_status: str = "analyzed",
    uses_proxy: bool = False,
) -> DatasetEntry:
    selected = deterministic_sample(records, max_examples=max_examples, seed=seed)
    write_records_jsonl(artifact_path, selected)
    analysis = evaluate_records(selected, sample_limit=sample_limit)

    return DatasetEntry(
        dataset_id=dataset_id,
        title=title,
        availability={
            "status": availability_status,
            "available_locally": True,
            "uses_proxy": uses_proxy,
            "source_paths": source_paths,
            "candidate_examples": len(records),
            "selected_examples": len(selected),
            "artifact_jsonl": str(artifact_path),
            "warnings": warnings,
        },
        analysis={
            **analysis,
            "claim_scope": claim_scope,
        },
    )



def _sql_from_row(row: dict[str, Any]) -> str:
    sql = row.get("sql")
    if isinstance(sql, str) and sql.strip():
        return sql.strip()
    query = row.get("query")
    if isinstance(query, str) and query.strip():
        return query.strip()
    return ""



def _question_from_row(row: dict[str, Any], fallback: str) -> str:
    question = row.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    return fallback



def _discover_synthetic(
    synthetic_path: Path,
    *,
    seed: int,
    max_examples: int,
    artifact_root: Path,
    sample_limit: int,
) -> DatasetEntry:
    if not synthetic_path.exists():
        return _missing_dataset_entry(
            dataset_id="synthetic_suite",
            title="Synthetic suite",
            reason="Synthetic suite file not found.",
            source_paths=[str(synthetic_path)],
        )

    rows, warnings = load_rows(synthetic_path)
    records: list[DatasetRecord] = []
    for index, row in enumerate(rows, start=1):
        sql = _sql_from_row(row)
        schema = normalize_schema(row.get("schema"))
        if not sql or not schema:
            continue

        example_id = str(row.get("id", "")).strip() or f"synthetic:{index:05d}"
        records.append(
            DatasetRecord(
                example_id=example_id,
                question=_question_from_row(row, fallback=f"Synthetic example {index}"),
                sql=sql,
                schema=schema,
                source=synthetic_path.name,
                metadata={"target_metric": row.get("target_metric")},
            )
        )

    if not records:
        return _missing_dataset_entry(
            dataset_id="synthetic_suite",
            title="Synthetic suite",
            reason="Synthetic suite exists but no usable records with sql+schema.",
            source_paths=[str(synthetic_path)],
            warnings=warnings,
            proxy_kind="invalid-local-data",
        )

    artifact_path = artifact_root / "synthetic_suite" / "phase5_synthetic_suite_subset.jsonl"
    return _analyzed_dataset_entry(
        dataset_id="synthetic_suite",
        title="Synthetic suite",
        records=records,
        seed=seed,
        max_examples=max_examples,
        artifact_path=artifact_path,
        source_paths=[str(synthetic_path)],
        warnings=warnings,
        claim_scope="Template-based synthetic records; detector-rate diagnostics only.",
        sample_limit=sample_limit,
    )



def _discover_spider_subset(
    spider_subset_path: Path,
    spider_dir: Path,
    *,
    seed: int,
    max_examples: int,
    artifact_root: Path,
    sample_limit: int,
) -> DatasetEntry:
    if not spider_subset_path.exists():
        return _missing_dataset_entry(
            dataset_id="spider_subset",
            title="Spider subset",
            reason="Spider subset JSONL not found.",
            source_paths=[str(spider_subset_path)],
        )

    rows, warnings = load_rows(spider_subset_path)
    schemas_by_db, schema_warnings = load_spider_schemas(spider_dir)
    warnings.extend(schema_warnings)

    if not schemas_by_db:
        return _missing_dataset_entry(
            dataset_id="spider_subset",
            title="Spider subset",
            reason=(
                "Spider schemas unavailable; unable to evaluate subset records safely."
            ),
            source_paths=[str(spider_subset_path), str(spider_dir)],
            warnings=warnings,
            proxy_kind="schema-missing",
        )

    records: list[DatasetRecord] = []
    skipped_missing_schema = 0
    for index, row in enumerate(rows, start=1):
        db_id = str(row.get("db_id", "")).strip()
        sql = _sql_from_row(row)
        if not db_id or not sql:
            continue
        schema = schemas_by_db.get(db_id)
        if schema is None:
            skipped_missing_schema += 1
            continue

        example_id = str(row.get("example_id", "")).strip() or f"spider:{index:05d}"
        records.append(
            DatasetRecord(
                example_id=example_id,
                question=_question_from_row(row, fallback=f"Spider subset row {index}"),
                sql=sql,
                schema=schema,
                source=str(row.get("source_file", spider_subset_path.name)),
                metadata={
                    "db_id": db_id,
                    "split": row.get("split"),
                },
            )
        )

    if skipped_missing_schema:
        warnings.append(f"Skipped {skipped_missing_schema} Spider rows due to missing db schema.")

    if not records:
        return _missing_dataset_entry(
            dataset_id="spider_subset",
            title="Spider subset",
            reason="Spider subset exists but no usable records after schema checks.",
            source_paths=[str(spider_subset_path), str(spider_dir)],
            warnings=warnings,
            proxy_kind="invalid-local-data",
        )

    artifact_path = artifact_root / "spider_subset" / "phase5_spider_subset_sample.jsonl"
    return _analyzed_dataset_entry(
        dataset_id="spider_subset",
        title="Spider subset",
        records=records,
        seed=seed,
        max_examples=max_examples,
        artifact_path=artifact_path,
        source_paths=[str(spider_subset_path), str(spider_dir)],
        warnings=warnings,
        claim_scope=(
            "Checker rates on local Spider subset SQL; these are not execution-accuracy metrics."
        ),
        sample_limit=sample_limit,
    )



def _normalize_sql_type(raw_type: str) -> str:
    normalized = raw_type.strip().lower()
    if normalized in {"integer", "bigint", "smallint", "numeric", "real", "double precision", "decimal"}:
        return "numeric"
    if normalized in {"boolean", "bool"}:
        return "boolean"
    if "time" in normalized or "date" in normalized:
        return "date"
    return "text"



def _discover_eicu_db_schema() -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    required = {
        "DB_NAME": os.getenv("DB_NAME", "").strip(),
        "DB_USER": os.getenv("DB_USER", "").strip(),
    }
    if not required["DB_NAME"] or not required["DB_USER"]:
        return {}, {
            "status": "credentials-missing",
            "reason": "DB_NAME/DB_USER are not configured for optional eICU DB discovery.",
        }

    try:
        import psycopg2  # type: ignore
    except Exception as error:  # pragma: no cover
        return {}, {
            "status": "driver-unavailable",
            "reason": f"psycopg2 unavailable for DB discovery: {error}",
        }

    host = os.getenv("DB_HOST", "localhost").strip() or "localhost"
    port = os.getenv("DB_PORT", "5432").strip() or "5432"
    db_name = required["DB_NAME"]
    db_user = required["DB_USER"]
    db_password = os.getenv("DB_PASSWORD", "")

    try:
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=db_name,
            user=db_user,
            password=db_password,
            connect_timeout=5,
        )
    except Exception as error:
        return {}, {
            "status": "connection-failed",
            "reason": f"DB connection failed during optional eICU discovery: {error.__class__.__name__}",
        }

    schema_mapping: dict[str, dict[str, str]] = {}
    table_refs: list[str] = []
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                  AND table_schema NOT IN ('pg_catalog', 'information_schema')
                  AND (
                    table_schema ILIKE '%eicu%'
                    OR table_name ILIKE 'eicu%'
                    OR table_name ILIKE '%icu%'
                    OR table_name ILIKE '%patient%'
                  )
                ORDER BY table_schema, table_name
                LIMIT 20
                """
            )
            table_rows = cursor.fetchall()

            for table_schema, table_name in table_rows:
                cursor.execute(
                    """
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    ORDER BY ordinal_position
                    LIMIT 40
                    """,
                    (table_schema, table_name),
                )
                column_rows = cursor.fetchall()
                columns = {
                    str(column_name): _normalize_sql_type(str(data_type))
                    for column_name, data_type in column_rows
                }
                if columns:
                    local_table_name = str(table_name)
                    schema_mapping[local_table_name] = columns
                    table_refs.append(f"{table_schema}.{table_name}")
    finally:
        connection.close()

    if not schema_mapping:
        return {}, {
            "status": "no-eicu-like-tables",
            "reason": "No eICU-like tables were detected in configured PostgreSQL database.",
            "table_refs": table_refs,
        }

    return schema_mapping, {
        "status": "discovered",
        "reason": "Discovered eICU-like tables in configured PostgreSQL database.",
        "table_refs": table_refs,
    }



def _build_schema_proxy_records(
    dataset_prefix: str,
    schema_mapping: dict[str, dict[str, str]],
) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for index, table_name in enumerate(sorted(schema_mapping), start=1):
        columns = schema_mapping[table_name]
        if not columns:
            continue
        column_names = sorted(columns)
        select_column = column_names[0]
        predicate_column = column_names[min(1, len(column_names) - 1)]

        sql = (
            f"SELECT t.{select_column} FROM {table_name} t "
            f"WHERE t.{predicate_column} = :{predicate_column}_value;"
        )
        records.append(
            DatasetRecord(
                example_id=f"{dataset_prefix}:proxy:{index:03d}",
                question=(
                    "Proxy query synthesized from discovered schema for checker smoke analysis."
                ),
                sql=sql,
                schema=schema_mapping,
                source="schema_proxy",
                metadata={"table": table_name, "mode": "schema_proxy"},
            )
        )
    return records



def _discover_eicu_dataset(
    eicu_dir: Path,
    *,
    seed: int,
    max_examples: int,
    artifact_root: Path,
    sample_limit: int,
    enable_db_discovery: bool,
) -> DatasetEntry:
    warnings: list[str] = []
    source_paths = [str(eicu_dir)]

    file_records: list[DatasetRecord] = []
    if eicu_dir.exists() and eicu_dir.is_dir():
        candidate_files = sorted(
            path
            for path in eicu_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jsonl", ".json"}
        )
        source_paths.extend(str(path) for path in candidate_files)
        for path in candidate_files:
            rows, row_warnings = load_rows(path)
            warnings.extend(row_warnings)
            for index, row in enumerate(rows, start=1):
                sql = _sql_from_row(row)
                schema = normalize_schema(row.get("schema"))
                if not sql or not schema:
                    continue
                file_records.append(
                    DatasetRecord(
                        example_id=str(row.get("id", "")).strip() or f"eicu:{path.stem}:{index:05d}",
                        question=_question_from_row(row, fallback=f"eICU record {index}"),
                        sql=sql,
                        schema=schema,
                        source=path.name,
                        metadata={"mode": "local_file"},
                    )
                )

    if file_records:
        artifact_path = artifact_root / "eicu_in_db" / "phase5_eicu_local_subset.jsonl"
        return _analyzed_dataset_entry(
            dataset_id="eicu_in_db",
            title="eICU-in-DB",
            records=file_records,
            seed=seed,
            max_examples=max_examples,
            artifact_path=artifact_path,
            source_paths=source_paths,
            warnings=warnings,
            claim_scope=(
                "Local eICU-like files with embedded schema. Detector rates only; not execution validation."
            ),
            sample_limit=sample_limit,
        )

    if not enable_db_discovery:
        return _missing_dataset_entry(
            dataset_id="eicu_in_db",
            title="eICU-in-DB",
            reason="No local eICU records found and DB discovery was disabled.",
            source_paths=source_paths,
            warnings=warnings,
            proxy_kind="disabled-db-discovery",
        )

    schema_mapping, db_meta = _discover_eicu_db_schema()
    warnings.append(f"DB discovery status: {db_meta.get('status')}")
    if db_meta.get("reason"):
        warnings.append(str(db_meta["reason"]))

    if not schema_mapping:
        return _missing_dataset_entry(
            dataset_id="eicu_in_db",
            title="eICU-in-DB",
            reason="No local eICU dataset rows discovered; DB schema probe did not yield analyzable schema.",
            source_paths=source_paths,
            warnings=warnings,
            proxy_kind="missing-local-and-db",
        )

    proxy_records = _build_schema_proxy_records("eicu", schema_mapping)
    if not proxy_records:
        return _missing_dataset_entry(
            dataset_id="eicu_in_db",
            title="eICU-in-DB",
            reason="eICU-like schema was discovered but could not generate safe proxy records.",
            source_paths=source_paths,
            warnings=warnings,
            proxy_kind="schema-without-records",
        )

    artifact_path = artifact_root / "eicu_in_db" / "phase5_eicu_db_proxy_subset.jsonl"
    return _analyzed_dataset_entry(
        dataset_id="eicu_in_db",
        title="eICU-in-DB",
        records=proxy_records,
        seed=seed,
        max_examples=max_examples,
        artifact_path=artifact_path,
        source_paths=source_paths,
        warnings=warnings,
        claim_scope=(
            "Proxy-only analysis from discovered DB schema (no local NL2SQL prediction set). "
            "Rates are diagnostic and must not be interpreted as benchmark performance."
        ),
        sample_limit=sample_limit,
        availability_status="proxy-analyzed",
        uses_proxy=True,
    )



def _discover_spider2_lite(
    spider2_dir: Path,
    *,
    seed: int,
    max_examples: int,
    artifact_root: Path,
    sample_limit: int,
) -> DatasetEntry:
    source_paths = [str(spider2_dir)]
    warnings: list[str] = []
    if not spider2_dir.exists() or not spider2_dir.is_dir():
        return _missing_dataset_entry(
            dataset_id="spider2_lite",
            title="Spider 2.0 Lite",
            reason="Spider 2.0 Lite directory not found locally.",
            source_paths=source_paths,
            proxy_kind="dataset-missing",
        )

    candidate_files = sorted(
        path
        for path in spider2_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jsonl", ".json"}
    )
    source_paths.extend(str(path) for path in candidate_files)

    tables_path = spider2_dir / "tables.json"
    schemas_by_db: dict[str, dict[str, dict[str, str]]] = {}
    if tables_path.exists():
        schemas_by_db, schema_warnings = load_spider_schemas(spider2_dir)
        warnings.extend(schema_warnings)

    records: list[DatasetRecord] = []
    for path in candidate_files:
        if path.name == "tables.json":
            continue
        rows, row_warnings = load_rows(path)
        warnings.extend(row_warnings)
        for index, row in enumerate(rows, start=1):
            sql = _sql_from_row(row)
            if not sql:
                continue

            row_schema = normalize_schema(row.get("schema"))
            db_id = str(row.get("db_id", "")).strip()
            schema = row_schema
            if not schema and db_id:
                schema = schemas_by_db.get(db_id, {})
            if not schema:
                continue

            example_id = str(row.get("id", "")).strip() or str(row.get("example_id", "")).strip()
            if not example_id:
                example_id = f"spider2:{path.stem}:{index:05d}"

            records.append(
                DatasetRecord(
                    example_id=example_id,
                    question=_question_from_row(row, fallback=f"Spider2Lite record {index}"),
                    sql=sql,
                    schema=schema,
                    source=path.name,
                    metadata={"db_id": db_id or None},
                )
            )

    if not records:
        return _missing_dataset_entry(
            dataset_id="spider2_lite",
            title="Spider 2.0 Lite",
            reason=(
                "Spider2Lite directory found but no usable sql+schema records were discovered."
            ),
            source_paths=source_paths,
            warnings=warnings,
            proxy_kind="invalid-or-incomplete-local-data",
        )

    artifact_path = artifact_root / "spider2_lite" / "phase5_spider2_lite_subset.jsonl"
    return _analyzed_dataset_entry(
        dataset_id="spider2_lite",
        title="Spider 2.0 Lite",
        records=records,
        seed=seed,
        max_examples=max_examples,
        artifact_path=artifact_path,
        source_paths=source_paths,
        warnings=warnings,
        claim_scope=(
            "Local Spider2Lite sample; detector rates only and conditioned on discovered local files."
        ),
        sample_limit=sample_limit,
    )



def _entry_to_output(entry: DatasetEntry) -> dict[str, Any]:
    return {
        "title": entry.title,
        "availability": entry.availability,
        "analysis": entry.analysis,
    }



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 5 cross-dataset checker analysis with explicit availability metadata."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic record selection.",
    )
    parser.add_argument(
        "--synthetic-path",
        type=Path,
        default=ROOT / "data" / "synthetic" / "phase1_synthetic.jsonl",
        help="Path to synthetic suite JSONL.",
    )
    parser.add_argument(
        "--spider-subset-path",
        type=Path,
        default=ROOT / "data" / "spider_subset" / "phase2_spider_subset.jsonl",
        help="Path to Spider subset JSONL.",
    )
    parser.add_argument(
        "--spider-dir",
        type=Path,
        default=ROOT.parent / "spider_data",
        help="Path to Spider root containing tables.json.",
    )
    parser.add_argument(
        "--eicu-dir",
        type=Path,
        default=ROOT / "data" / "eicu",
        help="Optional directory for local eICU dataset exports.",
    )
    parser.add_argument(
        "--spider2-lite-dir",
        type=Path,
        default=ROOT / "data" / "spider2_lite",
        help="Optional directory for local Spider2Lite files.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "data" / "cross_dataset",
        help="Directory for generated phase-5 subset artifacts.",
    )
    parser.add_argument(
        "--synthetic-max-examples",
        type=int,
        default=180,
        help="Max synthetic records to evaluate (negative means all).",
    )
    parser.add_argument(
        "--spider-max-examples",
        type=int,
        default=180,
        help="Max Spider subset records to evaluate (negative means all).",
    )
    parser.add_argument(
        "--eicu-max-examples",
        type=int,
        default=120,
        help="Max eICU records/proxy records to evaluate (negative means all).",
    )
    parser.add_argument(
        "--spider2-max-examples",
        type=int,
        default=120,
        help="Max Spider2Lite records to evaluate (negative means all).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="Number of per-dataset samples to retain in output JSON.",
    )
    parser.add_argument(
        "--disable-db-discovery",
        action="store_true",
        help="Disable optional eICU schema discovery via configured PostgreSQL connection.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=ROOT / "results" / "phase5_cross_dataset.json",
        help="Output JSON path for phase-5 results.",
    )
    args = parser.parse_args()

    log_path = get_run_log_path(phase="phase5")
    logger = setup_logger("phase5", log_file=log_path)

    log_section(logger, "Phase 5: Cross-Dataset Analysis")
    log_dict(logger, {
        "seed": args.seed,
        "synthetic_max": args.synthetic_max_examples,
        "spider_max": args.spider_max_examples,
    })
    log_line(logger, f"Log output: {log_path}")

    synthetic_entry = _discover_synthetic(
        args.synthetic_path,
        seed=args.seed,
        max_examples=args.synthetic_max_examples,
        artifact_root=args.artifact_dir,
        sample_limit=args.sample_limit,
    )
    spider_entry = _discover_spider_subset(
        args.spider_subset_path,
        args.spider_dir,
        seed=args.seed + 1,
        max_examples=args.spider_max_examples,
        artifact_root=args.artifact_dir,
        sample_limit=args.sample_limit,
    )
    eicu_entry = _discover_eicu_dataset(
        args.eicu_dir,
        seed=args.seed + 2,
        max_examples=args.eicu_max_examples,
        artifact_root=args.artifact_dir,
        sample_limit=args.sample_limit,
        enable_db_discovery=not args.disable_db_discovery,
    )
    spider2_entry = _discover_spider2_lite(
        args.spider2_lite_dir,
        seed=args.seed + 3,
        max_examples=args.spider2_max_examples,
        artifact_root=args.artifact_dir,
        sample_limit=args.sample_limit,
    )

    entries = [synthetic_entry, spider_entry, eicu_entry, spider2_entry]

    analyzed_ids = [
        entry.dataset_id
        for entry in entries
        if entry.analysis is not None and entry.availability.get("status") in {"analyzed", "proxy-analyzed"}
    ]
    missing_ids = [entry.dataset_id for entry in entries if entry.analysis is None]

    output = {
        "phase": "phase5_cross_dataset",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "core_metrics": list(CORE_METRICS),
        "scope_guardrails": [
            "Rates are checker detections over discovered local artifacts only.",
            "Missing datasets are represented as placeholders/proxy metadata without fabricated performance claims.",
            "Proxy-analyzed datasets are explicitly labeled and must not be interpreted as benchmark generalization evidence.",
        ],
        "config": {
            "synthetic_path": str(args.synthetic_path),
            "spider_subset_path": str(args.spider_subset_path),
            "spider_dir": str(args.spider_dir),
            "eicu_dir": str(args.eicu_dir),
            "spider2_lite_dir": str(args.spider2_lite_dir),
            "artifact_dir": str(args.artifact_dir),
            "synthetic_max_examples": args.synthetic_max_examples,
            "spider_max_examples": args.spider_max_examples,
            "eicu_max_examples": args.eicu_max_examples,
            "spider2_max_examples": args.spider2_max_examples,
            "sample_limit": args.sample_limit,
            "db_discovery_enabled": not args.disable_db_discovery,
        },
        "datasets": {
            entry.dataset_id: _entry_to_output(entry)
            for entry in entries
        },
        "summary": {
            "datasets_total": len(entries),
            "datasets_analyzed": len(analyzed_ids),
            "datasets_missing_or_placeholder": len(missing_ids),
            "analyzed_dataset_ids": analyzed_ids,
            "missing_dataset_ids": missing_ids,
        },
    }

    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    args.results_output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")

    log_section(logger, "Results Summary")
    log_dict(logger, {
        "datasets_analyzed": len(analyzed_ids),
        "datasets_missing": len(missing_ids),
    })
    log_line(logger, f"Analyzed: {', '.join(analyzed_ids) if analyzed_ids else 'none'}")
    log_line(logger, f"Missing: {', '.join(missing_ids) if missing_ids else 'none'}")
    log_line(logger, f"Results saved to {args.results_output}")
    logger.info("Phase 5 complete")

    print(f"Phase 5 results: {args.results_output}")
    print(f"Analyzed datasets: {', '.join(analyzed_ids) if analyzed_ids else 'none'}")
    print(f"Placeholder/missing datasets: {', '.join(missing_ids) if missing_ids else 'none'}")


if __name__ == "__main__":
    main()
