from __future__ import annotations

import argparse
import json
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
from semantic_bypass.checkers import ConstraintSuite
from semantic_bypass.schema import SchemaCatalog
from semantic_bypass.spider_utils import (
    create_stratified_subset,
    load_spider_examples,
    load_spider_schemas,
    summarize_subset,
    write_subset_artifact,
)

MANDATORY_METRICS = ("SHR", "POR", "DIVR")


def _empty_metric_summary() -> dict[str, Any]:
    return {
        "total": 0,
        "detected": 0,
        "detection_rate": 0.0,
        "implemented_count": 0,
        "not_implemented_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Spider subset extraction and Phase 2 checker analysis."
    )
    parser.add_argument(
        "--spider-dir",
        type=Path,
        default=ROOT.parent / "spider_data",
        help="Path to local Spider dataset directory.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=500,
        help="Target number of SQL examples for the stratified subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic subset creation.",
    )
    parser.add_argument(
        "--subset-output-dir",
        type=Path,
        default=ROOT / "data" / "spider_subset",
        help="Directory for subset artifact outputs.",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=ROOT / "results" / "phase2_spider.json",
        help="Output JSON file for Phase 2 analysis results.",
    )
    args = parser.parse_args()

    log_path = get_run_log_path(phase="phase2")
    logger = setup_logger("phase2", log_file=log_path)

    log_section(logger, "Phase 2: Spider Quantification")
    log_dict(logger, {"seed": args.seed, "subset_size": args.subset_size})
    log_line(logger, f"Log output: {log_path}")

    schemas_by_db, schema_warnings = load_spider_schemas(args.spider_dir)
    if not schemas_by_db:
        raise RuntimeError(
            f"No Spider schemas available in {args.spider_dir}. Warnings: {schema_warnings}"
        )
    log_dict(logger, {"schemas_loaded": len(schemas_by_db)})

    all_examples, source_counts, example_warnings = load_spider_examples(args.spider_dir)
    if not all_examples:
        raise RuntimeError(
            f"No Spider SQL examples available in {args.spider_dir}. Warnings: {example_warnings}"
        )
    log_dict(logger, {"examples_loaded": len(all_examples)})

    examples_with_schema = [example for example in all_examples if example.db_id in schemas_by_db]
    examples_without_schema = [example for example in all_examples if example.db_id not in schemas_by_db]
    if not examples_with_schema:
        raise RuntimeError("No Spider examples matched available schemas.")
    if examples_without_schema:
        missing_db_ids = sorted({example.db_id for example in examples_without_schema})
        example_warnings.append(
            "Skipped examples without schema definitions. "
            f"count={len(examples_without_schema)}, unique_db_ids={len(missing_db_ids)}"
        )
    log_dict(logger, {"examples_with_schema": len(examples_with_schema), "examples_without_schema": len(examples_without_schema)})

    log_line(logger, f"Creating stratified subset (seed={args.seed}, size={args.subset_size})...")
    subset = create_stratified_subset(
        examples=examples_with_schema,
        sample_size=args.subset_size,
        seed=args.seed,
    )
    if not subset:
        raise RuntimeError("Subset selection returned zero examples.")
    log_dict(logger, {"subset_size": len(subset)})

    log_line(logger, f"Writing subset artifacts to {args.subset_output_dir}")
    subset_artifacts = write_subset_artifact(
        subset,
        args.subset_output_dir,
        metadata={
            "seed": args.seed,
            "subset_size_requested": args.subset_size,
            "subset_size_selected": len(subset),
            "spider_dir": str(args.spider_dir),
        },
    )

    log_line(logger, "Running constraint checkers on subset...")
    suite = ConstraintSuite()
    metric_summary: dict[str, dict[str, Any]] = {}
    evaluated_examples = 0
    skipped_missing_schema: list[dict[str, str]] = []

    for example in subset:
        schema_mapping = schemas_by_db.get(example.db_id)
        if schema_mapping is None:
            skipped_missing_schema.append(
                {
                    "example_id": example.example_id,
                    "db_id": example.db_id,
                    "source_file": example.source_file,
                }
            )
            continue

        checks = suite.evaluate(example.sql, SchemaCatalog.from_mapping(schema_mapping))
        evaluated_examples += 1

        for metric, result in checks.items():
            metric_counts = metric_summary.setdefault(metric, _empty_metric_summary())
            metric_counts["total"] += 1
            metric_counts["detected"] += int(result.detected)
            if result.implemented:
                metric_counts["implemented_count"] += 1
            else:
                metric_counts["not_implemented_count"] += 1

    for metric in ("SHR", "POR", "DIVR", "SJR", "FDVR"):
        metric_counts = metric_summary.setdefault(metric, _empty_metric_summary())
        total = metric_counts["total"]
        metric_counts["detection_rate"] = (metric_counts["detected"] / total) if total else 0.0

    if evaluated_examples == 0:
        raise RuntimeError(
            "All subset examples were skipped due to missing schemas; no analysis was produced."
        )

    subset_summary = summarize_subset(subset)
    results: dict[str, Any] = {
        "phase": "phase2_spider",
        "config": {
            "seed": args.seed,
            "subset_size_requested": args.subset_size,
            "subset_size_selected": len(subset),
            "spider_dir": str(args.spider_dir),
            "subset_output_dir": str(args.subset_output_dir),
        },
        "inputs": {
            "schemas_loaded": len(schemas_by_db),
            "examples_loaded": len(all_examples),
            "examples_with_schema": len(examples_with_schema),
            "examples_missing_schema": len(examples_without_schema),
            "missing_schema_db_ids": sorted({example.db_id for example in examples_without_schema}),
            "source_counts": source_counts,
            "warnings": schema_warnings + example_warnings,
        },
        "subset": {
            **subset_summary,
            "artifact_paths": subset_artifacts,
        },
        "analysis": {
            "evaluated_examples": evaluated_examples,
            "skipped_missing_schema_count": len(skipped_missing_schema),
            "skipped_missing_schema_examples": skipped_missing_schema[:50],
            "metrics": metric_summary,
            "mandatory_metrics": {
                metric: metric_summary[metric] for metric in MANDATORY_METRICS
            },
        },
    }

    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    args.results_output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    log_section(logger, "Results Summary")
    log_dict(logger, {"evaluated_examples": evaluated_examples, "detection_rates": {
        m: metric_summary[m].get("detection_rate", 0.0) for m in MANDATORY_METRICS
    }})
    log_line(logger, f"Results saved to {args.results_output}")
    logger.info("Phase 2 complete")

    print(f"Loaded Spider schemas: {len(schemas_by_db)}")
    print(f"Loaded Spider SQL examples: {len(all_examples)}")
    print(f"Selected subset size: {len(subset)}")
    print(f"Evaluated subset examples: {evaluated_examples}")
    print(f"Subset artifacts: {subset_artifacts}")
    print(f"Phase 2 results: {args.results_output}")


if __name__ == "__main__":
    main()
