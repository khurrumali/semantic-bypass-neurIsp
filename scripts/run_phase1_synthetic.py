from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

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
from semantic_bypass.synthetic_generator import (
    CORE_METRICS,
    build_phase1_synthetic_records,
    write_phase1_synthetic_dataset,
)


def _compute_scores(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_records(records: list[dict]) -> dict[str, dict]:
    suite = ConstraintSuite()
    confusion = {
        metric: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for metric in CORE_METRICS
    }

    for index, record in enumerate(records, start=1):
        schema = SchemaCatalog.from_mapping(record["schema"])
        predictions = suite.core_flags(record["sql"], schema)

        for metric in CORE_METRICS:
            predicted = predictions[metric.upper()]
            actual = record["labels"][metric]
            if predicted and actual:
                confusion[metric]["tp"] += 1
            elif predicted and not actual:
                confusion[metric]["fp"] += 1
            elif not predicted and actual:
                confusion[metric]["fn"] += 1
            elif not predicted and not actual:
                confusion[metric]["tn"] += 1
            else:
                raise RuntimeError(f"Unexpected label state on record {index}")

    total = len(records)
    per_metric: dict[str, dict] = {}
    for metric in CORE_METRICS:
        counts = confusion[metric]
        tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        scores = _compute_scores(tp=tp, fp=fp, fn=fn)
        labeled_positive_rate = (tp + fn) / total if total else 0.0
        detected_rate = (tp + fp) / total if total else 0.0

        per_metric[metric] = {
            "counts": counts,
            "labeled_positive_rate": labeled_positive_rate,
            "detected_positive_rate": detected_rate,
            **scores,
        }

    optional_status: dict[str, dict[str, str | bool | None]] = {}
    sample_schema = SchemaCatalog.from_mapping(records[0]["schema"])
    sample_results = suite.evaluate(records[0]["sql"], sample_schema)
    for metric in ("SJR", "FDVR"):
        result = sample_results[metric]
        optional_status[metric.lower()] = {
            "implemented": result.implemented,
            "notes": result.notes,
        }

    return {
        "example_count": total,
        "core_metrics": per_metric,
        "optional_metrics": optional_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate Phase 1 synthetic SHR/POR/DIVR probes."
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=ROOT / "data" / "synthetic" / "phase1_synthetic.jsonl",
        help="Path for generated synthetic dataset (JSONL).",
    )
    parser.add_argument(
        "--results-output",
        type=Path,
        default=ROOT / "results" / "phase1_synthetic.json",
        help="Path for aggregate evaluation JSON output.",
    )
    parser.add_argument(
        "--target-per-metric",
        type=int,
        default=60,
        help="Number of examples to generate per core metric (SHR/POR/DIVR).",
    )
    parser.add_argument(
        "--clean-count",
        type=int,
        default=30,
        help="Number of clean (non-violation) controls.",
    )
    parser.add_argument(
        "--min-total",
        type=int,
        default=180,
        help="Minimum acceptable dataset size for Phase 1.",
    )
    args = parser.parse_args()

    log_path = get_run_log_path(phase="phase1")
    logger = setup_logger("phase1", log_file=log_path)

    log_section(logger, "Phase 1: Synthetic Constraint Probes")
    log_dict(logger, {"target_per_metric": args.target_per_metric, "clean_count": args.clean_count})
    log_line(logger, f"Log output: {log_path}")

    log_line(logger, "Generating synthetic records...")
    records = build_phase1_synthetic_records(
        target_per_metric=args.target_per_metric,
        clean_count=args.clean_count,
    )
    log_dict(logger, {"record_count": len(records)})

    if len(records) < args.min_total:
        raise RuntimeError(
            f"Generated dataset has {len(records)} examples, below required minimum {args.min_total}."
        )

    log_line(logger, f"Writing dataset to {args.dataset_output}")
    count = write_phase1_synthetic_dataset(args.dataset_output, records)
    log_dict(logger, {"written": count})

    log_line(logger, "Evaluating records with constraint checkers...")
    results = evaluate_records(records)

    results.update(
        {
            "dataset_output": str(args.dataset_output),
            "results_output": str(args.results_output),
            "target_per_metric": args.target_per_metric,
            "clean_count": args.clean_count,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "label_note": "Labels are template-based synthetic annotations and are not hand-labeled.",
        }
    )

    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    args.results_output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    log_section(logger, "Results Summary")
    log_dict(logger, {"example_count": results["example_count"]})
    log_line(logger, "metric | labeled_rate | detected_rate | precision | recall | f1")
    for metric in CORE_METRICS:
        metric_result = results["core_metrics"][metric]
        line = (
            f"{metric.upper()} | "
            f"{metric_result['labeled_positive_rate']:.3f} | "
            f"{metric_result['detected_positive_rate']:.3f} | "
            f"{metric_result['precision']:.3f} | "
            f"{metric_result['recall']:.3f} | "
            f"{metric_result['f1']:.3f}"
        )
        log_line(logger, line)

    log_line(logger, f"Results saved to {args.results_output}")
    logger.info("Phase 1 complete")
    print(f"Generated {count} examples at {args.dataset_output}")
    print("Results logged to:", log_path)


if __name__ == "__main__":
    main()
