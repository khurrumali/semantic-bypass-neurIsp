from __future__ import annotations

import argparse
import json
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
from semantic_bypass.bootstrap import write_bootstrap_dataset
from semantic_bypass.checkers import ConstraintSuite
from semantic_bypass.schema import SchemaCatalog

CORE_METRICS = ("shr", "por", "divr")


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
    return records


def validate_record_shape(record: dict, index: int) -> None:
    required = {"id", "sql", "schema", "labels"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"Record {index} missing fields: {sorted(missing)}")

    labels = record["labels"]
    if not isinstance(labels, dict):
        raise ValueError(f"Record {index} labels must be an object")

    for metric in CORE_METRICS:
        if metric not in labels:
            raise ValueError(f"Record {index} missing label for '{metric}'")
        if not isinstance(labels[metric], bool):
            raise ValueError(f"Record {index} label '{metric}' must be boolean")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate checker performance on exactly 50 labeled SQL examples."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data" / "bootstrap_labeled_sql.jsonl",
        help="Path to JSONL dataset.",
    )
    parser.add_argument(
        "--create-bootstrap-if-missing",
        action="store_true",
        help="Create bootstrap dataset when --dataset path does not exist.",
    )
    args = parser.parse_args()

    log_path = get_run_log_path(phase="validate")
    logger = setup_logger("validate", log_file=log_path)

    log_section(logger, "Validate Checker")
    log_line(logger, f"Dataset: {args.dataset}")
    log_line(logger, f"Log output: {log_path}")

    if not args.dataset.exists():
        if not args.create_bootstrap_if_missing:
            raise FileNotFoundError(
                f"Dataset not found: {args.dataset}. Run scripts/build_bootstrap_dataset.py first."
            )
        count = write_bootstrap_dataset(args.dataset)
        log_dict(logger, {"bootstrap_created": count})
        print(f"Created bootstrap dataset with {count} examples at {args.dataset}")

    records = load_jsonl(args.dataset)
    if len(records) != 50:
        raise ValueError(f"Dataset must contain exactly 50 examples. Found {len(records)}")
    log_dict(logger, {"record_count": len(records)})

    suite = ConstraintSuite()
    confusion = {
        metric: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for metric in CORE_METRICS
    }

    for index, record in enumerate(records, start=1):
        validate_record_shape(record, index)
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
            else:
                confusion[metric]["tn"] += 1

    log_section(logger, "Results")
    log_line(logger, "metric | tp | fp | fn | tn | precision | recall")
    for metric in CORE_METRICS:
        counts = confusion[metric]
        precision_denominator = counts["tp"] + counts["fp"]
        recall_denominator = counts["tp"] + counts["fn"]
        precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
        recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
        line = f"{metric.upper()} | {counts['tp']} | {counts['fp']} | {counts['fn']} | {counts['tn']} | {precision:.3f} | {recall:.3f}"
        log_line(logger, line)
    logger.info("Validation complete")

    print(f"Validated {len(records)} examples from {args.dataset}")
    print("metric\ttp\tfp\tfn\ttn\tprecision\trecall")
    for metric in CORE_METRICS:
        counts = confusion[metric]
        precision_denominator = counts["tp"] + counts["fp"]
        recall_denominator = counts["tp"] + counts["fn"]
        precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
        recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
        print(
            f"{metric.upper()}\t{counts['tp']}\t{counts['fp']}\t{counts['fn']}\t{counts['tn']}\t{precision:.3f}\t{recall:.3f}"
        )


if __name__ == "__main__":
    main()
