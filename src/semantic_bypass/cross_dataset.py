from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any

from .checkers import ConstraintSuite
from .schema import SchemaCatalog

CORE_METRICS: tuple[str, ...] = ("SHR", "POR", "DIVR")
ALL_METRICS: tuple[str, ...] = ("SHR", "POR", "DIVR", "SJR", "FDVR")


@dataclass(frozen=True)
class DatasetRecord:
    example_id: str
    question: str
    sql: str
    schema: dict[str, dict[str, str]]
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "question": self.question,
            "sql": self.sql,
            "schema": self.schema,
            "source": self.source,
            "metadata": self.metadata,
        }



def normalize_schema(raw_schema: Any) -> dict[str, dict[str, str]]:
    if not isinstance(raw_schema, dict):
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for table, columns in raw_schema.items():
        if not isinstance(table, str) or not isinstance(columns, dict):
            continue
        cleaned_columns: dict[str, str] = {}
        for column, column_type in columns.items():
            if not isinstance(column, str):
                continue
            cleaned_columns[column] = str(column_type).strip() or "text"
        if cleaned_columns:
            normalized[table] = cleaned_columns
    return normalized



def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not path.exists():
        return [], [f"File does not exist: {path}"]

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as error:
                    warnings.append(
                        f"Skipped invalid JSONL line {line_number} in {path.name}: {error}"
                    )
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
                else:
                    warnings.append(f"Skipped non-object row line {line_number} in {path.name}")
        return rows, warnings

    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            return [], [f"Invalid JSON in {path}: {error}"]

        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
            if len(rows) != len(payload):
                warnings.append(f"Skipped {len(payload) - len(rows)} non-object rows from {path.name}")
            return rows, warnings
        return [], [f"Unsupported JSON shape in {path}: expected a list of objects"]

    return [], [f"Unsupported file extension for dataset rows: {path}"]



def deterministic_sample(
    records: list[DatasetRecord],
    max_examples: int,
    seed: int,
) -> list[DatasetRecord]:
    if max_examples == 0 or not records:
        return []
    shuffled = sorted(records, key=lambda record: record.example_id)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if max_examples < 0:
        return shuffled
    return shuffled[: min(max_examples, len(shuffled))]



def write_records_jsonl(path: Path, records: list[DatasetRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(), sort_keys=True))
            handle.write("\n")
    return len(records)



def _empty_metric_summary() -> dict[str, Any]:
    return {
        "total": 0,
        "detected": 0,
        "detection_rate": 0.0,
        "implemented_count": 0,
        "not_implemented_count": 0,
        "notes": [],
    }



def evaluate_records(
    records: list[DatasetRecord],
    *,
    sample_limit: int = 5,
) -> dict[str, Any]:
    suite = ConstraintSuite()
    metric_summary = {metric: _empty_metric_summary() for metric in ALL_METRICS}
    any_core_violation_count = 0
    samples: list[dict[str, Any]] = []

    for record in records:
        checks = suite.evaluate(record.sql, SchemaCatalog.from_mapping(record.schema))
        core_flags = {metric: checks[metric].detected for metric in CORE_METRICS}
        if any(core_flags.values()):
            any_core_violation_count += 1

        for metric, result in checks.items():
            metric_counts = metric_summary[metric]
            metric_counts["total"] += 1
            metric_counts["detected"] += int(result.detected)
            if result.implemented:
                metric_counts["implemented_count"] += 1
            else:
                metric_counts["not_implemented_count"] += 1
            if result.notes:
                metric_counts["notes"].append(result.notes)

        if len(samples) < max(sample_limit, 0):
            samples.append(
                {
                    "example_id": record.example_id,
                    "source": record.source,
                    "question": record.question,
                    "sql": record.sql,
                    "core_flags": core_flags,
                }
            )

    for metric in ALL_METRICS:
        metric_counts = metric_summary[metric]
        total = metric_counts["total"]
        metric_counts["detection_rate"] = (
            metric_counts["detected"] / total if total else 0.0
        )
        metric_counts["notes"] = metric_counts["notes"][:5]

    total_examples = len(records)
    return {
        "examples_evaluated": total_examples,
        "any_core_violation": {
            "count": any_core_violation_count,
            "rate": any_core_violation_count / total_examples if total_examples else 0.0,
        },
        "core_metrics": {metric: metric_summary[metric] for metric in CORE_METRICS},
        "all_metrics": metric_summary,
        "samples": samples,
    }
