from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Sequence

SPIDER_JSON_FILES: tuple[tuple[str, str, str], ...] = (
    ("train", "train_spider", "train_spider.json"),
    ("train", "train_others", "train_others.json"),
    ("dev", "dev", "dev.json"),
    ("test", "test", "test.json"),
)

SPIDER_GOLD_FILES: tuple[tuple[str, str], ...] = (
    ("train", "train_gold.sql"),
    ("dev", "dev_gold.sql"),
    ("test", "test_gold.sql"),
)


@dataclass(frozen=True)
class SpiderExample:
    example_id: str
    split: str
    db_id: str
    sql: str
    question: str
    source_file: str

    def to_record(self) -> dict[str, str]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "db_id": self.db_id,
            "sql": self.sql,
            "question": self.question,
            "source_file": self.source_file,
        }


def _normalize_spider_type(raw_type: str) -> str:
    normalized = raw_type.strip().lower()
    if normalized in {"number", "int", "integer", "float", "double", "decimal", "numeric", "real"}:
        return "numeric"
    if normalized in {"time", "datetime", "timestamp", "year"}:
        return "date"
    if normalized in {"bool", "boolean"}:
        return "boolean"
    if normalized in {"others", "other"}:
        return "text"
    return normalized or "text"


def load_spider_schemas(spider_root: Path) -> tuple[dict[str, dict[str, dict[str, str]]], list[str]]:
    warnings: list[str] = []
    tables_path = spider_root / "tables.json"
    if not tables_path.exists():
        return {}, [f"Missing Spider schema file: {tables_path}"]

    try:
        payload = json.loads(tables_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {}, [f"Invalid JSON in {tables_path}: {error}"]

    if not isinstance(payload, list):
        return {}, [f"Unexpected tables.json shape in {tables_path}: expected a list"]

    schemas: dict[str, dict[str, dict[str, str]]] = {}
    for db_entry in payload:
        if not isinstance(db_entry, dict):
            continue
        db_id = str(db_entry.get("db_id", "")).strip()
        if not db_id:
            continue

        table_names_raw = db_entry.get("table_names_original") or db_entry.get("table_names") or []
        table_names: list[str] = [str(name) for name in table_names_raw if isinstance(name, str)]
        schema: dict[str, dict[str, str]] = {table_name: {} for table_name in table_names}

        col_names_original = db_entry.get("column_names_original") or []
        col_names_fallback = db_entry.get("column_names") or []
        col_types = db_entry.get("column_types") or []

        for index, column_entry in enumerate(col_names_original):
            if not isinstance(column_entry, list) or len(column_entry) < 2:
                continue
            table_index = column_entry[0]
            if not isinstance(table_index, int) or table_index < 0:
                continue
            if table_index >= len(table_names):
                warnings.append(
                    f"Skipped out-of-range table index in {db_id} at column index {index}"
                )
                continue

            column_name = column_entry[1]
            if not isinstance(column_name, str) or not column_name.strip():
                fallback_name = (
                    col_names_fallback[index][1]
                    if index < len(col_names_fallback)
                    and isinstance(col_names_fallback[index], list)
                    and len(col_names_fallback[index]) >= 2
                    else ""
                )
                column_name = str(fallback_name).strip()
            else:
                column_name = column_name.strip()

            if not column_name:
                continue

            raw_type = str(col_types[index]) if index < len(col_types) else "text"
            schema[table_names[table_index]][column_name] = _normalize_spider_type(raw_type)

        schemas[db_id] = schema

    return schemas, warnings


def _load_examples_from_json(
    path: Path,
    split: str,
    source_name: str,
) -> tuple[list[SpiderExample], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [], f"Invalid JSON in {path}: {error}"

    if not isinstance(payload, list):
        return [], f"Unexpected shape in {path}: expected a list"

    examples: list[SpiderExample] = []
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            continue
        db_id = str(row.get("db_id", "")).strip()
        sql = str(row.get("query") or "").strip()
        if not db_id or not sql:
            continue
        question_value = row.get("question", "")
        question = question_value.strip() if isinstance(question_value, str) else ""
        examples.append(
            SpiderExample(
                example_id=f"{source_name}:{index:05d}",
                split=split,
                db_id=db_id,
                sql=sql,
                question=question,
                source_file=path.name,
            )
        )
    return examples, None


def _load_examples_from_gold(path: Path, split: str) -> list[SpiderExample]:
    examples: list[SpiderExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split("\t")
            if len(fields) < 2:
                continue
            sql = fields[0].strip()
            db_id = fields[-1].strip()
            if not db_id or not sql:
                continue
            examples.append(
                SpiderExample(
                    example_id=f"{split}:gold:{line_number:05d}",
                    split=split,
                    db_id=db_id,
                    sql=sql,
                    question="",
                    source_file=path.name,
                )
            )
    return examples


def load_spider_examples(
    spider_root: Path,
) -> tuple[list[SpiderExample], dict[str, int], list[str]]:
    warnings: list[str] = []
    examples: list[SpiderExample] = []
    source_counts: dict[str, int] = {}
    json_loaded_by_split: dict[str, bool] = defaultdict(bool)

    for split, source_name, filename in SPIDER_JSON_FILES:
        path = spider_root / filename
        if not path.exists():
            warnings.append(f"Missing Spider file: {path}")
            continue

        source_examples, error = _load_examples_from_json(path, split=split, source_name=source_name)
        if error:
            warnings.append(error)
            continue

        if source_examples:
            examples.extend(source_examples)
            source_counts[path.name] = len(source_examples)
            json_loaded_by_split[split] = True
        else:
            warnings.append(f"No usable SQL examples found in {path}")

    for split, filename in SPIDER_GOLD_FILES:
        if json_loaded_by_split[split]:
            continue
        path = spider_root / filename
        if not path.exists():
            warnings.append(f"Missing fallback SQL file: {path}")
            continue
        source_examples = _load_examples_from_gold(path, split=split)
        if source_examples:
            examples.extend(source_examples)
            source_counts[path.name] = len(source_examples)
        else:
            warnings.append(f"No usable SQL examples found in fallback file {path}")

    examples.sort(key=lambda item: (item.db_id, item.example_id))
    return examples, source_counts, warnings


def create_stratified_subset(
    examples: Sequence[SpiderExample],
    sample_size: int,
    seed: int,
) -> list[SpiderExample]:
    if sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if not examples:
        return []

    target_size = min(sample_size, len(examples))
    grouped: dict[str, list[SpiderExample]] = defaultdict(list)
    for example in examples:
        grouped[example.db_id].append(example)

    rng = random.Random(seed)
    db_ids = sorted(grouped)
    for db_id in db_ids:
        grouped[db_id].sort(key=lambda item: item.example_id)
        rng.shuffle(grouped[db_id])
    rng.shuffle(db_ids)

    subset: list[SpiderExample] = []
    offset = 0
    while len(subset) < target_size:
        added = False
        for db_id in db_ids:
            bucket = grouped[db_id]
            if offset < len(bucket):
                subset.append(bucket[offset])
                added = True
                if len(subset) >= target_size:
                    break
        if not added:
            break
        offset += 1
    return subset


def summarize_subset(subset: Sequence[SpiderExample]) -> dict[str, Any]:
    db_distribution = Counter(example.db_id for example in subset)
    source_distribution = Counter(example.source_file for example in subset)
    split_distribution = Counter(example.split for example in subset)
    return {
        "subset_size": len(subset),
        "db_count": len(db_distribution),
        "db_distribution": dict(sorted(db_distribution.items())),
        "source_distribution": dict(sorted(source_distribution.items())),
        "split_distribution": dict(sorted(split_distribution.items())),
    }


def write_subset_artifact(
    subset: Sequence[SpiderExample],
    output_dir: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = output_dir / "phase2_spider_subset.jsonl"
    manifest_path = output_dir / "phase2_spider_subset_manifest.json"

    with subset_path.open("w", encoding="utf-8") as handle:
        for example in subset:
            handle.write(json.dumps(example.to_record(), sort_keys=True))
            handle.write("\n")

    manifest: dict[str, Any] = {
        "subset_summary": summarize_subset(subset),
    }
    if metadata:
        manifest.update(metadata)

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"subset_jsonl": str(subset_path), "manifest_json": str(manifest_path)}
