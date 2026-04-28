from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bootstrap import default_schema
from .checkers import ConstraintSuite
from .schema import SchemaCatalog

CORE_METRICS = ("shr", "por", "divr")
OPTIONAL_METRICS = ("sjr", "fdvr")
PHASE1_LABEL_SOURCE = "PHASE1_SYNTHETIC_TEMPLATE_NOT_HAND_LABELED"


def _build_clean_candidates() -> list[dict[str, Any]]:
    templates = [
        (
            "List employee names for department parameter #{i}.",
            "SELECT e.name FROM employees e WHERE e.department_id = :department_id_{i};",
        ),
        (
            "Show employees with salary above parameter #{i}.",
            "SELECT e.name FROM employees e WHERE e.salary > :min_salary_{i};",
        ),
        (
            "Filter active employees with parameterized flag #{i}.",
            "SELECT e.name FROM employees e WHERE e.active = :is_active_{i};",
        ),
        (
            "Find employees hired after parameterized date #{i}.",
            "SELECT e.name FROM employees e WHERE e.hired_at >= :start_date_{i};",
        ),
        (
            "Find departments by parameterized location #{i}.",
            "SELECT d.name FROM departments d WHERE d.location = :location_{i};",
        ),
        (
            "Find countries by parameterized region #{i}.",
            "SELECT c.name FROM countries c WHERE c.region = :region_{i};",
        ),
        (
            "Join employees and departments with parameterized department id #{i}.",
            "SELECT e.name, d.name FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.id = :department_id_{i};",
        ),
        (
            "Join employees and countries with parameterized country code #{i}.",
            "SELECT e.name, c.name FROM employees e JOIN countries c ON e.country_code = c.code WHERE c.code = :country_code_{i};",
        ),
        (
            "Count employees per department with minimum count parameter #{i}.",
            "SELECT d.name, COUNT(*) FROM departments d JOIN employees e ON e.department_id = d.id GROUP BY d.name HAVING COUNT(*) > :min_count_{i};",
        ),
        (
            "Retrieve employee id using parameterized identifier #{i}.",
            "SELECT e.id FROM employees e WHERE e.id = :employee_id_{i};",
        ),
    ]

    candidates: list[dict[str, Any]] = []
    for index in range(1, 31):
        for question_template, sql_template in templates:
            candidates.append(
                {
                    "question": question_template.format(i=index),
                    "sql": sql_template.format(i=index),
                    "labels": {"shr": False, "por": False, "divr": False},
                    "target_metric": "clean",
                }
            )
    return candidates


def _build_shr_candidates() -> list[dict[str, Any]]:
    bad_columns = [
        "nickname",
        "dept_name",
        "manager",
        "region_code",
        "office_id",
        "country",
        "grade",
        "skill_level",
        "joined_year",
        "division_name",
        "project_code",
        "salary_grade",
        "timezone",
        "cost_center",
        "badge_number",
        "seniority_band",
    ]
    bad_aliases = ["x", "z", "ghost", "m", "alias_q", "missing"]
    bad_tables = ["employee", "division", "offices", "salary_bands", "team_map", "regions"]

    candidates: list[dict[str, Any]] = []

    for index, bad_column in enumerate(bad_columns, start=1):
        candidates.append(
            {
                "question": f"SHR probe: select unknown employee field {bad_column} (variant {index}).",
                "sql": f"SELECT e.{bad_column} FROM employees e WHERE e.department_id = :department_id_{index};",
                "labels": {"shr": True, "por": False, "divr": False},
                "target_metric": "shr",
            }
        )
        candidates.append(
            {
                "question": f"SHR probe: use unknown predicate field {bad_column} (variant {index}).",
                "sql": f"SELECT e.name FROM employees e WHERE e.{bad_column} = :value_{index};",
                "labels": {"shr": True, "por": False, "divr": False},
                "target_metric": "shr",
            }
        )
        candidates.append(
            {
                "question": f"SHR probe: use unknown unqualified field {bad_column} (variant {index}).",
                "sql": f"SELECT e.name FROM employees e WHERE {bad_column} = :value_{index};",
                "labels": {"shr": True, "por": False, "divr": False},
                "target_metric": "shr",
            }
        )

    for index, bad_alias in enumerate(bad_aliases, start=1):
        candidates.append(
            {
                "question": f"SHR probe: use missing alias {bad_alias} (variant {index}).",
                "sql": f"SELECT {bad_alias}.name FROM employees e WHERE e.department_id = :department_id_{index};",
                "labels": {"shr": True, "por": False, "divr": False},
                "target_metric": "shr",
            }
        )

    for index, bad_table in enumerate(bad_tables, start=1):
        candidates.append(
            {
                "question": f"SHR probe: query unknown table {bad_table} (variant {index}).",
                "sql": f"SELECT e.name FROM {bad_table} e WHERE e.department_id = :department_id_{index};",
                "labels": {"shr": True, "por": False, "divr": False},
                "target_metric": "shr",
            }
        )
        candidates.append(
            {
                "question": f"SHR probe: join unknown table {bad_table} (variant {index}).",
                "sql": f"SELECT e.name FROM employees e JOIN {bad_table} d ON e.department_id = d.id WHERE e.id = :employee_id_{index};",
                "labels": {"shr": True, "por": False, "divr": False},
                "target_metric": "shr",
            }
        )

    return candidates


def _build_por_candidates() -> list[dict[str, Any]]:
    department_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    salary_values = [45000, 52000, 61000, 70000, 83000, 92000, 104000, 118000, 130000, 145000]
    locations = ["Berlin", "London", "Tokyo", "Toronto", "Austin", "Paris", "Lima", "Oslo", "Seoul", "Delhi"]
    regions = ["EMEA", "APAC", "AMER", "LATAM", "MEA", "EU", "NA", "SA", "ANZ", "GLOBAL"]
    country_codes = ["US", "CA", "DE", "JP", "BR", "FR", "IN", "KR", "NO", "MX"]

    candidates: list[dict[str, Any]] = []

    for index, value in enumerate(department_ids, start=1):
        candidates.append(
            {
                "question": f"POR probe: hardcode department id {value} (variant {index}).",
                "sql": f"SELECT e.name FROM employees e WHERE e.department_id = {value};",
                "labels": {"shr": False, "por": True, "divr": False},
                "target_metric": "por",
            }
        )
        candidates.append(
            {
                "question": f"POR probe: hardcode department id set around {value} (variant {index}).",
                "sql": f"SELECT e.name FROM employees e WHERE e.department_id IN ({value}, {value + 10}, {value + 20});",
                "labels": {"shr": False, "por": True, "divr": False},
                "target_metric": "por",
            }
        )

    for index, value in enumerate(salary_values, start=1):
        candidates.append(
            {
                "question": f"POR probe: hardcode salary threshold {value} (variant {index}).",
                "sql": f"SELECT e.name FROM employees e WHERE e.salary > {value};",
                "labels": {"shr": False, "por": True, "divr": False},
                "target_metric": "por",
            }
        )

    for index, location in enumerate(locations, start=1):
        candidates.append(
            {
                "question": f"POR probe: hardcode location {location} (variant {index}).",
                "sql": f"SELECT d.name FROM departments d WHERE d.location = '{location}';",
                "labels": {"shr": False, "por": True, "divr": False},
                "target_metric": "por",
            }
        )

    for index, region in enumerate(regions, start=1):
        candidates.append(
            {
                "question": f"POR probe: hardcode region {region} (variant {index}).",
                "sql": f"SELECT c.name FROM countries c WHERE c.region = '{region}';",
                "labels": {"shr": False, "por": True, "divr": False},
                "target_metric": "por",
            }
        )

    for index, code in enumerate(country_codes, start=1):
        candidates.append(
            {
                "question": f"POR probe: hardcode country code {code} (variant {index}).",
                "sql": f"SELECT e.name FROM employees e WHERE e.country_code = '{code}';",
                "labels": {"shr": False, "por": True, "divr": False},
                "target_metric": "por",
            }
        )

    for index in range(1, 11):
        boolean_literal = "true" if index % 2 else "false"
        candidates.append(
            {
                "question": f"POR probe: hardcode active flag {boolean_literal} (variant {index}).",
                "sql": f"SELECT e.name FROM employees e WHERE e.active = {boolean_literal};",
                "labels": {"shr": False, "por": True, "divr": False},
                "target_metric": "por",
            }
        )

    for index in range(1, 11):
        month = ((index - 1) % 12) + 1
        day = ((index * 2 - 1) % 28) + 1
        date_literal = f"2022-{month:02d}-{day:02d}"
        candidates.append(
            {
                "question": f"POR probe: hardcode hire date {date_literal} (variant {index}).",
                "sql": f"SELECT e.name FROM employees e WHERE e.hired_at >= '{date_literal}';",
                "labels": {"shr": False, "por": True, "divr": False},
                "target_metric": "por",
            }
        )

    return candidates


def _build_divr_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for index, token in enumerate(["'high'", "'low'", "'premium'", "'entry'", "'n/a'", "'avg'", "'band-a'", "'tier-3'", "'~100k'", "'unknown'"], start=1):
        candidates.append(
            {
                "question": f"DIVR probe: compare numeric salary to text token {token} (variant {index}).",
                "sql": f"SELECT e.name FROM employees e WHERE e.salary = {token};",
                "labels": {"shr": False, "por": True, "divr": True},
                "target_metric": "divr",
            }
        )

    for index, token in enumerate(["'sales'", "'north'", "'central'", "'dept-a'", "'alpha'", "'team-7'", "'exec'", "'field'", "'support'", "'biz'"], start=1):
        candidates.append(
            {
                "question": f"DIVR probe: compare integer department id to text token {token} (variant {index}).",
                "sql": f"SELECT e.name FROM employees e WHERE e.department_id = {token};",
                "labels": {"shr": False, "por": True, "divr": True},
                "target_metric": "divr",
            }
        )

    for index, token in enumerate(["'yes'", "'no'", "'enabled'", "'disabled'", "'active'", "'inactive'", "'1'", "'0'", "'t'", "'f'"], start=1):
        candidates.append(
            {
                "question": f"DIVR probe: compare boolean column to text token {token} (variant {index}).",
                "sql": f"SELECT e.name FROM employees e WHERE e.active = {token};",
                "labels": {"shr": False, "por": True, "divr": True},
                "target_metric": "divr",
            }
        )

    for index, value in enumerate([20240101, 20240202, 20240303, 20240404, 20240505, 20240606, 20240707, 20240808, 20240909, 20241010], start=1):
        candidates.append(
            {
                "question": f"DIVR probe: compare date to numeric token {value} (variant {index}).",
                "sql": f"SELECT e.name FROM employees e WHERE e.hired_at = {value};",
                "labels": {"shr": False, "por": True, "divr": True},
                "target_metric": "divr",
            }
        )

    for index, value in enumerate([404, 505, 606, 707, 808, 909, 1001, 1111, 1212, 1313], start=1):
        candidates.append(
            {
                "question": f"DIVR probe: compare text location to numeric token {value} (variant {index}).",
                "sql": f"SELECT d.name FROM departments d WHERE d.location = {value};",
                "labels": {"shr": False, "por": True, "divr": True},
                "target_metric": "divr",
            }
        )

    for index, pair in enumerate(
        [
            "'low', 'high'",
            "'gold', 'silver'",
            "'small', 'large'",
            "'junior', 'senior'",
            "'left', 'right'",
            "'foo', 'bar'",
            "'alpha', 'beta'",
            "'x', 'y'",
            "'north', 'south'",
            "'std', 'plus'",
        ],
        start=1,
    ):
        candidates.append(
            {
                "question": f"DIVR probe: numeric salary IN text set ({pair}) (variant {index}).",
                "sql": f"SELECT e.name FROM employees e WHERE e.salary IN ({pair});",
                "labels": {"shr": False, "por": True, "divr": True},
                "target_metric": "divr",
            }
        )

    for index, pair in enumerate(
        [
            "1, 2",
            "3, 4",
            "5, 6",
            "7, 8",
            "9, 10",
            "11, 12",
            "13, 14",
            "15, 16",
            "17, 18",
            "19, 20",
        ],
        start=1,
    ):
        candidates.append(
            {
                "question": f"DIVR probe: date column IN numeric set ({pair}) (variant {index}).",
                "sql": f"SELECT e.name FROM employees e WHERE e.hired_at IN ({pair});",
                "labels": {"shr": False, "por": True, "divr": True},
                "target_metric": "divr",
            }
        )

    return candidates


def _select_examples(candidates: list[dict[str, Any]], count: int, label: str) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise ValueError(
            f"Not enough {label} candidates to satisfy target count {count}. Available={len(candidates)}"
        )
    return candidates[:count]


def _validate_labels(records: list[dict[str, Any]]) -> None:
    suite = ConstraintSuite()
    catalog = SchemaCatalog.from_mapping(default_schema())

    for index, record in enumerate(records, start=1):
        predicted = suite.core_flags(record["sql"], catalog)
        predicted_labels = {
            "shr": predicted["SHR"],
            "por": predicted["POR"],
            "divr": predicted["DIVR"],
        }
        if predicted_labels != record["labels"]:
            raise ValueError(
                "Template labels disagree with checker output for record "
                f"{index} ({record['target_metric']}): expected {record['labels']}, got {predicted_labels}."
            )


def build_phase1_synthetic_records(
    target_per_metric: int = 60,
    clean_count: int = 30,
) -> list[dict[str, Any]]:
    if target_per_metric <= 0:
        raise ValueError("target_per_metric must be positive")
    if clean_count <= 0:
        raise ValueError("clean_count must be positive")

    schema = default_schema()

    selected_records: list[dict[str, Any]] = []
    category_specs = [
        ("clean", _build_clean_candidates(), clean_count),
        ("shr", _build_shr_candidates(), target_per_metric),
        ("por", _build_por_candidates(), target_per_metric),
        ("divr", _build_divr_candidates(), target_per_metric),
    ]

    for category, candidates, count in category_specs:
        selected = _select_examples(candidates, count=count, label=category)
        for local_index, row in enumerate(selected, start=1):
            selected_records.append(
                {
                    "id": f"phase1-{category}-{local_index:03d}",
                    "question": row["question"],
                    "sql": row["sql"],
                    "schema": schema,
                    "labels": row["labels"],
                    "optional_labels": {"sjr": None, "fdvr": None},
                    "optional_label_notes": "SJR/FDVR are scaffolded only in phase 1 and are not hand-labeled.",
                    "target_metric": row["target_metric"],
                    "label_source": PHASE1_LABEL_SOURCE,
                }
            )

    _validate_labels(selected_records)
    return selected_records


def write_phase1_synthetic_dataset(path: Path, records: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
    return len(records)
