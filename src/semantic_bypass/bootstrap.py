from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkers import ConstraintSuite
from .schema import SchemaCatalog

BOOTSTRAP_LABEL_SOURCE = "BOOTSTRAP_SYNTHETIC_NOT_HAND_LABELED"


def default_schema() -> dict[str, dict[str, str]]:
    return {
        "employees": {
            "id": "integer",
            "name": "text",
            "department_id": "integer",
            "salary": "numeric",
            "active": "boolean",
            "hired_at": "date",
            "country_code": "text",
        },
        "departments": {
            "id": "integer",
            "name": "text",
            "location": "text",
        },
        "countries": {
            "code": "text",
            "name": "text",
            "region": "text",
        },
    }


def bootstrap_sql_queries() -> list[str]:
    clean = [
        "SELECT e.id, e.name FROM employees e WHERE e.department_id = :department_id;",
        "SELECT e.name, e.salary FROM employees e WHERE e.salary > :min_salary;",
        "SELECT d.name FROM departments d WHERE d.location = :location;",
        "SELECT e.name, d.name FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.id = :department_id;",
        "SELECT e.name FROM employees e WHERE e.active = :is_active;",
        "SELECT e.name FROM employees e WHERE e.hired_at >= :start_date;",
        "SELECT e.name, c.name FROM employees e JOIN countries c ON e.country_code = c.code WHERE c.region = :region;",
        "SELECT e.id FROM employees e WHERE e.country_code = :country_code;",
        "SELECT d.id, d.name FROM departments d WHERE d.id IN (:dept_one, :dept_two);",
        "SELECT e.name FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.location = :dept_location;",
        "SELECT e.name FROM employees e WHERE e.salary BETWEEN :min_salary AND :max_salary;",
        "SELECT c.name FROM countries c WHERE c.code = :country_code;",
        "SELECT e.name FROM employees e WHERE e.id = ?;",
        "SELECT e.name FROM employees e WHERE e.department_id = $1;",
        "SELECT d.name, COUNT(*) FROM departments d JOIN employees e ON e.department_id = d.id GROUP BY d.name;",
    ]

    por = [
        "SELECT e.name FROM employees e WHERE e.department_id = 3;",
        "SELECT e.name FROM employees e WHERE e.salary > 120000;",
        "SELECT d.name FROM departments d WHERE d.location = 'Berlin';",
        "SELECT e.name FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.id = 4;",
        "SELECT e.name FROM employees e WHERE e.active = true;",
        "SELECT e.name FROM employees e WHERE e.hired_at >= '2021-01-01';",
        "SELECT c.name FROM countries c WHERE c.region = 'EMEA';",
        "SELECT e.name FROM employees e WHERE e.country_code = 'US';",
        "SELECT e.name FROM employees e WHERE e.department_id IN (1, 2, 3);",
        "SELECT e.name FROM employees e WHERE e.name = 'Alice';",
        "SELECT e.name FROM employees e WHERE e.id = 42;",
        "SELECT d.name FROM departments d WHERE d.id IN (10, 11);",
        "SELECT e.name FROM employees e WHERE e.salary = 85000;",
        "SELECT e.name FROM employees e WHERE e.department_id <> 8;",
        "SELECT c.name FROM countries c WHERE c.code = 'CA';",
    ]

    shr = [
        "SELECT e.nickname FROM employees e;",
        "SELECT x.name FROM employees e;",
        "SELECT e.name FROM employee e;",
        "SELECT e.name FROM employees e JOIN division d ON e.department_id = d.id;",
        "SELECT d.region_code FROM departments d;",
        "SELECT e.name FROM employees e WHERE e.dept_id = :department_id;",
        "SELECT e.name FROM employees e WHERE office_id = :office_id;",
        "SELECT c.full_name FROM countries c;",
        "SELECT e.name FROM employees e JOIN countries c ON e.country_code = c.iso_code;",
        "SELECT z.id FROM employees e JOIN departments d ON e.department_id = d.id;",
    ]

    divr = [
        "SELECT e.name FROM employees e WHERE e.salary = 'high';",
        "SELECT e.name FROM employees e WHERE e.department_id = 'sales';",
        "SELECT e.name FROM employees e WHERE e.active = 'yes';",
        "SELECT e.name FROM employees e WHERE e.hired_at = 20240101;",
        "SELECT d.name FROM departments d WHERE d.id = 'north';",
        "SELECT e.name FROM employees e WHERE e.id = 'abc';",
        "SELECT e.name FROM employees e WHERE e.salary IN ('low', 'high');",
        "SELECT e.name FROM employees e WHERE e.hired_at IN (1, 2);",
        "SELECT e.name FROM employees e WHERE e.active IN ('true', 'false');",
        "SELECT e.name FROM employees e WHERE e.department_id IN ('1', '2');",
    ]

    queries = clean + por + shr + divr
    if len(queries) != 50:
        raise RuntimeError(f"Bootstrap query set must contain exactly 50 SQL examples, found {len(queries)}")
    return queries


def build_bootstrap_records() -> list[dict[str, Any]]:
    schema = default_schema()
    catalog = SchemaCatalog.from_mapping(schema)
    suite = ConstraintSuite()

    records: list[dict[str, Any]] = []
    for index, query in enumerate(bootstrap_sql_queries(), start=1):
        flags = suite.core_flags(query, catalog)
        records.append(
            {
                "id": f"bootstrap-{index:02d}",
                "question": f"Bootstrap synthetic example {index}",
                "sql": query,
                "schema": schema,
                "labels": {
                    "shr": flags["SHR"],
                    "por": flags["POR"],
                    "divr": flags["DIVR"],
                },
                "label_source": BOOTSTRAP_LABEL_SOURCE,
            }
        )
    return records


def write_bootstrap_dataset(path: Path) -> int:
    records = build_bootstrap_records()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
    return len(records)
