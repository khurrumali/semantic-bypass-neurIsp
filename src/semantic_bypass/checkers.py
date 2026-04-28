from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .schema import SchemaCatalog

CORE_METRICS = ("SHR", "POR", "DIVR")

RESERVED_WORDS = {
    "as",
    "and",
    "or",
    "on",
    "where",
    "group",
    "order",
    "limit",
    "having",
    "inner",
    "left",
    "right",
    "full",
    "cross",
    "join",
    "from",
    "select",
}

TABLE_ALIAS_RE = re.compile(
    r"\b(?:from|join)\s+([a-zA-Z_][\w]*)(?:\s+(?:as\s+)?([a-zA-Z_][\w]*))?",
    re.IGNORECASE,
)
EXPLICIT_COLUMN_RE = re.compile(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b")
PREDICATE_RE = re.compile(
    r"\b(?P<left>[a-zA-Z_][\w]*(?:\.[a-zA-Z_][\w]*)?)\s*"
    r"(?P<op>=|!=|<>|>=|<=|>|<|like|in)\s*"
    r"(?P<right>\([^)]*\)|'[^']*'|\"[^\"]*\"|-?\d+(?:\.\d+)?|true|false|:[a-zA-Z_][\w]*|\$\d+|\?)",
    re.IGNORECASE,
)
DATE_LITERAL_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2})?$")


@dataclass(frozen=True)
class Violation:
    metric: str
    message: str
    fragment: str | None = None


@dataclass
class CheckResult:
    metric: str
    violations: list[Violation] = field(default_factory=list)
    implemented: bool = True
    notes: str | None = None

    @property
    def detected(self) -> bool:
        return len(self.violations) > 0


@dataclass(frozen=True)
class Predicate:
    left: str
    op: str
    right: str


class ConstraintChecker(Protocol):
    metric: str

    def check(self, sql: str, schema: SchemaCatalog) -> CheckResult: ...


def _extract_alias_map(sql_query: str) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for table, alias in TABLE_ALIAS_RE.findall(sql_query):
        table_lower = table.lower()
        alias_map[table_lower] = table_lower
        alias_lower = alias.lower() if alias else ""
        if alias_lower and alias_lower not in RESERVED_WORDS:
            alias_map[alias_lower] = table_lower
    return alias_map


def _extract_explicit_columns(sql_query: str) -> list[tuple[str, str]]:
    return [(prefix.lower(), column.lower()) for prefix, column in EXPLICIT_COLUMN_RE.findall(sql_query)]


def _extract_predicates(sql_query: str) -> list[Predicate]:
    predicates: list[Predicate] = []
    for match in PREDICATE_RE.finditer(sql_query):
        predicates.append(
            Predicate(
                left=match.group("left").strip(),
                op=match.group("op").upper(),
                right=match.group("right").strip(),
            )
        )
    return predicates


def _is_parameter_token(token: str) -> bool:
    stripped = token.strip()
    return stripped == "?" or stripped.startswith(":") or bool(re.fullmatch(r"\$\d+", stripped))


def _literal_kind(token: str) -> str | None:
    stripped = token.strip()
    if re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
        return "numeric"
    if stripped.lower() in {"true", "false"}:
        return "boolean"
    if (stripped.startswith("'") and stripped.endswith("'")) or (
        stripped.startswith('"') and stripped.endswith('"')
    ):
        inner = stripped[1:-1]
        if DATE_LITERAL_RE.fullmatch(inner):
            return "date"
        return "text"
    return None


def _literal_tokens(token: str) -> list[str]:
    stripped = token.strip()
    if _is_parameter_token(stripped):
        return []
    if stripped.startswith("(") and stripped.endswith(")"):
        inner = stripped[1:-1].strip()
        if not inner or re.search(r"\bselect\b", inner, flags=re.IGNORECASE):
            return []
        parts = [part.strip() for part in inner.split(",") if part.strip()]
        return [part for part in parts if _literal_kind(part) is not None]
    return [stripped] if _literal_kind(stripped) is not None else []


def _resolve_column_type(
    column_ref: str, alias_map: dict[str, str], schema: SchemaCatalog
) -> tuple[str, str] | None:
    if "." in column_ref:
        prefix, column = column_ref.split(".", 1)
        table = alias_map.get(prefix.lower())
        if table is None:
            return None
        column_type = schema.column_type(table, column)
        if column_type is None:
            return None
        return table, column_type

    referenced_tables = {table for table in alias_map.values() if schema.has_table(table)}
    return schema.resolve_unqualified_column(column_ref, referenced_tables)


def _type_domain(column_type: str) -> str | None:
    normalized = column_type.lower()
    if any(token in normalized for token in ("int", "numeric", "decimal", "real", "double", "float")):
        return "numeric"
    if any(token in normalized for token in ("char", "text", "string", "uuid")):
        return "text"
    if any(token in normalized for token in ("date", "time")):
        return "date"
    if "bool" in normalized:
        return "boolean"
    return None


def _is_domain_compatible(expected_domain: str, literal_token: str) -> bool:
    literal_kind = _literal_kind(literal_token)
    if literal_kind is None:
        return True

    if expected_domain == "numeric":
        return literal_kind == "numeric"
    if expected_domain == "text":
        return literal_kind in {"text", "date"}
    if expected_domain == "date":
        return literal_kind == "date"
    if expected_domain == "boolean":
        if literal_kind == "boolean":
            return True
        if literal_kind == "numeric":
            return literal_token.strip() in {"0", "1", "-0", "+0", "+1", "-1"}
        return False
    return True


class SHRChecker:
    metric = "SHR"

    def check(self, sql: str, schema: SchemaCatalog) -> CheckResult:
        result = CheckResult(metric=self.metric)
        alias_map = _extract_alias_map(sql)
        referenced_tables = set(alias_map.values())

        for table in sorted(referenced_tables):
            if not schema.has_table(table):
                result.violations.append(
                    Violation(
                        metric=self.metric,
                        message=f"Unknown table reference: {table}",
                        fragment=table,
                    )
                )

        for qualifier, column in _extract_explicit_columns(sql):
            table = alias_map.get(qualifier)
            if table is None:
                result.violations.append(
                    Violation(
                        metric=self.metric,
                        message=f"Unknown table alias/prefix: {qualifier}",
                        fragment=f"{qualifier}.{column}",
                    )
                )
                continue
            if schema.has_table(table) and not schema.has_column(table, column):
                result.violations.append(
                    Violation(
                        metric=self.metric,
                        message=f"Unknown column {column} on table {table}",
                        fragment=f"{qualifier}.{column}",
                    )
                )

        known_tables = {table for table in referenced_tables if schema.has_table(table)}
        for predicate in _extract_predicates(sql):
            left_operand = predicate.left
            if "." in left_operand:
                continue
            if left_operand.lower() in RESERVED_WORDS:
                continue
            matching_tables = [
                table for table in known_tables if schema.has_column(table, left_operand)
            ]
            if not matching_tables and known_tables:
                result.violations.append(
                    Violation(
                        metric=self.metric,
                        message=f"Unknown unqualified column: {left_operand}",
                        fragment=left_operand,
                    )
                )

        return result


class PORChecker:
    metric = "POR"

    def check(self, sql: str, schema: SchemaCatalog) -> CheckResult:
        del schema
        result = CheckResult(metric=self.metric)

        for predicate in _extract_predicates(sql):
            literal_values = _literal_tokens(predicate.right)
            for literal in literal_values:
                result.violations.append(
                    Violation(
                        metric=self.metric,
                        message="Hardcoded literal found in predicate; prefer parameter/lookup/join",
                        fragment=f"{predicate.left} {predicate.op} {literal}",
                    )
                )

        return result


class DIVRChecker:
    metric = "DIVR"

    def check(self, sql: str, schema: SchemaCatalog) -> CheckResult:
        result = CheckResult(metric=self.metric)
        alias_map = _extract_alias_map(sql)

        for predicate in _extract_predicates(sql):
            literals = _literal_tokens(predicate.right)
            if not literals:
                continue

            resolved_column = _resolve_column_type(predicate.left, alias_map, schema)
            if resolved_column is None:
                continue

            _, column_type = resolved_column
            expected_domain = _type_domain(column_type)
            if expected_domain is None:
                continue

            for literal in literals:
                if not _is_domain_compatible(expected_domain, literal):
                    result.violations.append(
                        Violation(
                            metric=self.metric,
                            message=(
                                f"Type mismatch for {predicate.left}: expected {expected_domain}, "
                                f"found literal {literal}"
                            ),
                            fragment=f"{predicate.left} {predicate.op} {literal}",
                        )
                    )

        return result


class SJRChecker:
    metric = "SJR"

    def check(self, sql: str, schema: SchemaCatalog) -> CheckResult:
        del sql
        del schema
        return CheckResult(
            metric=self.metric,
            implemented=False,
            notes="TODO: semantic join relevance checker not implemented yet.",
        )


class FDVRChecker:
    metric = "FDVR"

    def check(self, sql: str, schema: SchemaCatalog) -> CheckResult:
        del sql
        del schema
        return CheckResult(
            metric=self.metric,
            implemented=False,
            notes="TODO: functional dependency violation checker not implemented yet.",
        )


class ConstraintSuite:
    def __init__(self, checkers: list[ConstraintChecker] | None = None) -> None:
        self.checkers = checkers or [
            SHRChecker(),
            PORChecker(),
            DIVRChecker(),
            SJRChecker(),
            FDVRChecker(),
        ]

    def evaluate(self, sql: str, schema: SchemaCatalog) -> dict[str, CheckResult]:
        return {checker.metric: checker.check(sql, schema) for checker in self.checkers}

    def core_flags(self, sql: str, schema: SchemaCatalog) -> dict[str, bool]:
        results = self.evaluate(sql, schema)
        return {metric: results[metric].detected for metric in CORE_METRICS}
