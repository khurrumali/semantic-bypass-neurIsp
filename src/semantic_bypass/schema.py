from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

SchemaMapping = Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class SchemaCatalog:
    tables: dict[str, dict[str, str]]

    @classmethod
    def from_mapping(cls, mapping: SchemaMapping) -> "SchemaCatalog":
        normalized = {
            table.lower(): {
                column.lower(): column_type.lower() for column, column_type in columns.items()
            }
            for table, columns in mapping.items()
        }
        return cls(tables=normalized)

    def has_table(self, table: str) -> bool:
        return table.lower() in self.tables

    def has_column(self, table: str, column: str) -> bool:
        return column.lower() in self.tables.get(table.lower(), {})

    def column_type(self, table: str, column: str) -> str | None:
        return self.tables.get(table.lower(), {}).get(column.lower())

    def resolve_unqualified_column(
        self, column: str, candidate_tables: set[str]
    ) -> tuple[str, str] | None:
        matches: list[tuple[str, str]] = []
        column_lower = column.lower()
        for table in candidate_tables:
            table_lower = table.lower()
            table_columns = self.tables.get(table_lower, {})
            if column_lower in table_columns:
                matches.append((table_lower, table_columns[column_lower]))
        if len(matches) == 1:
            return matches[0]
        return None
