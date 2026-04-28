from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

from .checkers import CheckResult, ConstraintSuite
from .llm import LLMClient
from .schema import SchemaCatalog


@dataclass(frozen=True)
class PipelineRun:
    question: str
    sql: str
    core_flags: dict[str, bool]
    checks: dict[str, CheckResult]


class NL2SQLPipeline:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        constraint_suite: ConstraintSuite | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient.from_env()
        self.constraint_suite = constraint_suite or ConstraintSuite()
        self.logger = logger

    def run(self, question: str, schema: Mapping[str, Mapping[str, str]]) -> PipelineRun:
        schema_catalog = SchemaCatalog.from_mapping(schema)
        schema_hint = _format_schema(schema)

        if self.logger:
            self.logger.debug(f"[INFERENCE] question={question}")
            self.logger.debug(f"[INFERENCE] schema_hint={schema_hint[:200]}...")

        sql_query = self.llm_client.generate_sql(question=question, schema_hint=schema_hint)
        if not sql_query:
            raise RuntimeError("LLM returned an empty SQL query")

        if self.logger:
            self.logger.debug(f"[INFERENCE] sql={sql_query}")

        checks = self.constraint_suite.evaluate(sql_query, schema_catalog)
        core_flags = {metric: checks[metric].detected for metric in ("SHR", "POR", "DIVR")}

        if self.logger:
            self.logger.debug(f"[RESULT] core_flags={core_flags}")

        return PipelineRun(question=question, sql=sql_query, core_flags=core_flags, checks=checks)


def _format_schema(schema: Mapping[str, Mapping[str, str]]) -> str:
    rows: list[str] = []
    for table, columns in schema.items():
        cols = ", ".join(f"{column}:{column_type}" for column, column_type in columns.items())
        rows.append(f"- {table}({cols})")
    return "\n".join(rows)
