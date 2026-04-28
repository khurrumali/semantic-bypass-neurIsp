from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

SchemaMapping = Mapping[str, Mapping[str, str]]


@dataclass(frozen=True)
class PromptVariant:
    prompt_id: str
    title: str
    description: str
    system_prompt: str
    user_instruction: str

    def build_user_prompt(self, question: str, schema_hint: str) -> str:
        return (
            f"Question: {question}\n\n"
            f"Schema:\n{schema_hint}\n\n"
            f"Instructions:\n{self.user_instruction}\n\n"
            "Return SQL only."
        )


PROMPT_VARIANTS: tuple[PromptVariant, ...] = (
    PromptVariant(
        prompt_id="baseline_nl2sql",
        title="Baseline NL2SQL",
        description="Standard direct NL2SQL instruction with no explicit constraint checklist.",
        system_prompt=(
            "You are an NL2SQL assistant. Produce exactly one SQL query and no explanation."
        ),
        user_instruction=(
            "Translate the question into a single executable SQL query using the provided schema."
        ),
    ),
    PromptVariant(
        prompt_id="constraint_explicit",
        title="Constraint-Explicit",
        description="Explicitly names SHR/POR/DIVR constraints before generation.",
        system_prompt=(
            "You are an NL2SQL assistant. Produce exactly one SQL query and no explanation. "
            "Before finalizing, satisfy these hard constraints: "
            "SHR (schema references must exist), "
            "POR (avoid hardcoded predicate literals when a parameter can be used), and "
            "DIVR (literal/type compatibility in predicates)."
        ),
        user_instruction=(
            "Generate SQL that obeys SHR/POR/DIVR. "
            "Use named placeholders (for example :value) for runtime values instead of hardcoded literals."
        ),
    ),
    PromptVariant(
        prompt_id="formal_spec",
        title="Formal-Spec",
        description="Frames SQL generation as a constrained specification-satisfaction task.",
        system_prompt=(
            "You are an NL2SQL synthesis engine. Output exactly one SQL query. "
            "Treat generation as specification satisfaction under three invariants: "
            "(1) schema soundness (SHR), "
            "(2) parameterized predicates over literals (POR), and "
            "(3) operand domain compatibility (DIVR). "
            "If uncertain, prefer a conservative query that still respects all invariants."
        ),
        user_instruction=(
            "Construct SQL that satisfies the question and all invariants. "
            "Use only declared tables/columns and parameter placeholders for external values."
        ),
    ),
)


def list_prompt_variants() -> tuple[PromptVariant, ...]:
    return PROMPT_VARIANTS


def get_prompt_variant(prompt_id: str) -> PromptVariant:
    for variant in PROMPT_VARIANTS:
        if variant.prompt_id == prompt_id:
            return variant
    raise KeyError(f"Unknown prompt variant: {prompt_id}")


def render_schema_hint(schema: SchemaMapping) -> str:
    rows: list[str] = []
    for table, columns in schema.items():
        ordered_columns = sorted(columns.items(), key=lambda item: item[0])
        column_block = ", ".join(f"{column}:{column_type}" for column, column_type in ordered_columns)
        rows.append(f"- {table}({column_block})")
    return "\n".join(rows)
