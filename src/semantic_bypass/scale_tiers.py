from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import re

from .llm import LLMClient, LLMConfig

TIER_ORDER: tuple[str, ...] = ("small", "medium", "large")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ModelTierConfig:
    tier: str
    descriptor: str
    provider: str = "auto"
    openai_model: str | None = None
    anthropic_model: str | None = None
    ollama_model: str | None = None
    proxy_shr_rate: float = 0.0
    proxy_por_rate: float = 0.0
    proxy_divr_rate: float = 0.0


def load_model_tier_configs(provider_override: str | None = None) -> list[ModelTierConfig]:
    base_configs = {
        "small": ModelTierConfig(
            tier="small",
            descriptor="proxy-small local tier (high-noise baseline)",
            provider="auto",
            openai_model="gpt-4o-mini",
            anthropic_model="claude-3-5-haiku-latest",
            ollama_model="qwen2.5-coder:7b",
            proxy_shr_rate=0.28,
            proxy_por_rate=0.56,
            proxy_divr_rate=0.20,
        ),
        "medium": ModelTierConfig(
            tier="medium",
            descriptor="proxy-medium local tier (moderate-noise baseline)",
            provider="auto",
            openai_model="gpt-4o-mini",
            anthropic_model="claude-3-5-haiku-latest",
            ollama_model="qwen2.5-coder:7b",
            proxy_shr_rate=0.18,
            proxy_por_rate=0.40,
            proxy_divr_rate=0.12,
        ),
        "large": ModelTierConfig(
            tier="large",
            descriptor="proxy-large local tier (lower-noise baseline)",
            provider="auto",
            openai_model="gpt-4o-mini",
            anthropic_model="claude-3-5-haiku-latest",
            ollama_model="qwen2.5-coder:7b",
            proxy_shr_rate=0.10,
            proxy_por_rate=0.26,
            proxy_divr_rate=0.07,
        ),
    }

    configs: list[ModelTierConfig] = []
    for tier in TIER_ORDER:
        base = base_configs[tier]
        env_prefix = f"PHASE4_{tier.upper()}"
        provider = (
            (provider_override or os.getenv(f"{env_prefix}_PROVIDER") or base.provider)
            .strip()
            .lower()
        )
        descriptor = os.getenv(f"{env_prefix}_DESCRIPTOR", base.descriptor).strip() or base.descriptor
        openai_model = os.getenv(f"{env_prefix}_OPENAI_MODEL", base.openai_model or "").strip() or None
        anthropic_model = (
            os.getenv(f"{env_prefix}_ANTHROPIC_MODEL", base.anthropic_model or "").strip() or None
        )
        ollama_model = (
            os.getenv(f"{env_prefix}_OLLAMA_MODEL", base.ollama_model or "").strip() or None
        )
        configs.append(
            ModelTierConfig(
                tier=tier,
                descriptor=descriptor,
                provider=provider,
                openai_model=openai_model,
                anthropic_model=anthropic_model,
                ollama_model=ollama_model,
                proxy_shr_rate=base.proxy_shr_rate,
                proxy_por_rate=base.proxy_por_rate,
                proxy_divr_rate=base.proxy_divr_rate,
            )
        )
    return configs


def provider_available(provider: str) -> bool:
    normalized = provider.strip().lower()
    if normalized == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if normalized == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if normalized == "ollama":
        return True
    if normalized == "auto":
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
    if normalized == "mock":
        return True
    return False


def build_tier_llm_client(config: ModelTierConfig) -> LLMClient:
    return LLMClient(
        config=LLMConfig(
            provider=config.provider,
            openai_model=config.openai_model or "gpt-4o-mini",
            anthropic_model=config.anthropic_model or "claude-3-5-haiku-latest",
            ollama_model=config.ollama_model or "qwen2.5-coder:7b",
        )
    )


class DeterministicTierProxyClient:
    def __init__(self, config: ModelTierConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = seed

    def generate_sql(self, question: str, schema_hint: str) -> str:
        schema = _parse_schema_hint(schema_hint)
        table = self._pick_table(schema, question)
        columns = schema.get(table, {"id": "numeric"}) if table else {"id": "numeric"}
        select_column = self._pick_column(columns, question, "select")
        predicate_column = self._pick_column(columns, question, "predicate")
        predicate_type = columns.get(predicate_column, "text")

        flag_shr = self._should_trigger("SHR", question, self.config.proxy_shr_rate)
        flag_por = self._should_trigger("POR", question, self.config.proxy_por_rate)
        flag_divr = self._should_trigger("DIVR", question, self.config.proxy_divr_rate)

        select_expr = f"t.{select_column}"
        if flag_shr:
            select_expr = f"{select_expr}, t.__proxy_missing_column__"

        query = f"SELECT {select_expr} FROM {table} t"
        if flag_divr:
            predicate = f"t.{predicate_column} = {self._incompatible_literal(predicate_type)}"
        elif flag_por:
            predicate = f"t.{predicate_column} = {self._compatible_literal(predicate_type)}"
        else:
            predicate = f"t.{predicate_column} = :{predicate_column}_param"
        query += f" WHERE {predicate} LIMIT 10;"
        return query

    def _stable_unit(self, *parts: str) -> float:
        payload = "|".join(parts) + f"|seed={self.seed}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int(digest[:12], 16) / float(16**12 - 1)

    def _should_trigger(self, metric: str, question: str, threshold: float) -> bool:
        return self._stable_unit(self.config.tier, metric, question) < threshold

    def _pick_table(self, schema: dict[str, dict[str, str]], question: str) -> str:
        candidates = [table for table in sorted(schema) if _IDENTIFIER_RE.fullmatch(table)]
        if not candidates:
            return "employees"
        index = int(self._stable_unit(self.config.tier, question, "table") * len(candidates))
        return candidates[min(index, len(candidates) - 1)]

    def _pick_column(self, columns: dict[str, str], question: str, role: str) -> str:
        candidates = [column for column in sorted(columns) if _IDENTIFIER_RE.fullmatch(column)]
        if not candidates:
            return "id"
        index = int(self._stable_unit(self.config.tier, question, role) * len(candidates))
        return candidates[min(index, len(candidates) - 1)]

    def _compatible_literal(self, column_type: str) -> str:
        domain = _type_domain(column_type)
        if domain == "numeric":
            return "1"
        if domain == "date":
            return "'2024-01-01'"
        if domain == "boolean":
            return "true"
        return "'proxy_value'"

    def _incompatible_literal(self, column_type: str) -> str:
        domain = _type_domain(column_type)
        if domain == "numeric":
            return "'proxy_bad'"
        if domain == "date":
            return "99"
        if domain == "boolean":
            return "'proxy_bad'"
        return "999"


def _parse_schema_hint(schema_hint: str) -> dict[str, dict[str, str]]:
    schema: dict[str, dict[str, str]] = {}
    for raw_line in schema_hint.splitlines():
        line = raw_line.strip()
        if not line.startswith("- ") or "(" not in line or not line.endswith(")"):
            continue
        table_name, column_blob = line[2:].split("(", 1)
        table = table_name.strip()
        if not table:
            continue
        columns: dict[str, str] = {}
        for item in column_blob[:-1].split(","):
            piece = item.strip()
            if not piece or ":" not in piece:
                continue
            column, column_type = piece.split(":", 1)
            column_name = column.strip()
            if not column_name:
                continue
            columns[column_name] = column_type.strip().lower() or "text"
        if columns:
            schema[table] = columns
    return schema


def _type_domain(column_type: str) -> str:
    normalized = column_type.lower()
    if any(token in normalized for token in ("int", "numeric", "decimal", "real", "double", "float")):
        return "numeric"
    if any(token in normalized for token in ("date", "time")):
        return "date"
    if "bool" in normalized:
        return "boolean"
    return "text"
