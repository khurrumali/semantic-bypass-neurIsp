from __future__ import annotations

import os
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "auto"
    timeout_seconds: int = 120
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-haiku-latest"
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_base_url: str = "http://localhost:11434"
    openrouter_model: str = "qwen/qwen-2.5-72b-instruct"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    @classmethod
    def from_env(cls) -> "LLMClient":
        config = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "auto").strip().lower(),
            timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            openrouter_model=os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct"),
            openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        return cls(config=config)

    def _resolved_provider(self) -> str:
        provider = self.config.provider
        if provider in {"openai", "anthropic", "mock", "ollama", "openrouter"}:
            return provider
        if provider != "auto":
            raise ValueError(f"Unsupported LLM provider: {provider}")
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.getenv("OPENROUTER_API_KEY"):
            return "openrouter"
        return "mock"

    def resolved_provider(self) -> str:
        return self._resolved_provider()

    def generate_sql(
        self,
        question: str,
        schema_hint: str,
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str:
        provider = self._resolved_provider()
        resolved_system_prompt = system_prompt or self._system_prompt()
        resolved_user_prompt = user_prompt or self._user_prompt(question, schema_hint)
        if provider == "openai":
            return self._generate_openai(
                system_prompt=resolved_system_prompt,
                user_prompt=resolved_user_prompt,
            )
        if provider == "anthropic":
            return self._generate_anthropic(
                system_prompt=resolved_system_prompt,
                user_prompt=resolved_user_prompt,
            )
        if provider == "ollama":
            return self._generate_ollama(
                system_prompt=resolved_system_prompt,
                user_prompt=resolved_user_prompt,
            )
        if provider == "openrouter":
            return self._generate_openrouter(
                system_prompt=resolved_system_prompt,
                user_prompt=resolved_user_prompt,
            )
        return self._generate_mock(question)

    def _system_prompt(self) -> str:
        return (
            "You are an NL2SQL assistant. Produce exactly one SQL query and no explanation. "
            "Prefer parameterized predicates over hardcoded values when possible."
        )

    def _user_prompt(self, question: str, schema_hint: str) -> str:
        return f"Question: {question}\n\nSchema:\n{schema_hint}\n\nReturn SQL only."

    def _generate_openai(self, *, system_prompt: str, user_prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.openai_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "temperature": 0,
            },
            timeout=self.config.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenAI request failed ({response.status_code}): {response.text[:300]}"
            )
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        return _strip_code_fences(str(content))

    def _generate_anthropic(self, *, system_prompt: str, user_prompt: str) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for provider=anthropic")

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.config.anthropic_model,
                "max_tokens": 512,
                "temperature": 0,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            },
            timeout=self.config.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Anthropic request failed ({response.status_code}): {response.text[:300]}"
            )
        payload = response.json()
        content_blocks = payload.get("content", [])
        if not content_blocks:
            raise RuntimeError("Anthropic response did not include SQL content")
        content = content_blocks[0].get("text", "")
        return _strip_code_fences(content)

    def _generate_ollama(self, *, system_prompt: str, user_prompt: str) -> str:
        base_url = self.config.ollama_base_url
        model = self.config.ollama_model

        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "stream": False,
            },
            timeout=self.config.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Ollama request failed ({response.status_code}): {response.text[:300]}"
            )
        payload = response.json()
        content = payload.get("message", {}).get("content", "")
        return _strip_code_fences(content)

    def _generate_openrouter(self, *, system_prompt: str, user_prompt: str) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for provider=openrouter")

        base_url = self.config.openrouter_base_url
        model = self.config.openrouter_model

        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/anomalyco/semantic-bypass-neurISP",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
            },
            timeout=self.config.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter request failed ({response.status_code}): {response.text[:300]}"
            )
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        return _strip_code_fences(content)

    def _generate_mock(self, question: str) -> str:
        normalized = question.lower()
        if "department" in normalized:
            return (
                "SELECT e.name, d.name AS department_name "
                "FROM employees e JOIN departments d ON e.department_id = d.id "
                "WHERE d.id = :department_id;"
            )
        if "salary" in normalized:
            return "SELECT name, salary FROM employees WHERE salary > :min_salary ORDER BY salary DESC;"
        if "country" in normalized:
            return (
                "SELECT e.name, c.name AS country_name "
                "FROM employees e JOIN countries c ON e.country_code = c.code "
                "WHERE c.code = :country_code;"
            )
        return "SELECT id, name FROM employees LIMIT 10;"


def _strip_code_fences(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        stripped = text.strip("`")
        lines = [line for line in stripped.splitlines() if line.lower().strip() != "sql"]
        return "\n".join(lines).strip()
    return text
