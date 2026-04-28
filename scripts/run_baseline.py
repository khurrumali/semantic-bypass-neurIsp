from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from semantic_bypass import (
    get_run_log_path,
    log_dict,
    log_line,
    log_section,
    setup_logger,
)
from semantic_bypass.bootstrap import default_schema
from semantic_bypass.llm import LLMClient, LLMConfig
from semantic_bypass.pipeline import NL2SQLPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a minimal NL2SQL baseline and constraint checks."
    )
    parser.add_argument(
        "--question",
        default="List employee names and salaries above a threshold.",
        help="Natural language query.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "openai", "anthropic", "ollama", "openrouter", "mock"],
        default="openrouter",
        help="LLM provider to use.",
    )
    args = parser.parse_args()

    log_path = get_run_log_path(phase="baseline")
    logger = setup_logger("baseline", log_file=log_path)

    log_section(logger, "Baseline Run")
    log_dict(logger, {"question": args.question, "provider": args.provider})

    llm_client = LLMClient(config=LLMConfig(provider=args.provider))
    pipeline = NL2SQLPipeline(llm_client=llm_client, logger=logger)
    run = pipeline.run(question=args.question, schema=default_schema())

    log_dict(logger, {
        "question": run.question,
        "sql": run.sql,
        "core_flags": run.core_flags,
    })
    log_line(logger, f"SQL: {run.sql}")
    for metric, detected in run.core_flags.items():
        log_line(logger, f"  {metric}: {detected}")

    logger.info("Baseline run complete")

    print(f"Question: {run.question}")
    print(f"SQL: {run.sql}")
    print("Core metric detections:")
    for metric, detected in run.core_flags.items():
        print(f"  - {metric}: {detected}")

    print("Extended checker status:")
    for metric in ("SJR", "FDVR"):
        result = run.checks[metric]
        status = "implemented" if result.implemented else "TODO"
        note = f" ({result.notes})" if result.notes else ""
        print(f"  - {metric}: {status}{note}")

    print("Logged to:", log_path)


if __name__ == "__main__":
    main()
