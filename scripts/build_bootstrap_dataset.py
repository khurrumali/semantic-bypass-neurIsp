from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_bypass import (
    get_run_log_path,
    log_dict,
    log_line,
    log_section,
    setup_logger,
)
from semantic_bypass.bootstrap import write_bootstrap_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the bootstrap NL2SQL validation dataset (exactly 50 examples)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "bootstrap_labeled_sql.jsonl",
        help="Output JSONL file path.",
    )
    args = parser.parse_args()

    log_path = get_run_log_path(phase="bootstrap")
    logger = setup_logger("bootstrap", log_file=log_path)

    log_section(logger, "Build Bootstrap Dataset")
    log_line(logger, f"Output: {args.output}")

    count = write_bootstrap_dataset(args.output)
    log_dict(logger, {"count": count})
    logger.info("Bootstrap dataset created")

    print(f"Wrote {count} bootstrap examples to {args.output}")
    print("Logged to:", log_path)


if __name__ == "__main__":
    main()
