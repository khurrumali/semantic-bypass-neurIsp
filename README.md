# Semantic Bypass NeurISP

Minimal NL2SQL experiment scaffold with semantic-bypass constraint checks.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project layout

- `db.py` – PostgreSQL connectivity utility.
- `src/semantic_bypass/` – pipeline, LLM wrapper, and constraint checkers.
- `src/semantic_bypass/prompting.py` – prompt-variant templates for Phase 3 ablations.
- `scripts/run_baseline.py` – run one NL2SQL inference + checks.
- `scripts/build_bootstrap_dataset.py` – generate 50-example bootstrap labels.
- `scripts/validate_checker.py` – validate SHR/POR/DIVR precision/recall on labeled set.
- `scripts/run_phase1_synthetic.py` – generate and evaluate the Phase 1 synthetic probe set.
- `scripts/run_phase2_spider.py` – build deterministic Spider subset and run phase-2 checker analysis.
- `scripts/run_phase4_scale.py` – run small/medium/large scale analysis with provider-aware proxy fallback.
- `scripts/run_phase3_prompting.py` – run prompt-engineering ablation (baseline vs. constraint-explicit vs. formal-spec).
- `data/bootstrap_labeled_sql.jsonl` – bootstrap synthetic dataset format (not hand-labeled).

## LLM providers

Set `.env` or shell vars:

- `LLM_PROVIDER=mock` for local fallback.
- `LLM_PROVIDER=openai` with `OPENAI_API_KEY`.
- `LLM_PROVIDER=anthropic` with `ANTHROPIC_API_KEY`.
- `LLM_PROVIDER=auto` chooses OpenAI/Anthropic if keys exist, otherwise mock.

## Run baseline pipeline

```bash
python scripts/run_baseline.py --provider mock --question "Show employee names by department"
```

## Validation workflow (50 examples)

If real hand labels are unavailable, bootstrap a synthetic format:

```bash
python scripts/build_bootstrap_dataset.py
python scripts/validate_checker.py --dataset data/bootstrap_labeled_sql.jsonl
```

## Phase 2 Spider subset analysis

Run deterministic subset extraction (default from `../spider_data`) and checker analysis:

```bash
python scripts/run_phase2_spider.py --subset-size 500 --seed 42
```

Outputs:

- subset artifact files under `data/spider_subset/`
- phase-2 analysis JSON at `results/phase2_spider.json`

Validator requirements:

- dataset must contain exactly 50 JSONL records,
- each record must provide boolean labels for `shr`, `por`, `divr`,
- script prints precision/recall for SHR/POR/DIVR.

## Phase 1 synthetic run

```bash
python scripts/run_phase1_synthetic.py
```

Outputs:

- dataset at `data/synthetic/phase1_synthetic.jsonl`,
- aggregate metrics at `results/phase1_synthetic.json`.

## Phase 3 prompt ablation

Run prompt variants on sampled Spider + synthetic examples and evaluate checker-based bypass rates:

```bash
python scripts/run_phase3_prompting.py --provider mock --seed 42
```

Outputs:

- aggregate ablation JSON at `results/phase3_prompting.json`.

## Phase 4 scale analysis

Run tiered model-scale analysis on the available Spider subset artifact:

```bash
python scripts/run_phase4_scale.py --seed 42 --max-examples 180
```

Outputs:

- aggregate scale JSON at `results/phase4_scale.json`,
- per-tier SHR/POR/DIVR rates and SJR/FDVR status,
- simple trend/correlation summary across small/medium/large tiers.
