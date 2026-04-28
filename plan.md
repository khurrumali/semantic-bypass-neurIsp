# NeurIPS Paper: Semantic Bypass in NL2SQL

## Problem Statement

Neural sequence models in NL2SQL exhibit a critical failure mode we term **semantic bypass**: models generate SQL that is *surface-plausible* (parses, executes, often returns a non-empty result on the dev split) yet violates **relational/structural constraints** that the question logically demands.

We focus on four operationally detectable violation classes:

1. **Schema-Item Fabrication** — referencing tables/columns absent from the schema (a.k.a. *schema-item hallucination*; Suhr et al. ACL 2020; Ji et al. 2023 survey).
2. **Value-Linking Bypass** — when the question requires resolving a natural-language entity through a normalized lookup table (or a multi-hop join), the model instead embeds the entity literally against a denormalized column (or against the wrong column). This is the operational form of **schema-linking failure** (Wang et al. RAT-SQL, ACL 2020; Lei et al. EMNLP 2020).
3. **Domain Integrity Violation** — predicates whose operand types are incompatible with the schema-declared column types (independent of the engine's type-coercion behavior).
4. **Foreign-Key / Join-Path Shortcut** — using a join condition that is *not* on a declared FK pair when the gold query (or the schema's FK graph) requires multi-hop traversal.

A fifth class, **GROUP BY / Functional-Dependency Violation**, is included as an *optional* probe; its definition is dialect-dependent (SQL:1999 §7.9 allows FD-based relaxation; PostgreSQL enforces strictly, SQLite/MySQL-default do not), so we will only score it under a single, explicit dialect (Postgres-strict semantics).

**Hypothesis (sharpened).** Under standard prompting, LLMs perform NL2SQL primarily as conditional language modeling over training-distribution SQL surface forms, rather than as constraint satisfaction over the schema. Consequently, surface-correct outputs systematically underperform on cases that require *constraint-respecting* generation, and this gap is *not* closed by explicit constraint instructions.

This positions the work alongside the **shortcut-learning** (Geirhos et al. *Nat. Mach. Intell.* 2020) and **spurious-correlation** (Sagawa et al. ICLR 2020) literatures, applied to a structured-output setting.

## Compute & Cost Budget (binding constraint)

**No frontier-API budget available.** All experiments must run on (a) self-hosted open-weight models via vLLM, or (b) cheap inference APIs. Concretely:

- **Qwen2.5-Coder family** (0.5B / 1.5B / 7B / 14B / 32B) — Hui et al. 2024. Same training recipe across sizes → primary scaling axis. Run locally with vLLM, or via Together AI / Fireworks (~$0.20–0.90 per M tokens at the larger sizes).
- **Gemma family** — Google DeepMind. **Gemma 2** (2B / 9B / 27B; Riviere et al. 2024) and/or **Gemma 3** (1B / 4B / 12B / 27B; Gemma Team 2025). Provides a *second* same-recipe scaling axis, lets us replicate Phase 4 across two model families, and adds a non-Qwen, non-Llama lineage to the cross-family check. Open weights, run locally with vLLM. *Use base/instruct, not CodeGemma* — code-specialized variants would confound the "general LM treats SQL as text" thesis.
- **Llama-3.1-8B / 70B** — third family for cross-family generalization. Available cheaply on Together / Groq.
- **DeepSeek-Coder-V2 / DeepSeek-V3** API — DeepSeek-AI 2024. Acts as the "frontier-equivalent" anchor at ~$0.14 / $0.28 per M input/output tokens (≈5% of GPT-4-class pricing) and is competitive on Spider/BIRD leaderboards.
- **No GPT-4 / Claude / Gemini API spend.** If a single frontier reference run is needed for the camera-ready, scope it to the Phase 2 sample only and quote the cost in the paper.
- **Excluded by design**: SQL-fine-tuned models (CodeS, SQLCoder, defog) — they would muddy the "general LM treats SQL as text" hypothesis. Mention as a non-target in scoping.

**Budget envelope (target)**: ≤$200 in API spend across all phases; remainder on local vLLM inference. Token-cost estimates per phase are tracked in `results/cost_log.jsonl`.

**Sample-size adjustment under budget**: with the model count reduced and self-hosted inference dominant, the original n=300 power target is retained for Phase 2 (it's free locally). Where API spend dominates (DeepSeek-V3 anchor runs), use n=200 — Wilson 95% CI half-width on a 10% rate is ±4.3pp, still usable for the headline.

## Related Work & Positioning (to be expanded in paper)

- **Benchmarks**: Spider (Yu et al. EMNLP 2018), Spider 2.0 (Lei et al. 2024), BIRD (Li et al. NeurIPS 2023), EHR-SQL (Lee et al. NeurIPS-D&B 2022), MIMIC-SQL (Wang et al. 2020).
- **Evaluation**: *Test-Suite Execution Accuracy* (Zhong et al. EMNLP 2020) — addresses the well-known false-positive problem of single-DB execution accuracy.
- **Schema linking**: RAT-SQL (Wang et al. 2020), Lei et al. EMNLP 2020 (re-examination), IRNet (Guo et al. 2019).
- **Constrained decoding**: PICARD (Scholak et al. EMNLP 2021).
- **LLM prompting baselines**: CoT (Wei et al. 2022), DIN-SQL (Pourreza & Rafiei NeurIPS 2023), MAC-SQL (Wang et al. 2024), C3 (Dong et al. 2023).
- **Error taxonomies**: Suhr et al. ACL 2020, Lei et al. EMNLP 2020.
- **Scaling behavior**: BIG-Bench (Srivastava et al. 2022), Inverse Scaling Prize (McKenzie et al. 2023) — required citation for any "bigger-is-worse" claim.

**Novelty claim**: prior NL2SQL error analyses are largely accuracy-decomposition exercises on a fixed model. We (i) define a *constraint-violation* axis orthogonal to execution accuracy; (ii) measure its rate *conditional on* test-suite execution success (the under-reported failure mode); (iii) probe its prompt- and scale-sensitivity.

## Core Contributions

1. **Operational definitions** of four constraint-violation classes detectable from the generated SQL + schema alone, plus an optional FD-based fifth.
2. **Validated detectors** (precision *and* recall on hand-labeled examples; Cohen's κ inter-annotator agreement reported).
3. **Empirical quantification** on Spider and BIRD, conditional on test-suite execution accuracy (Zhong et al. 2020) — the headline number is the rate of *execution-correct* queries that nonetheless violate ≥1 constraint.
4. **Prompt-sensitivity ablation** against documented baselines (vanilla, CoT, DIN-SQL-style decomposition, schema-linking-augmented).
5. **Scale analysis** with a pre-registered hypothesis form (monotonic / U-shaped / null), grounded in the inverse-scaling literature; we do *not* presuppose direction.

## Metrics

### Constraint-Violation Rates
| ID | Name | Tier | Operational definition | Notes |
|----|------|------|------------------------|-------|
| **SIF** | Schema-Item Fabrication Rate | core | Generated query references a table or qualified column not in the schema. | Replaces "SHR"; aligns with Suhr et al. terminology. |
| **VLB** | Value-Linking Bypass Rate | **headline** | Question entity resolvable only via lookup/multi-hop is instead matched as a literal against a non-key text column, OR matched against the wrong column. Detector requires (a) a *value-link spec* per example (gold join path), (b) string-literal extraction. | Replaces "POR"; framing aligned with Wang et al. 2020 / Lei et al. 2020. |
| **DIV** | Domain Integrity Violation Rate | core | Predicate operand vs. schema-declared column type are incompatible (numeric vs. string, date vs. string, enum vs. arbitrary text). Engine-independent: checked against schema metadata, not SQLite affinity. | |
| **FKS** | Foreign-Key Shortcut Rate | secondary | A JOIN's ON-clause uses a column pair *not* in the declared FK set, where the gold query traverses ≥1 declared FK. | Uses Spider/BIRD FK metadata. |
| **FDV** | Functional-Dependency Violation Rate | optional | Postgres-strict semantics: SELECT contains a non-aggregated, non-grouped column with no implied FD from grouped keys. | Single-dialect only. |

A query may violate multiple categories; we report (a) per-class marginal rates, (b) the **any-violation** rate, and (c) the **violation-conditional-on-execution-correct** rate (the headline).

### Accuracy Metrics (ground truth axis)
- **EX** — single-DB execution accuracy (Spider/BIRD default).
- **TS** — **Test-Suite Execution Accuracy** (Zhong et al. 2020). *This is the primary accuracy axis*; EX is reported for backward comparability only.
- **EM** — exact set-match (string-form), reported for completeness.

The headline finding is computed against **TS**, not EX, to control for the Zhong et al. false-positive problem.

## Experimental Design

### Phase 0 (NEW): Detector Validation
**Goal**: Establish detector precision *and* recall before any downstream claim.

- **Audit set**: 200 hand-labeled examples (50 per core metric: SIF, VLB, DIV, FKS), drawn from Spider dev predictions of one mid-tier model. Balanced positive/negative.
- **Two annotators** label independently; report **Cohen's κ**. Disagreements adjudicated.
- **Targets**: precision ≥0.90 *and* recall ≥0.80 on SIF, VLB, DIV. FKS may be lower (≥0.80/0.70).
- **Output**: confusion matrices per detector; if targets miss, iterate detector before Phase 1.

### Phase 1: Synthetic Constraint Probes
**Goal**: Isolate each violation class from confounds (data scarcity, ambiguity).

- **Synthetic set**: 50 examples per class (250 total), templated NL → schema with planted violation opportunity.
- **Use**: detector sanity (recall floor) and *not* as a primary empirical claim.
- **Models**: 4 cost-aware: Qwen2.5-Coder-7B (local), Gemma-2-9B (local), Llama-3.1-8B (local), DeepSeek-Coder-V2 (cheap API anchor).

### Phase 2: Spider + BIRD Quantification (CORE)
**Goal**: Quantify violation rates on natural NL2SQL data, conditional on TS-accuracy.

- **Data**:
  - **Spider dev** (Yu et al. 2018), stratified random sample of n=300 (power-justified — see below).
  - **BIRD dev** (Li et al. NeurIPS 2023), stratified n=300. *BIRD replaces eICU as the primary cross-distribution test*: it is purpose-built for value-linking and FK-shortcut failures and is the standard "harder" benchmark.
- **Power analysis (pre-registered)**: assuming a 10% baseline violation rate, detecting a 5-pp shift between two conditions at α=0.05, β=0.20 requires n≈300 per arm (two-proportion z-test). n=50–100 in the original plan was underpowered.
- **Models**: 4 LLMs — Qwen2.5-Coder-32B (local), Gemma-2-27B (local), Llama-3.1-70B (Together/Groq), DeepSeek-V3 (cheap API anchor). All non-frontier-priced; total Phase-2 API spend budgeted ≤$80.
- **Pipeline**: generate SQL → run TS-accuracy harness (Zhong et al. 2020 distilled test suites; available for Spider) → run all 4 core detectors → cross-tabulate.
- **Headline output**: P(any violation | TS-correct), with 95% Wilson CI.
- **Secondary**: per-class rates; joint distribution; per-database breakdown.

### Phase 3: Prompt-Sensitivity Ablation
**Goal**: Test whether documented prompting interventions reduce bypass.

Replace ad-hoc prompts with **literature baselines**:
1. **Vanilla** zero-shot prompt (Spider standard).
2. **CoT** (Wei et al. 2022) — "think step by step".
3. **Schema-linking-augmented** prompt (à la Lei et al. 2020 / RESDSQL) — pre-extract relevant tables/columns and inject.
4. **DIN-SQL-style decomposition** (Pourreza & Rafiei NeurIPS 2023) — schema linking → classification → generation → self-correction.
5. **Constraint-explicit** prompt — our intervention: enumerate the 4 violation classes verbatim with negative examples.

- **Data**: held-out subset of Phase 2 data (n=200, same items across prompts → paired analysis).
- **Statistic**: McNemar's test on paired violation flags; report effect size with 95% CI.
- **Falsification**: if (4) or (5) materially reduces VLB on TS-correct queries (Δ ≥ 5pp, p<0.01), the "fundamental" framing is weakened — this would be a finding, not a failure.

### Phase 4: Scale Sensitivity
**Goal**: Pre-registered test of the model-size relationship to bypass.

- **Models**: two same-recipe scaling axes, both open-weight, run locally via vLLM:
  - **Qwen2.5-Coder** at 5 sizes (0.5B / 1.5B / 7B / 14B / 32B; Hui et al. 2024) — primary axis.
  - **Gemma 2** at 3 sizes (2B / 9B / 27B; Riviere et al. 2024) — *replication axis*. A scaling pattern that holds across two independently trained families is materially stronger evidence than a single-family curve.
  - Optional: Gemma-3 (1B/4B/12B/27B; Gemma Team 2025) for a 4-point cross-check, and Qwen-72B via Together AI if budget permits.
- Pre-registered hypothesis is evaluated *per family* and *jointly* (mixed-effects regression with family as random intercept).
- **Data**: same Phase 2 sample (n=300 Spider + 300 BIRD).
- **Hypothesis space (pre-registered, not pre-ordained)**:
  - H0: violation rate constant across scale.
  - H1a: monotonic decrease (alignment with TS-accuracy).
  - H1b: monotonic increase ("inverse scaling" à la McKenzie et al. 2023).
  - H1c: U-shaped.
- **Test**: log-log regression of violation rate on parameter count, plus model-vs.-model pairwise tests with Holm correction.
- **N≥4 sizes** is required for any regression claim; N=3 (the original plan) is insufficient.

### Phase 5: Cross-Distribution Validation
**Goal**: Test transfer of the headline finding.

- **Datasets**:
  - **BIRD** (already in Phase 2 — used here for the *transfer* claim).
  - **EHR-SQL** (Lee et al. NeurIPS-D&B 2022) — real clinical NL2SQL benchmark; replaces vague "eICU" reference. eICU-CRD itself is a *database*, not an NL2SQL benchmark — citing it as a benchmark would be a category error.
  - **Spider 2.0-Lite** (Lei et al. 2024) — newer enterprise distribution.
- **Sample**: n=150 per dataset (power-justified for detecting Δ≥10pp vs. Spider baseline).
- **Models**: best Phase 4 performer per family (likely Qwen2.5-Coder-32B and Gemma-2-27B) + DeepSeek-V3 anchor. No frontier API.

## Datasets Summary

| Phase | Dataset | n | Citation | Status |
|-------|---------|---|----------|--------|
| 0 | Spider dev (audit) | 200 | Yu et al. 2018 | have |
| 1 | Synthetic | 250 | own | generate |
| 2 | Spider dev | 300 | Yu et al. 2018 | have |
| 2 | BIRD dev | 300 | Li et al. 2023 | acquire |
| 3 | Phase-2 paired subset | 200 | — | reuse |
| 4 | Phase-2 sample | 600 | — | reuse |
| 5 | EHR-SQL | 150 | Lee et al. 2022 | acquire |
| 5 | Spider 2.0-Lite | 150 | Lei et al. 2024 | acquire |

## Implementation Artifacts

### Core modules
1. **`detectors/`** — one file per class (`sif.py`, `vlb.py`, `div.py`, `fks.py`, `fdv.py`). Each exposes `check(parsed_sql, schema, gold=None) -> Violation`. Use a real SQL parser (**`sqlglot`**, dialect-aware) — *not* regex over query strings.
2. **`evaluators/test_suite.py`** — wrapper around Zhong et al.'s test-suite harness; primary accuracy axis.
3. **`evaluators/exec_accuracy.py`** — Spider/BIRD official EX (legacy).
4. **`prompts/`** — one file per Phase 3 baseline; version-pinned.
5. **`models/`** — unified async client (OpenAI / Anthropic / vLLM); structured logging of (prompt, response, model, version, seed, timestamp).
6. **`spider_utils.py`**, **`bird_utils.py`**, **`ehrsql_utils.py`** — loaders, FK-graph extraction, stratified sampling.
7. **`runner.py`** — phase orchestration, deterministic seeding, JSON-line result logs.
8. **`analysis/`** — Wilson CIs, McNemar paired tests, log-log scaling regression, plots.

### Infrastructure fix
- **`db.py` currently targets PostgreSQL; Spider/BIRD ship as SQLite.** Either replace with a SQLite loader for benchmark execution, or scope `db.py` strictly to the Phase 4 *FDV* dialect run (Postgres-strict). Currently the file's purpose is undefined — flag for refactor.

### Data structures
- **Per-prediction record**: `{example_id, dataset, model, prompt_id, sql, parsed_ast, ts_correct, ex_correct, em, sif, vlb, div, fks, fdv, latency_ms, seed}`.
- **Aggregate**: bootstrap & Wilson CIs; effect sizes (Cohen's h for proportions, log-OR for paired).

## Project Structure

```
project_files/
├── README.md
├── requirements.txt        # add: sqlglot, sqlite-utils, datasets, statsmodels
├── .env.example
├── db.py                   # SCOPE: refactor or restrict to Postgres FDV run
├── data/
│   ├── synthetic/
│   ├── spider/
│   ├── bird/
│   ├── ehrsql/
│   └── spider2_lite/
├── src/
│   ├── detectors/{sif,vlb,div,fks,fdv}.py
│   ├── evaluators/{test_suite,exec_accuracy}.py
│   ├── prompts/{vanilla,cot,schema_linking,din_sql,constraint_explicit}.py
│   ├── models/client.py
│   ├── spider_utils.py
│   ├── bird_utils.py
│   ├── ehrsql_utils.py
│   ├── runner.py
│   └── analysis/
├── notebooks/
│   ├── 00_detector_audit.ipynb        # Phase 0 (κ, P/R)
│   ├── 01_synthetic.ipynb             # Phase 1
│   ├── 02_spider_bird.ipynb           # Phase 2 (CORE)
│   ├── 03_prompt_ablation.ipynb       # Phase 3
│   ├── 04_scale.ipynb                 # Phase 4
│   └── 05_cross_distribution.ipynb    # Phase 5
├── results/                # JSONL per-prediction + aggregate JSON
└── paper/
```

## Critical Implementation Notes

### Headline finding (sharpened)
Not "POR is high" but: **P(constraint violation | TS-execution-correct) on Spider and BIRD**, with Wilson 95% CI, computed across multiple models and prompts. This is the cell that prior accuracy-only evaluation hides.

### VLB (value-linking bypass) — sharpened definition
Per example, define a **value-link spec**: the set of (entity-mention, required-resolution-path) pairs derivable from the gold SQL or schema. A VLB violation is then: an entity mention whose required path involves ≥1 join *not* present in the prediction, *and* whose surface form appears as a string literal against a non-key column. This:
- avoids the false positive "literal in WHERE clause is bad" (it isn't, in general),
- aligns with schema-linking literature (Wang et al. 2020; Lei et al. 2020),
- is detectable from `(parsed_ast, schema, gold_ast)`.

### Avoid term overload
Use **constraint violation**, not "hallucination" — except for SIF, where "schema-item hallucination" is the literature's term and we should cite Ji et al. 2023 + Suhr et al. 2020.

### Statistical rigor
- Wilson 95% CIs on all rate point estimates (better small-n behavior than normal approx).
- Paired tests (McNemar) for prompt ablation, since the same items are scored across prompts.
- Holm or Benjamini-Hochberg for multi-prompt / multi-model comparisons.
- Pre-register Phase 4 hypothesis form before running the regression.
- Bootstrap CIs only when distributional assumptions of the closed-form fail.

### Reproducibility
- Pin model versions (provider model IDs + dates), prompt-template hashes, and seeds.
- Distill Spider test suites locally and check hashes against Zhong et al. release.
- Release JSONL prediction logs + analysis notebooks.

### Threats to validity (write into the paper)
1. **Detector error** — bounded by Phase 0 audit; report worst-case rate corrections under recall=lower-bound.
2. **Synthetic ↔ natural distribution mismatch** — Phase 1 used only for sanity, no headline claim.
3. **Single-family scaling** — Phase 4 conclusions strictly about that family; cross-family is future work.
4. **TS-accuracy false negatives** — Zhong et al. report residual rate; we report sensitivity to it.
5. **Prompt-sensitivity floor** — we cannot claim "no prompt fixes this", only "these documented prompts don't".
6. **No frontier-model coverage** — explicit limitation. We anchor to DeepSeek-V3 as the closest cheap proxy and discuss what frontier-model behavior could change. Reviewers may ask; pre-empt with a single-prompt sanity run on a small (n≈50) subsample if any free credits become available.

## Success Criteria (revised)

1. ✅ Detectors validated: P≥0.90 / R≥0.80 on SIF, VLB, DIV; κ≥0.7 inter-annotator (Phase 0).
2. ✅ Headline rate **P(any violation | TS-correct)** reported on Spider and BIRD with 95% CI, ≥3 models.
3. ✅ Prompt-ablation paired analysis: report effect of best documented baseline (DIN-SQL or schema-linking-augmented) on the headline; finding is *informative either way*.
4. ✅ Scale relationship reported as one of {monotonic↑, monotonic↓, U, null} with regression CI; pre-registered.
5. ✅ Transfer to BIRD + EHR-SQL + Spider 2.0-Lite reported with cross-dataset deltas.
6. ✅ Limitations section explicitly addresses the five threats above.

---

**Immediate next steps**:
1. Refactor `db.py` (SQLite loader for Spider/BIRD; Postgres reserved for FDV).
2. Pull Zhong et al. distilled test suites; verify hashes.
3. Implement `detectors/sif.py` and `detectors/div.py` (cheapest); hand-label Phase 0 audit set in parallel.
4. Acquire BIRD dev set (Hugging Face: `bird-bench`).
