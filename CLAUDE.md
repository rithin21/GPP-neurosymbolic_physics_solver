# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CalcMate is a neuro-symbolic physics tutor prototype for one chapter: Motion in a Straight Line / Kinematics. The core design principle: **LLM involvement is narrowly boxed to two boundaries (extraction and narration); all physics reasoning, equation solving, unit handling, and validation is deterministic and inspectable code.** Never let an LLM call decide which equation to use, what a constraint implies, or what the numeric answer is — that logic belongs in `reasoning.py`/`knowledge_graph.py`/SymPy.

## Commands

```powershell
# Install
pip install -r requirements.txt

# Run the server (needs GROQ_API_KEY; see .env)
python -m uvicorn calcmate.server:app --reload
# then open http://127.0.0.1:8000

# Run all tests (fake extractor is injected, no network/API key needed)
python -m unittest discover -s tests

# Run a single test file / case
python -m unittest tests.test_reasoning
python -m unittest tests.test_reasoning.ReasoningTests.test_max_height_uses_graph_constraints_and_third_equation

# Regenerate the SUVAT case dataset (~200 SymPy-verified cases)
python -m scripts.generate_kinematics_cases

# Validate the full case dataset
python -m scripts.validate_dataset

# Rebuild the FAISS index (only used when CALCMATE_RETRIEVAL_BACKEND=faiss)
python -m scripts.build_case_index

# Retrain node2vec structural embeddings (after the case dataset changes)
python -m scripts.train_node2vec
```

Scripts under `scripts/` and the `train_node2vec`/`build_case_index` entry points must be run as modules (`python -m scripts.foo`), not as bare script paths — otherwise `calcmate` doesn't resolve as an importable package.

Environment variables (see `.env`, loaded via `python-dotenv` in `server.py`):
- `GROQ_API_KEY` — required for the real DSPy extractor/narrator; tests avoid this via fixture extractors.
- `GROQ_MODEL` — defaults to `llama-3.1-8b-instant`.
- `GROQ_API_BASE` — defaults to Groq's OpenAI-compatible endpoint.
- `CALCMATE_RETRIEVAL_BACKEND` — `tfidf` (default), `faiss`, or `memory`.
- `CALCMATE_USE_DSPY_NARRATOR` — set `0` to force the deterministic template narrator instead of the DSPy one.

## Architecture: the nine-phase pipeline

Everything funnels through `CalcMatePipeline.solve()` in `calcmate/pipeline.py`. Reading that file's `solve()` method top-to-bottom is the fastest way to relearn the whole system. Phases, in order:

1. **Extraction** (`calcmate/extraction.py`, `DSPyPhysicsExtractor`) — Layer 1, LLM-facing. DSPy + Groq extracts *only* raw facts: `knowns_raw`, `units`, `unknown_hint`, `domain_hint`, `matched_trigger_phrases`. The signature explicitly forbids the LLM from solving, choosing equations, or inferring hidden physics (e.g. it must not infer v=0 at max height — that's a graph constraint's job, not extraction's). Units are reported as written in the problem (not pre-converted) because `source_text` is a more trustworthy quote of the original unit than the LLM's separate `units` claim, and the two occasionally disagree.
2. **1b. SI normalization** (`pipeline.py`) — Converts every extracted quantity into SI *before* anything touches it, using `calcmate/units.py`. This is what makes "solve in SI" actually true rather than only true for problems already phrased in SI.
3. **Retrieval** (`calcmate/retrieval.py`) — Finds similar solved cases. **Deferred**: `pipeline.solve()` does *not* call this upfront. The graph solver (phases 4-7) gets the first attempt using only trigger-phrase and meta-rule constraints; retrieval only runs if that attempt raises a `ReasoningError`, immediately after a `5a_graph_solver_failed` trace entry explains why and that a case-based fallback is being attempted. Three interchangeable backends behind the `CaseRetriever` protocol, selected by `CALCMATE_RETRIEVAL_BACKEND`: `TfidfCaseRetriever` (default; sklearn TF-IDF + cosine over the merged case pool), `FaissCaseRetriever` (sentence-transformers + FAISS, falls back to in-memory on any load failure), `InMemoryCaseRetriever` (tiny structural-overlap fallback used in tests).
4. **Domain resolution + constraint resolution** (`calcmate/reasoning.py`, `PhysicsReasoner.resolve_domain` / `resolve_constraints`) — Loads the chapter's equation/constraint nodes from the knowledge graph, then fires three constraint layers in order: (4a) trigger-phrase constraints matched directly against the problem text, (4b) constraints suggested by retrieved cases — on the graph solver's first attempt there are no retrieved cases yet (see phase 3), so this layer only has anything to accept during a second, post-fallback pass; even then it's only accepted if the case's own trigger phrases are independently corroborated by this problem's text, so in practice it never contributes anything phase 4a wouldn't already have caught — (4c) meta-rule fallbacks (free fall → a = -9.8, "from rest" → u = 0). Then checks the known symbols can actually reach the target (`_has_potential_equation`) before proceeding, raising `UnderConstrainedError` if not.
5. **SymPy solving** (`reasoning.py`, `solve_with_sympy`) — Iteratively finds equations with exactly one unknown symbol and solves them with SymPy until the target is reached, a fixed-point is hit, or a 20-iteration safety cap is exceeded. Raises `ContradictionError` if solving the target equation itself yields no solution; intermediate-equation failures are skipped rather than fatal.
   - **5a. Graph solver failed** (`pipeline.py`, `solve`) — Any `ReasoningError` from phases 3-7 (under-constrained, contradiction, or failed unit validation) lands here first. Appends a `5a_graph_solver_failed` trace entry naming the target and the error, then triggers phase 3 retrieval (which was skipped until now) to feed phase 5b.
   - **5b. Case fallback** (`calcmate/fallback_solver.py`, `CaseFallbackSolver`) — Only engaged after 5a. Borrows the equation + implied values from a structurally similar solved case rather than failing outright. Candidates come from the just-run retrieval *plus* their node2vec structural neighbors (`calcmate/case_graph.py`) — cases that weren't a direct text match but sit in the same region of the equation/constraint similarity graph. Reports which case powered the answer via `Solution.fallback_case_id`. The graph solver stays authoritative when it can solve a problem; this is a safety net, not a replacement path.
6. **Path reconstruction** (`reasoning.py`, `reconstruct_solution_path`) — When multiple valid equations could reach the target, the active overlay's `equation_priority` and `blocked_methods` pick which one to present, and `show_intermediate_steps`/`structure` decide whether intermediate steps are shown.
7. **Unit validation** (`calcmate/unit_validation.py`, `UnitValidator`) — Dimensionality check via Pint if installed, else exact unit-string comparison against `UNIT_BY_SYMBOL`. A failed validation raises `ReasoningError`, which triggers case fallback (phase 5b).
   - **7b. Output unit conversion** (`pipeline.py`, `_apply_requested_output_unit`) — The target is always solved in SI first; this step only converts the already-correct SI answer into whatever unit the question asked for (detected via regex on phrases like "in km/h"), never re-derives it.
8. **Narration** (`calcmate/narration.py`, `Narrator`) — Layer 3, LLM-facing. DSPy narrator rewrites the verified `steps`/`constraints_fired` payload into student-facing prose; it is given the verified numbers and told not to add equations or change them. Falls back to a deterministic template narrator when no Groq key is set or `CALCMATE_USE_DSPY_NARRATOR=0`.
9. **Output + logging** (`pipeline.py`, `_phase_9_log`) — Builds the `Solution` object and records an `AttemptLog` via the `AttemptLogger` protocol (`NoopAttemptLogger` in memory today; `PostgresAttemptLogger` in `calcmate/postgres_logging.py` defines the production contract but has no DB wiring yet — see `schema/postgres.sql`).

Every phase appends a `PhaseTrace` (phase name, status, detail) so a full attempt can be replayed/debugged from `Solution.phase_trace` / `Solution.to_jsonable()`.

## Key data structures

- **Knowledge graph** (`calcmate/knowledge_graph.py`, data in `data/kinematics_graph.json`): a `networkx.MultiDiGraph` loaded from strict `{meta, nodes, edges}` JSON. Node types include `equation` (has `expression`, `symbols`) and `constraint` (has `trigger_phrases`, `implies`). `PhysicsKnowledgeGraph._validate_graph_data` enforces the shape and that every edge references a real node.
- **Overlays** (`calcmate/overlay.py`, data in `data/overlays/{id}_kinematics.json`): per-institution presentation rules layered on top of the same graph — `equation_priority`, `blocked_methods`, `show_intermediate_steps`, `narration_tone`, `structure`. Two exist: `ncert` and `jee`. Overlays never change what's *correct*, only which valid path/style is shown.
- **Case dataset** (`calcmate/case_graph.py`, `load_all_cases`): merges and de-duplicates (by `case_id`) `data/cases/kinematics_cases.jsonl` and `data/cases/kinematics_suvat_cases.jsonl`. This exact merged pool is shared between the default `TfidfCaseRetriever`, the `FaissCaseRetriever`'s build step, and `CaseFallbackSolver`'s node2vec expansion — they're loaded once in `CalcMatePipeline.__init__` and passed through, specifically so "similar case" means the same thing at every stage.
- **Node2vec embeddings** (`data/cases/node2vec_embeddings.json`, trained via `scripts/train_node2vec.py`): a hand-rolled biased-random-walk + PyTorch skip-gram implementation (chosen because gensim's node2vec wheel needs a C toolchain that isn't available here). These embed *structural* similarity (shared equations/constraints/target/concept), independent of the `FaissCaseRetriever`'s *text* embeddings — see the module docstring in `case_graph.py` for the distinction. Transductive: only embeds cases present at training time, so it's used to expand outward from a retrieval seed, never to embed a fresh query directly.
- **Canonical symbols/units** (`calcmate/constants.py`): `UNIT_BY_SYMBOL` is the single ground-truth SI-unit table for SUVAT symbols (`u, v, a, t, s`) and the Grade 6/7 plain-quantity symbols (`speed, distance, time`). `extraction.py` reconciles a mix of the two symbol families onto whichever one the problem's target belongs to (`_reconcile_symbol_family`) since the LLM sometimes extracts knowns from both families in the same problem.
- **Regex-based matching, not string lists** (`calcmate/text_patterns.py`): trigger phrases, symbol aliases, and target-detection are all compiled into inflection-tolerant regexes (`compile_phrase`/`compile_phrase_set`) rather than enumerated as exact accepted strings, so plural/tense/spacing variants are matched without listing every spelling. When adding a new phrase or alias, add the base form to the relevant set/dict in `constants.py` or `extraction.py` — do not hand-write a new regex.
- **SI-first unit conversion** (`calcmate/units.py`): converts a solved SI value to whatever unit the question asked for, at the very end of the pipeline only. Tries Pint, falls back to a manual conversion table. Never used to convert *inputs* mid-solve — that happens once in phase 1b.

## Working on this codebase

- If you touch equation/constraint semantics, they live in `data/kinematics_graph.json`, not in Python — the graph is the single source of truth for what equations exist and what each constraint implies.
- If you add a new accepted phrasing (trigger phrase, symbol alias, target-detection verb), extend the phrase/alias sets in `calcmate/constants.py` or `calcmate/extraction.py`; the regex tolerance in `text_patterns.py` is generic and shouldn't need new patterns per phrase.
- Case dataset changes (`data/cases/*.jsonl`) should be followed by re-running `scripts/validate_dataset.py`, and `scripts/train_node2vec.py` if you want the structural-neighbor fallback to reflect the new cases.
- Tests inject fixture extractors (see `FixtureExtractor` pattern in `tests/test_reasoning.py`) so they never need `GROQ_API_KEY` or network access — follow that pattern for new tests instead of hitting the real DSPy extractor.
