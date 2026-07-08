# CalcMate

CalcMate is a small neuro-symbolic physics tutor prototype. The LLM-facing layers are represented as narrow interfaces, while all physics reasoning is deterministic and inspectable.

This draft includes one chapter: **Motion in a Straight Line / Kinematics**.

## What is included

- Layer 1 extraction boundary: DSPy + Groq extracts quantities, units, unknown hint, domain hint, and matched trigger phrases.
- Integrated neuro-symbolic RAG pipeline: extraction, **unit normalization**, hybrid retrieval, domain resolution, constraint resolution, **LLM planning**, SymPy solving, **LLM reasoning fallback**, path reconstruction, dimensional verification, **output-unit conversion**, narration, and logging.
- Layer 2 reasoning engine: loads the chapter graph, applies graph/CBR/meta constraints, plans an equation strategy from graph + retrieved-example context, passes chapter equations through SymPy, reconstructs the path, and verifies dimensions.
- Verified-only trust model: the planning and fallback LLM stages are advisory. The planner may only reorder graph equations; the fallback may only select existing graph constraints. Every LLM proposal is re-checked by the deterministic solver and verifier, so an LLM guess can never become an unverified answer. Genuinely unsolvable problems return a structured `was_unresolved` result instead of a wrong number.
- Hybrid retrieval: `FaissCaseRetriever` fuses semantic (embedding) and structural signals; `InMemoryCaseRetriever` fuses a lexical-semantic signal with structural overlap, so hybrid behaviour is available offline. Retrieved worked examples are injected into the planning and fallback reasoning prompts (RAG).
- Stronger verification: dimensional balance of every equation used, per-substitution unit consistency, and the final-answer dimensionality check.
- Unit awareness: every extracted quantity is normalized to SI before solving (`45 min -> 2700 s`, `36 km -> 36000 m`), and the final answer is converted into the unit the problem asks for (e.g. `in km/h`, `in kilometres per minute`). All conversions run through Pint in `calcmate/units.py`, so no per-unit conversion tables are hard-coded.
- Layer 3 narration boundary: DSPy narrator rewrites verified steps only. A faithfulness guard rejects narration that disputes or diverges from the verified answer and falls back to the deterministic template (also used for local tests).
- FAISS/Chroma-style retrieval interface with an in-memory placeholder case store.
- Optional FAISS-backed retrieval over a solved-case JSONL dataset in `data/cases/`.
- PostgreSQL logging contract and schema for attempts, law nodes, constraints, under-constrained cases, contradictions, and problem bank rows.
- One base knowledge graph in `data/kinematics_graph.json`.
- Two institution overlays:
  - `data/overlays/ncert_kinematics.json`
  - `data/overlays/jee_kinematics.json`
- Minimal FastAPI backend and plain frontend.
- PostgreSQL schema draft in `schema/postgres.sql`.
- Focused unit tests for graph reasoning, extraction, retrieval, dimensional verification, and unit conversion.

## Run

```powershell
pip install -r requirements.txt
$env:GROQ_API_KEY="your_groq_api_key_here"
python -m uvicorn calcmate.server:app --reload
```

Optional model override:

```powershell
$env:GROQ_MODEL="llama-3.1-8b-instant"
```

Optional retrieval backend:

```powershell
$env:CALCMATE_RETRIEVAL_BACKEND="faiss"
```

The extraction layer uses DSPy with Groq's OpenAI-compatible endpoint. The LLM only returns structured extraction JSON: `knowns_raw`, `units`, `unknown_hint`, `domain_hint`, and `matched_trigger_phrases`. Reasoning and computation still happen in deterministic code.

Set this to disable the DSPy narrator and use the deterministic template narrator:

```powershell
$env:CALCMATE_USE_DSPY_NARRATOR="0"
```

The reasoning stages (planning + fallback) use a stronger model on the same DSPy + Groq boundary. They are enabled only when a key is present; disable them to stay fully deterministic:

```powershell
$env:CALCMATE_USE_LLM_REASONING="0"
```

Override the reasoning model (default `llama-3.3-70b-versatile`):

```powershell
$env:CALCMATE_REASONING_MODEL="llama-3.3-70b-versatile"
```

Open:

```text
http://127.0.0.1:8000
```

## Try these

```text
A ball is thrown upward with initial velocity 20 m/s. Find the maximum height.
```

```text
A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find the final velocity.
```

```text
A train moving at 10 m/s accelerates at 2 m/s^2 over 50 m. Find the final velocity.
```

```text
A school bus travels 36 km in 45 minutes. What is the average speed in km/h?
```

## Test

```powershell
python -m unittest discover -s tests
```

Tests inject a fake extractor, so they do not need a Groq key and do not call the network.

## Architecture Notes

- Graph JSON now uses the strict top-level shape `{ "meta": ..., "nodes": ..., "edges": ... }`.
- `InMemoryCaseRetriever` is the lightweight fallback. `FaissCaseRetriever` is the real embedding-backed path.
- Put your solved examples in `data/cases/kinematics_cases.jsonl`, then run `python scripts/build_case_index.py`.
- The FAISS path still needs a Groq key for the main pipeline because extraction uses DSPy; the retriever itself is separate and can be built from the dataset file.
- `UnitValidator` uses Pint when installed, and falls back to exact unit-string checks.
- `calcmate/units.py` owns all unit handling: `to_si` normalizes inputs, `convert` does the Pint arithmetic, and `parse_requested_unit` reads the output unit ("in km/h") from the problem text.
- The server loads a local `.env` (via `python-dotenv`), so `GROQ_API_KEY` can live there instead of the shell environment.
- `NoopAttemptLogger` keeps logs in memory; `PostgresAttemptLogger` defines the production boundary.
