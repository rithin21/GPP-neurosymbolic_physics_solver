# CalcMate

CalcMate is a small neuro-symbolic physics tutor prototype. The LLM-facing layers are represented as narrow interfaces, while all physics reasoning is deterministic and inspectable.

This draft includes one chapter: **Motion in a Straight Line / Kinematics**.

## What is included

- Layer 1 extraction boundary: DSPy + Groq extracts quantities, units, unknown hint, domain hint, and matched trigger phrases.
- Nine-phase pipeline skeleton: extraction, retrieval, domain resolution, constraint resolution, SymPy solving, path reconstruction, unit validation, narration, and logging.
- Layer 2 reasoning engine: loads the chapter graph, applies graph/CBR/meta constraints, passes chapter equations through SymPy, reconstructs the path, and validates units.
- Layer 3 narration boundary: DSPy narrator rewrites verified steps only, with a deterministic fallback for local tests.
- FAISS/Chroma-style retrieval interface with an in-memory placeholder case store.
- Optional FAISS-backed retrieval over a solved-case JSONL dataset in `data/cases/`.
- PostgreSQL logging contract and schema for attempts, law nodes, constraints, under-constrained cases, contradictions, and problem bank rows.
- One base knowledge graph in `data/kinematics_graph.json`.
- Two institution overlays:
  - `data/overlays/ncert_kinematics.json`
  - `data/overlays/jee_kinematics.json`
- Minimal FastAPI backend and plain frontend.
- PostgreSQL schema draft in `schema/postgres.sql`.
- Focused unit tests for graph reasoning.

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
- `NoopAttemptLogger` keeps logs in memory; `PostgresAttemptLogger` defines the production boundary.
