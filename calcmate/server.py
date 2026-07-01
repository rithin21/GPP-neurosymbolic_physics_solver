from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from calcmate.attempt_store import AttemptStore
from calcmate.knowledge_graph import PROJECT_ROOT
from calcmate.pipeline import CalcMatePipeline
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(title="CalcMate")
pipeline: CalcMatePipeline | None = None
attempt_store = AttemptStore()
web_dir = PROJECT_ROOT / "web"
app.mount("/static", StaticFiles(directory=web_dir), name="static")


class SolveRequest(BaseModel):
    text: str #problem text
    overlay_id: str = "ncert" #edtech platform
    correct: bool = True #doubt


@app.get("/")
def index() -> FileResponse:
    return FileResponse(web_dir / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

#wen i click solve button this is the endpoint i reach
@app.post("/api/solve")
def solve(request: SolveRequest) -> dict:
    global pipeline
    try:
        if pipeline is None:
            pipeline = CalcMatePipeline()
        solution = pipeline.solve(request.text, request.overlay_id)#passing the problem and the solvingtype expected
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    attempt = attempt_store.record(
        request.text,
        request.overlay_id,
        solution.law_nodes,
        solution.applied_constraints,
        solution.was_under_constrained,
        solution.was_contradiction,
        request.correct,
    )
    payload = solution.to_jsonable()
    payload["attempt_id"] = attempt.id
    return payload


@app.get("/api/weak-nodes")
def weak_nodes() -> list[dict]:
    return attempt_store.weak_nodes()
