from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from calcmate.models import ExtractedProblem, RetrievedCase


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_DIR = PROJECT_ROOT / "data" / "cases"
DEFAULT_CASES_JSONL = DEFAULT_CASE_DIR / "kinematics_cases.jsonl"
DEFAULT_FAISS_INDEX = DEFAULT_CASE_DIR / "kinematics_cases.faiss"
DEFAULT_FAISS_META = DEFAULT_CASE_DIR / "kinematics_cases.meta.json"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class CaseRetriever(Protocol):
    def retrieve(self, problem: ExtractedProblem, top_k: int = 3) -> list[RetrievedCase]:
        raise RuntimeError("INSIDE RETRIEVE")


def case_to_text(case: RetrievedCase | dict) -> str:
    """Create the text that gets embedded for semantic search."""
    if isinstance(case, RetrievedCase):
        data = asdict(case)
    else:
        data = dict(case)
    parts = [
        f"problem: {data.get('problem_text', '')}",
        f"knowns: {sorted(data.get('known_symbols', []))}",
        f"unknown: {data.get('unknown', '')}",
        f"domain: {data.get('domain', '')}",
        f"constraints: {data.get('constraints_fired', [])}",
        f"equations: {data.get('equations_used', [])}",
        f"laws: {data.get('law_nodes', [])}",
    ]
    return " | ".join(parts)


def load_cases_jsonl(path: Path | str) -> list[RetrievedCase]:
    cases: list[RetrievedCase] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        data = json.loads(line)
        cases.append(
            RetrievedCase(
                case_id=data["case_id"],
                problem_text=data["problem_text"],
                known_symbols=set(data["known_symbols"]),
                unknown=data["unknown"],
                domain=data.get("domain", data.get("chapter")),
                constraints_fired=list(data.get("constraints_fired", [])),
                implied_values=dict(data.get("implied_values", {})),
                equations_used=list(data.get("equations_used", [])),
                law_nodes=list(data.get("law_nodes", [])),
                score=float(data.get("score", 0.0)),
            )
        )
    return cases


class InMemoryCaseRetriever:
    """Simple fallback retriever used in tests and when FAISS is unavailable."""

    def __init__(self, cases: list[RetrievedCase] | None = None) -> None:
        self.cases = cases or self._default_cases()

    def retrieve(self, problem: ExtractedProblem, top_k: int = 3) -> list[RetrievedCase]:
        query_known = set(problem.quantities)
        scored: list[RetrievedCase] = []
        for case in self.cases:
            structural_overlap = len(query_known & case.known_symbols)
            unknown_match = 1 if problem.target == case.unknown else 0
            domain_match = 1 if problem.domain_hint == case.domain else 0
            score = structural_overlap + unknown_match + domain_match
            if score <= 0:
                continue
            scored.append(
                RetrievedCase(
                    case_id=case.case_id,
                    problem_text=case.problem_text,
                    known_symbols=case.known_symbols,
                    unknown=case.unknown,
                    domain=case.domain,
                    constraints_fired=case.constraints_fired,
                    implied_values=case.implied_values,
                    equations_used=case.equations_used,
                    law_nodes=case.law_nodes,
                    score=float(score),
                )
            )
        return sorted(scored, key=lambda case: case.score, reverse=True)[:top_k]

    def _default_cases(self) -> list[RetrievedCase]:
        return [
            RetrievedCase(
                case_id="case_max_height_upward_throw",
                problem_text="A ball is thrown upward. Find maximum height.",
                known_symbols={"u"},
                unknown="s",
                domain="kinematics",
                constraints_fired=["constraint_max_height_v_zero", "constraint_free_fall_upward_a_minus_g"],
                implied_values={"v": 0.0, "a": -9.8},
                equations_used=["eq_v2_u2_2as"],
                law_nodes=["law_kinematics_uniform_acceleration"],
                score=0.0,
            )
        ]


class FaissCaseRetriever:
    """Real embedding-backed retriever.

    It stores solved cases in a JSONL file, embeds each case text, and searches
    a FAISS index by nearest neighbors. Structural score is used as a secondary
    rerank after semantic retrieval.
    """

    def __init__(
        self,
        cases_path: Path | str = DEFAULT_CASES_JSONL,
        index_path: Path | str = DEFAULT_FAISS_INDEX,
        meta_path: Path | str = DEFAULT_FAISS_META,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.cases_path = Path(cases_path)
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.embedding_model_name = embedding_model
        self.cases = load_cases_jsonl(self.cases_path)

        print("\n========== RETRIEVER INIT ==========")
        print("Cases path:", self.cases_path)
        print("Index path:", self.index_path)
        print("Meta path:", self.meta_path)
        print("Total cases loaded:", len(self.cases))
        print("First case:", self.cases[0].case_id if self.cases else "None")
        print("Last case:", self.cases[-1].case_id if self.cases else "None")
        print("====================================\n")


        self._index = None
        self._case_ids: list[str] = []
        self._model = None
        self._load_backend()

    def retrieve(self, problem: ExtractedProblem, top_k: int = 3) -> list[RetrievedCase]:
        query_text = self._query_text(problem)
        print("\nQUERY:", query_text)

        vector = self._embed([query_text])
        distances, indices = self._index_search(vector, top_k=max(top_k * 4, top_k))

        print("Indices:", indices)
        print("Distances:", distances)

        candidates: list[RetrievedCase] = []
        for distance, index in zip(distances[0], indices[0]):
            if index < 0 or index >= len(self.cases):
                continue

            case = self.cases[index]

            semantic_score = float(1.0 / (1.0 + distance))
            structural_score = self._structural_score(problem, case)
            total_score = semantic_score * 0.7 + structural_score * 0.3

            candidates.append(
                RetrievedCase(
                    case_id=case.case_id,
                    problem_text=case.problem_text,
                    known_symbols=case.known_symbols,
                    unknown=case.unknown,
                    domain=case.domain,
                    constraints_fired=case.constraints_fired,
                    implied_values=case.implied_values,
                    equations_used=case.equations_used,
                    law_nodes=case.law_nodes,
                    score=total_score,
                )
            )
        return sorted(candidates, key=lambda case: case.score, reverse=True)[:top_k]

    def _load_backend(self) -> None:
        try:
            import faiss  # type: ignore
            import numpy as np  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "FAISS retriever requires `faiss-cpu`, `numpy`, and `sentence-transformers`."
            ) from exc

        self._faiss = faiss
        self._np = np
        self._model = SentenceTransformer(self.embedding_model_name)
        self._load_index_or_build()

    def _load_index_or_build(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            self._index = self._faiss.read_index(str(self.index_path))
            self._case_ids = json.loads(self.meta_path.read_text(encoding="utf-8"))["case_ids"]
            return
        self.build_index()

    def build_index(self) -> None:
        texts = [case_to_text(case) for case in self.cases]
        embeddings = self._embed(texts)
        embeddings = self._normalize(embeddings)
        self._index = self._faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)
        self._case_ids = [case.case_id for case in self.cases]
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(self.index_path))
        self.meta_path.write_text(
            json.dumps({"case_ids": self._case_ids, "embedding_model": self.embedding_model_name}, indent=2),
            encoding="utf-8",
        )

    def _embed(self, texts: list[str]):
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype("float32")

    def _normalize(self, embeddings):
        norms = self._np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = self._np.clip(norms, 1e-12, None)
        return embeddings / norms

    def _index_search(self, query_vector, top_k: int):
        distances, indices = self._index.search(query_vector, top_k)
        return distances, indices

    def _query_text(self, problem: ExtractedProblem) -> str:
        return " | ".join(
            [
                f"problem: {problem.raw_text}",
                f"knowns: {sorted(problem.quantities)}",
                f"unknown: {problem.target}",
                f"domain: {problem.domain_hint}",
                f"triggers: {problem.trigger_phrases}",
            ]
        )

    def _structural_score(self, problem: ExtractedProblem, case: RetrievedCase) -> float:
        query_known = set(problem.quantities)
        structural_overlap = len(query_known & case.known_symbols)
        unknown_match = 1 if problem.target == case.unknown else 0
        domain_match = 1 if problem.domain_hint == case.domain else 0
        return float(structural_overlap + unknown_match + domain_match)

