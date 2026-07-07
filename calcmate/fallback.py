from __future__ import annotations

"""LLM reasoning fallback when symbolic solving dead-ends (enhancement #5).

When the deterministic solver cannot reach the target (an under-constrained
problem: some physically-implied known was never stated), the fallback asks the
stronger reasoning model which *existing graph constraints* apply. It returns
constraint node_ids only -- never raw numbers -- so the values it introduces are
exactly the graph's own ``implies`` maps. The caller re-applies those graph
constraints and re-runs the deterministic solve + verifier. Nothing is returned
to the user unless it passes verification (verified-only trust model).
"""

import json
from dataclasses import dataclass, field
from typing import Protocol

from calcmate.graph_context import GraphReasoningContext
from calcmate.llm_config import build_reasoning_lm, use_lm
from calcmate.models import ExtractedProblem, RetrievedCase
from calcmate.rag import format_retrieved_examples


@dataclass(frozen=True)
class FallbackProposal:
    constraint_ids: list[str] = field(default_factory=list)
    rationale: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.constraint_ids


class Fallback(Protocol):
    def attempt(
        self,
        problem: ExtractedProblem,
        graph_context: GraphReasoningContext,
        retrieved_cases: list[RetrievedCase],
        failure_reason: str,
    ) -> FallbackProposal:
        ...


class NullFallback:
    """Default no-op fallback: under-constrained problems stay unresolved."""

    def attempt(
        self,
        problem: ExtractedProblem,
        graph_context: GraphReasoningContext,
        retrieved_cases: list[RetrievedCase],
        failure_reason: str,
    ) -> FallbackProposal:
        return FallbackProposal()


def _build_fallback_signature():
    import dspy

    class ProposeApplicableConstraints(dspy.Signature):
        """Identify which known physics constraints apply to an unsolved problem.

        The deterministic solver could not reach the target because a physically
        implied value was not stated explicitly (e.g. "v=0 at maximum height").
        Using the problem text, the available constraints, and similar solved
        examples, pick the constraint node_ids whose physical situation is
        clearly present in THIS problem. Do NOT invent numbers or equations.

        Return strict JSON:
        {
          "constraint_node_ids": ["constraint_max_height_v_zero"],
          "rationale": "one short sentence"
        }
        Only include node_ids present in the provided context, and only when the
        problem text genuinely satisfies the constraint's situation.
        """

        problem_json: str = dspy.InputField()
        knowledge_graph_context: str = dspy.InputField()
        retrieved_examples: str = dspy.InputField()
        failure_reason: str = dspy.InputField()
        proposal_json: str = dspy.OutputField(desc="Strict JSON only, no markdown.")

    return ProposeApplicableConstraints


class DSPyReasoningFallback:
    def __init__(self) -> None:
        import dspy

        self._lm = build_reasoning_lm()
        self._program = dspy.Predict(_build_fallback_signature())

    def attempt(
        self,
        problem: ExtractedProblem,
        graph_context: GraphReasoningContext,
        retrieved_cases: list[RetrievedCase],
        failure_reason: str,
    ) -> FallbackProposal:
        problem_json = json.dumps(
            {
                "problem_text": problem.raw_text,
                "knowns": sorted(problem.quantities),
                "target": problem.target,
                "trigger_phrases": problem.trigger_phrases,
            }
        )
        try:
            with use_lm(self._lm):
                prediction = self._program(
                    problem_json=problem_json,
                    knowledge_graph_context=graph_context.to_prompt_text(),
                    retrieved_examples=format_retrieved_examples(retrieved_cases),
                    failure_reason=failure_reason,
                )
            data = _parse_json(getattr(prediction, "proposal_json", ""))
        except Exception:  # noqa: BLE001 - fallback is best-effort
            return FallbackProposal()

        valid_ids = set(graph_context.constraint_ids())
        constraint_ids = [
            node_id for node_id in data.get("constraint_node_ids", []) if node_id in valid_ids
        ]
        return FallbackProposal(
            constraint_ids=constraint_ids,
            rationale=str(data.get("rationale", "")),
        )


def _parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
