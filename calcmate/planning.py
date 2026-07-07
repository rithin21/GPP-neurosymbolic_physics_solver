from __future__ import annotations

"""Reasoning / planning stage before symbolic solving (enhancement #3).

A planner looks at the extracted problem, the graph reasoning context, and the
retrieved worked examples, and proposes a *strategy*: which graph equations to
try, in what order, to reach the target. Under the verified-only trust model the
planner is purely advisory -- it can only reorder equations that already exist
in the graph. A wrong plan can never produce a wrong answer: the deterministic
SymPy solver and the dimensional verifier remain the sole source of truth, and
if the plan is unhelpful the solver falls back to its full search.
"""

import json
from dataclasses import dataclass, field
from typing import Protocol

from calcmate.graph_context import GraphReasoningContext
from calcmate.llm_config import build_reasoning_lm, use_lm
from calcmate.models import ExtractedProblem, RetrievedCase
from calcmate.rag import format_retrieved_examples


@dataclass(frozen=True)
class SolutionPlan:
    ordered_equations: list[str] = field(default_factory=list)
    strategy_note: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.ordered_equations


class Planner(Protocol):
    def plan(
        self,
        problem: ExtractedProblem,
        graph_context: GraphReasoningContext,
        retrieved_cases: list[RetrievedCase],
    ) -> SolutionPlan:
        ...


class NullPlanner:
    """Default no-op planner: keeps the pipeline fully deterministic."""

    def plan(
        self,
        problem: ExtractedProblem,
        graph_context: GraphReasoningContext,
        retrieved_cases: list[RetrievedCase],
    ) -> SolutionPlan:
        return SolutionPlan()


def _build_planner_signature():
    import dspy

    class PlanSolutionStrategy(dspy.Signature):
        """Choose which known physics equations to apply, and in what order.

        You are given the knowns/target of a 1D kinematics problem, the
        equations and constraints available in a knowledge graph, and similar
        solved examples. Do NOT solve the problem and do NOT invent equations.
        Only select from the provided equation node_ids.

        Return strict JSON:
        {
          "ordered_equation_node_ids": ["eq_v2_u2_2as", "eq_v_u_at"],
          "strategy": "one short sentence explaining the plan"
        }
        Order the node_ids from the most direct route to the target to the
        least. Include only node_ids that appear in the provided context.
        """

        problem_json: str = dspy.InputField()
        knowledge_graph_context: str = dspy.InputField()
        retrieved_examples: str = dspy.InputField()
        plan_json: str = dspy.OutputField(desc="Strict JSON only, no markdown.")

    return PlanSolutionStrategy


class DSPySolutionPlanner:
    """LLM planner using the stronger reasoning model.

    Its output is intersected with the graph's real equation node_ids, so it can
    never introduce an equation that does not exist.
    """

    def __init__(self) -> None:
        import dspy

        self._lm = build_reasoning_lm()
        self._program = dspy.Predict(_build_planner_signature())

    def plan(
        self,
        problem: ExtractedProblem,
        graph_context: GraphReasoningContext,
        retrieved_cases: list[RetrievedCase],
    ) -> SolutionPlan:
        problem_json = json.dumps(
            {
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
                )
            data = _parse_json(getattr(prediction, "plan_json", ""))
        except Exception:  # noqa: BLE001 - planning is advisory; never fatal
            return SolutionPlan()

        valid_ids = set(graph_context.equation_ids())
        ordered = [
            node_id
            for node_id in data.get("ordered_equation_node_ids", [])
            if node_id in valid_ids
        ]
        return SolutionPlan(ordered_equations=ordered, strategy_note=str(data.get("strategy", "")))


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
