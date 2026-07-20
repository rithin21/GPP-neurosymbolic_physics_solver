from __future__ import annotations

"""Case-based fallback solving.

When the deterministic graph solver (calcmate.reasoning.PhysicsReasoner)
can't reach the target - the problem is under-constrained, contradicts the
graph's rules, or fails unit validation - this module tries to solve it
anyway by borrowing the method (equation + implied values) from a
structurally similar solved case. Candidates come from retrieval
(calcmate.retrieval, text + structural score) and are expanded through the
case-similarity graph's node2vec neighborhood (calcmate.case_graph). The
graph stays authoritative when it can solve a problem; this only engages
when it can't, and it always reports which case it borrowed from.
"""

from dataclasses import dataclass

import sympy as sp

from calcmate.case_graph import load_all_cases, load_embeddings, nearest_cases
from calcmate.constants import CANONICAL_SYMBOLS, UNIT_BY_SYMBOL
from calcmate.knowledge_graph import PhysicsKnowledgeGraph
from calcmate.models import ExtractedProblem, PhaseTrace, RetrievedCase, SolutionStep

_SYMPY_SYMBOLS = {name: sp.Symbol(name) for name in CANONICAL_SYMBOLS}


@dataclass(frozen=True)
class FallbackResult:
    steps: list[SolutionStep]
    source_case_id: str
    applied_constraints: list[str]


class CaseFallbackSolver:
    def __init__(self, graph: PhysicsKnowledgeGraph, cases_by_id: dict[str, dict] | None = None):
        self.graph = graph
        self.cases_by_id = cases_by_id if cases_by_id is not None else {
            case["case_id"]: case for case in load_all_cases()
        }
        self._embeddings: dict[str, list[float]] | None = None

    def _node2vec_embeddings(self) -> dict[str, list[float]]:
        if self._embeddings is None:
            self._embeddings = load_embeddings()
        return self._embeddings

    def candidate_case_ids(self, retrieved_cases: list[RetrievedCase], expand_top_k: int = 5) -> list[str]:
        """Retrieved cases first (already ranked by the primary retriever),
        then each of the top retrieved cases' node2vec structural
        neighbors - cases that weren't a direct text/structural match but
        sit in the same neighborhood of the equation/constraint graph."""
        ordered: list[str] = []
        seen: set[str] = set()
        for case in retrieved_cases:
            if case.case_id in self.cases_by_id and case.case_id not in seen:
                ordered.append(case.case_id)
                seen.add(case.case_id)

        embeddings = self._node2vec_embeddings()
        for case in retrieved_cases[:2]:
            for neighbor_id, _similarity in nearest_cases(case.case_id, embeddings, top_k=expand_top_k):
                if neighbor_id in self.cases_by_id and neighbor_id not in seen:
                    ordered.append(neighbor_id)
                    seen.add(neighbor_id)
        return ordered

    def solve(
        self,
        problem: ExtractedProblem,
        retrieved_cases: list[RetrievedCase],
        trace: list[PhaseTrace],
    ) -> FallbackResult | None:
        candidate_ids = self.candidate_case_ids(retrieved_cases)
        for case_id in candidate_ids:
            case = self.cases_by_id[case_id]
            if case.get("unknown") != problem.target:
                continue
            result = self._try_case(problem, case)
            if result is not None:
                trace.append(PhaseTrace(
                    phase="5b_case_fallback_solving",
                    status="ok",
                    detail=(
                        f"Graph solver could not reach {problem.target!r}; borrowed the method "
                        f"from similar case {case_id!r} instead."
                    ),
                ))
                return result

        trace.append(PhaseTrace(
            phase="5b_case_fallback_solving",
            status="blocked",
            detail=f"None of {len(candidate_ids)} candidate case(s) could solve target {problem.target!r}.",
        ))
        return None

    def _try_case(self, problem: ExtractedProblem, case: dict) -> FallbackResult | None:
        quantities = {symbol: quantity.value for symbol, quantity in problem.quantities.items()}
        implied_values = case.get("implied_values") or {}
        applied_constraints: list[str] = []
        if implied_values:
            for symbol, value in implied_values.items():
                quantities.setdefault(symbol, float(value))
            applied_constraints = list(case.get("constraints_fired") or [])

        target = problem.target
        for equation_id in case.get("equations_used") or []:
            attrs = self._equation_attrs(equation_id)
            if attrs is None:
                continue
            symbols = set(attrs["symbols"])
            if target not in symbols:
                continue
            known_symbols = symbols - {target}
            if not known_symbols.issubset(quantities):
                continue

            value = self._solve_equation(attrs["expression"], quantities, target)
            if value is None:
                continue

            law_nodes = self.graph.incoming(equation_id, "expressed_as")
            step = SolutionStep(
                equation_node=equation_id,
                law_node=law_nodes[0] if law_nodes else "law_unknown",
                equation=attrs["expression"],
                substitution={symbol: quantities[symbol] for symbol in known_symbols},
                solved_symbol=target,
                value=round(value, 4),
                unit=UNIT_BY_SYMBOL.get(target, ""),
            )
            return FallbackResult(steps=[step], source_case_id=case["case_id"], applied_constraints=applied_constraints)
        return None

    def _equation_attrs(self, equation_id: str) -> dict | None:
        try:
            attrs = self.graph.node(equation_id)
        except KeyError:
            return None
        return attrs if attrs.get("type") == "equation" else None

    def _solve_equation(self, expression: str, quantities: dict[str, float], target: str) -> float | None:
        if target not in _SYMPY_SYMBOLS:
            return None
        left, right = expression.split("=")
        equation = sp.Eq(
            sp.sympify(left.strip(), locals=_SYMPY_SYMBOLS),
            sp.sympify(right.strip(), locals=_SYMPY_SYMBOLS),
        )
        substitutions = {
            _SYMPY_SYMBOLS[symbol]: value
            for symbol, value in quantities.items()
            if symbol in _SYMPY_SYMBOLS and symbol != target
        }
        try:
            solved = sp.solve(equation.subs(substitutions), _SYMPY_SYMBOLS[target])
        except Exception:
            return None

        numeric = [float(root) for root in solved if root.is_number]
        if not numeric:
            return None
        if target == "v" and len(numeric) > 1:
            positive = [root for root in numeric if root >= 0]
            return positive[0] if positive else numeric[0]
        return numeric[0]
