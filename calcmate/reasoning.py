from __future__ import annotations

from dataclasses import dataclass, replace

import sympy as sp

from calcmate.constants import UNIT_BY_SYMBOL
from calcmate.fallback import Fallback, NullFallback
from calcmate.graph_context import build_reasoning_context
from calcmate.knowledge_graph import PhysicsKnowledgeGraph
from calcmate.models import ExtractedProblem, PhaseTrace, Quantity, RetrievedCase, SolutionStep, UnitValidation
from calcmate.overlay import Overlay
from calcmate.planning import NullPlanner, Planner, SolutionPlan
from calcmate.unit_validation import UnitValidator
from calcmate.verification import DimensionalVerifier


class ReasoningError(ValueError):
    pass


class UnderConstrainedError(ReasoningError):
    pass


class ContradictionError(ReasoningError):
    pass


@dataclass(frozen=True)
class ReasoningResult:
    problem: ExtractedProblem
    constraints_fired: list[str]
    steps: list[SolutionStep]
    law_nodes: list[str]
    unit_validation: UnitValidation
    phase_trace: list[PhaseTrace]
    was_under_constrained: bool = False
    was_contradiction: bool = False
    was_unresolved: bool = False


class PhysicsReasoner:
    """Layer 2. Deterministic graph traversal, SymPy solving, and validation."""

    def __init__(
        self,
        graph: PhysicsKnowledgeGraph,
        unit_validator: UnitValidator | None = None,
        planner: Planner | None = None,
        fallback: Fallback | None = None,
        verifier: DimensionalVerifier | None = None,
    ):
        self.graph = graph
        self.verifier = verifier or DimensionalVerifier(unit_validator)
        self.planner = planner or NullPlanner()
        self.fallback = fallback or NullFallback()
        self._fallback_enabled = not isinstance(self.fallback, NullFallback)

    def solve(
        self,
        problem: ExtractedProblem,
        overlay: Overlay,
        retrieved_cases: list[RetrievedCase] | None = None,
    ) -> ReasoningResult:
        trace: list[PhaseTrace] = []
        retrieved_cases = list(retrieved_cases or [])
        working_set = self.resolve_domain(problem, trace)#all the equations and constraints pertaining to that domain are loaded
        try:
            return self._solve_core(problem, overlay, working_set, retrieved_cases, trace)
        except UnderConstrainedError as exc:
            recovered = self._recover_with_fallback(
                problem, overlay, working_set, retrieved_cases, trace, exc
            )
            if recovered is not None:
                return recovered
            if self._fallback_enabled:
                return self._unresolved_result(problem, trace, exc)
            raise

    def _solve_core(
        self,
        problem: ExtractedProblem,
        overlay: Overlay,
        working_set: dict[str, list[tuple[str, dict]]],
        retrieved_cases: list[RetrievedCase],
        trace: list[PhaseTrace],
    ) -> ReasoningResult:
        constrained_problem, constraints_fired = self.resolve_constraints(problem, retrieved_cases, trace)#all possible implications are made and whether a suitable equation is present or not is also verified
        plan = self._plan(constrained_problem, retrieved_cases, trace)#LLM (advisory) equation-selection strategy over graph+RAG context
        raw_steps = self.solve_with_sympy(constrained_problem, working_set, trace, plan)#return SolutionStep(...)
        steps = self.reconstruct_solution_path(raw_steps, overlay, trace, plan)#redo the steps based on the overlay/plan
        unit_validation = self.validate_units(steps, constrained_problem, trace)#dimensional balance + unit consistency + final-answer check
        if not unit_validation.is_valid:
            raise ReasoningError(unit_validation.message)
        return ReasoningResult(
            problem=constrained_problem,
            constraints_fired=constraints_fired,
            steps=steps,
            law_nodes=[step.law_node for step in steps],
            unit_validation=unit_validation,
            phase_trace=trace,
        )

    def _plan(
        self,
        problem: ExtractedProblem,
        retrieved_cases: list[RetrievedCase],
        trace: list[PhaseTrace],
    ) -> SolutionPlan:
        graph_context = build_reasoning_context(self.graph, problem)
        plan = self.planner.plan(problem, graph_context, retrieved_cases)
        if plan.is_empty:
            detail = "No LLM plan; using deterministic equation search."
            status = "skipped"
        else:
            detail = f"Planner ordered equations {plan.ordered_equations}. {plan.strategy_note}".strip()
            status = "ok"
        trace.append(PhaseTrace(phase="4e_planning", status=status, detail=detail))
        return plan

    def _recover_with_fallback(
        self,
        problem: ExtractedProblem,
        overlay: Overlay,
        working_set: dict[str, list[tuple[str, dict]]],
        retrieved_cases: list[RetrievedCase],
        trace: list[PhaseTrace],
        exc: UnderConstrainedError,
    ) -> ReasoningResult | None:
        if not self._fallback_enabled:
            return None
        graph_context = build_reasoning_context(self.graph, problem)
        proposal = self.fallback.attempt(problem, graph_context, retrieved_cases, str(exc))
        trace.append(
            PhaseTrace(
                phase="5b_llm_fallback",
                status="ok" if proposal.constraint_ids else "skipped",
                detail=(
                    f"Fallback selected graph constraints {proposal.constraint_ids}. {proposal.rationale}".strip()
                    if proposal.constraint_ids
                    else "Fallback found no applicable graph constraint."
                ),
            )
        )
        if proposal.is_empty:
            return None

        # Apply the graph's own implied values for the selected constraints only.
        quantities = dict(problem.quantities)
        fallback_fired: list[str] = []
        for constraint_id in proposal.constraint_ids:
            implies = self.graph.node(constraint_id).get("implies", {})
            self._apply_implied_values(constraint_id, implies, quantities, fallback_fired, override=True)
        augmented = replace(problem, quantities=quantities)

        try:
            result = self._solve_core(augmented, overlay, working_set, retrieved_cases, trace)
        except (UnderConstrainedError, ReasoningError):
            return None  # still unresolved / unverifiable -> caller reports unresolved

        merged = fallback_fired + [c for c in result.constraints_fired if c not in fallback_fired]
        return replace(result, constraints_fired=merged)

    def _unresolved_result(
        self,
        problem: ExtractedProblem,
        trace: list[PhaseTrace],
        exc: Exception,
    ) -> ReasoningResult:
        trace.append(
            PhaseTrace(
                phase="5c_unresolved",
                status="blocked",
                detail=f"No verified solution: {exc}",
            )
        )
        return ReasoningResult(
            problem=problem,
            constraints_fired=[],
            steps=[],
            law_nodes=[],
            unit_validation=UnitValidation("", "", False, str(exc)),
            phase_trace=trace,
            was_under_constrained=True,
            was_unresolved=True,
        )

    def resolve_domain(self, problem: ExtractedProblem, trace: list[PhaseTrace]) -> dict[str, list[tuple[str, dict]]]:#"Are we solving a kinematics problem?"↓ yes"Load all kinematics equations and constraints."
        meta = self.graph.meta
        if problem.domain_hint != meta["chapter"]:
            raise ReasoningError(f"Domain hint {problem.domain_hint!r} does not match graph chapter {meta['chapter']!r}.")
        equations = self.graph.nodes_by_type("equation")
        constraints = self.graph.nodes_by_type("constraint")
        trace.append(
            PhaseTrace(
                phase="3_domain_resolution",
                status="ok",
                detail=f"Loaded {len(equations)} equations and {len(constraints)} constraints for {meta['chapter']}.",
            )
        )
        return {"equations": equations, "constraints": constraints}

    def resolve_constraints(
        self,
        problem: ExtractedProblem,
        retrieved_cases: list[RetrievedCase],
        trace: list[PhaseTrace],
    ) -> tuple[ExtractedProblem, list[str]]:
        quantities = dict(problem.quantities)
        fired: list[str] = []

        self._apply_trigger_constraints(problem, quantities, fired)#implications according to the trigger phrases are applied,the applied ones are stored in fired
        trace.append(
            PhaseTrace(
                phase="4a_trigger_constraints",
                status="ok",
                detail=f"Fired {len(fired)} graph trigger constraints.",
            )
        )

        before_cbr = len(fired)
        self._apply_case_suggested_constraints(problem, retrieved_cases, quantities, fired)#constraints from retrieved cases and their implications have been applied
        trace.append(
            PhaseTrace(
                phase="4b_cbr_constraint_suggestion",
                status="ok",
                detail=f"Accepted {len(fired) - before_cbr} retrieved-case constraint suggestions.",
            )
        )

        before_meta = len(fired)
        self._apply_meta_rules(problem, quantities, fired)#these are some of the base rules that are being applied 
        trace.append(
            PhaseTrace(
                phase="4c_meta_rule_fallback",
                status="ok",
                detail=f"Applied {len(fired) - before_meta} fallback constraints.",
            )
        )

        constrained = replace(problem, quantities=quantities)#updated q with implied values also
        if not self._has_potential_equation(constrained):#makes sure the known quantities we have are enough for an equation
            trace.append(
                PhaseTrace(
                    phase="4d_completeness_check",
                    status="blocked",
                    detail=f"Knowns {sorted(quantities)} cannot determine {problem.target}.",
                )
            )
            raise UnderConstrainedError(f"Problem is under-constrained for target {problem.target!r}.")

        trace.append(
            PhaseTrace(
                phase="4d_completeness_check",
                status="ok",
                detail=f"Known symbols after constraints: {sorted(quantities)}.",
            )
        )
        return constrained, fired

    def solve_with_sympy(
        self,
        problem: ExtractedProblem,
        working_set: dict[str, list[tuple[str, dict]]],
        trace: list[PhaseTrace],
        plan: SolutionPlan | None = None,
    ) -> list[SolutionStep]:
        MAX_ITER = 20
        knowns: set[str] = set(problem.quantities.keys())
        solution_steps: list[SolutionStep] = []

        for iteration in range(MAX_ITER):                              # Stop 3: safety limit
            enabled = self._find_enabled_equations(knowns, working_set["equations"])
            enabled = self._order_by_plan(enabled, plan)               # advisory equation ordering

            new_discoveries = False
            for node_id, attrs, sym in enabled:
                if sym in knowns:
                    continue
                try:
                    step = self._solve_equation(node_id, attrs, problem, solve_for=sym)
                except ContradictionError:
                    if sym == problem.target:
                        raise                                          # Stop 4: true contradiction
                    continue                                           # intermediate eq inconsistent; skip
                except UnderConstrainedError:
                    continue

                knowns.add(sym)
                derived_value = step.raw_value if step.raw_value is not None else step.value
                problem = replace(
                    problem,
                    quantities={
                        **problem.quantities,
                        sym: Quantity(sym, derived_value, step.unit, source_text="derived"),
                    },
                )
                solution_steps.append(step)
                new_discoveries = True

                if sym == problem.target:                              # Stop 1: target solved
                    trace.append(
                        PhaseTrace(
                            phase="5_sympy_solving",
                            status="ok",
                            detail=f"Solved {problem.target} in {iteration + 1} pass(es) via {len(solution_steps)} step(s).",
                        )
                    )
                    return solution_steps

            if not new_discoveries:                                    # Stop 2 / Stop 5
                break

        if problem.target in knowns:                                   # survived Stop 3
            trace.append(
                PhaseTrace(
                    phase="5_sympy_solving",
                    status="ok",
                    detail=f"Solved {problem.target} via {len(solution_steps)} step(s).",
                )
            )
            return solution_steps

        trace.append(
            PhaseTrace(
                phase="5_sympy_solving",
                status="blocked",
                detail=f"No numeric result for {problem.target} from knowns {sorted(knowns)}.",
            )
        )
        raise UnderConstrainedError(f"No numeric SymPy result for target {problem.target!r}.")

    def reconstruct_solution_path(
        self,
        steps: list[SolutionStep],
        overlay: Overlay,
        trace: list[PhaseTrace],
        plan: SolutionPlan | None = None,
    ) -> list[SolutionStep]:
        # The last step always solves the target; earlier steps are intermediates.
        target_sym = steps[-1].solved_symbol
        intermediate = [s for s in steps if s.solved_symbol != target_sym]
        answer_candidates = [s for s in steps if s.solved_symbol == target_sym]

        allowed = [s for s in answer_candidates if s.equation_node not in overlay.blocked_methods]
        if not allowed:
            raise ReasoningError("All candidate equations were blocked by overlay.")

        # A non-empty LLM plan (verified downstream) takes precedence over the
        # overlay's static priority for choosing among valid answer equations.
        if plan is not None and not plan.is_empty:
            plan_rank = {node_id: idx for idx, node_id in enumerate(plan.ordered_equations)}
            best = sorted(allowed, key=lambda s: plan_rank.get(s.equation_node, 10_000))[0]
        else:
            priority = {node_id: idx for idx, node_id in enumerate(overlay.equation_priority)}
            best = sorted(allowed, key=lambda s: priority.get(s.equation_node, 999))[0]

        if overlay.show_intermediate_steps and overlay.structure == "multi_step":
            chosen = intermediate + [best]
        else:
            chosen = [best]

        trace.append(
            PhaseTrace(
                phase="6_solution_path_reconstruction",
                status="ok",
                detail=f"Overlay chose {', '.join(s.equation_node for s in chosen)}.",
            )
        )
        return chosen

    def validate_units(
        self,
        steps: list[SolutionStep],
        problem: ExtractedProblem,
        trace: list[PhaseTrace],
    ) -> UnitValidation:
        # Dimensional balance of every equation + per-substitution unit
        # consistency + the original final-answer dimensionality check.
        report = self.verifier.verify(steps, problem.quantities)
        trace.append(
            PhaseTrace(
                phase="7_unit_validation",
                status="ok" if report.is_valid else "blocked",
                detail=report.summary(),
            )
        )
        if not report.is_valid:
            # Surface the strengthened verdict (which may fail even when the
            # final unit alone looks fine) to the verified-only gate upstream.
            return replace(report.final_answer, is_valid=False, message=report.summary())
        return report.final_answer

    def _order_by_plan(
        self,
        enabled: list[tuple[str, dict, str]],
        plan: SolutionPlan | None,
    ) -> list[tuple[str, dict, str]]:
        if plan is None or plan.is_empty:
            return enabled
        rank = {node_id: idx for idx, node_id in enumerate(plan.ordered_equations)}
        return sorted(enabled, key=lambda item: rank.get(item[0], 10_000))

    def _apply_trigger_constraints(
        self,
        problem: ExtractedProblem,
        quantities: dict[str, Quantity],
        fired: list[str],
    ) -> None:#get the constraints and imply values for the specific trigger phrases
        trigger_set = set(problem.trigger_phrases)
        for node_id, attrs in self.graph.nodes_by_type("constraint"):
            triggers = attrs.get("trigger_phrases", [])
            if not any(trigger in trigger_set for trigger in triggers):
                continue
            self._apply_implied_values(node_id, attrs.get("implies", {}), quantities, fired, override=True)

    def _apply_case_suggested_constraints(
        self,
        problem: ExtractedProblem,
        retrieved_cases: list[RetrievedCase],
        quantities: dict[str, Quantity],
        fired: list[str],
    ) -> None:
        current_signature = set(problem.quantities)
        for case in retrieved_cases:
            # Gate on the *structural* signal (shared symbols + unknown + domain),
            # which survives regardless of how a hybrid retriever fuses its scores.
            structural = case.score_breakdown.get("structural", case.score)
            if structural < 2 or case.unknown != problem.target:#condition to accept implications from a similar case
                continue
            if not current_signature & case.known_symbols:#set intersection btwn the question quantities and the retrieved case symbols...if not even one matches then continue
                continue
            for constraint_id, value_map in self._constraint_maps_for_case(case):#get the relevant constraints and respective implications for the retrieved cases
                self._apply_implied_values(constraint_id, value_map, quantities, fired)#apply watever has been retrieved in the prvs case

    def _constraint_maps_for_case(self, case: RetrievedCase) -> list[tuple[str, dict[str, float]]]:
        maps: list[tuple[str, dict[str, float]]] = []
        for constraint_id in case.constraints_fired:#iterate through all the constraints in the retrieved case
            if constraint_id in [node_id for node_id, _ in self.graph.nodes_by_type("constraint")]:#check if that constraint still exists in the knowledge graph
                maps.append((constraint_id, self.graph.node(constraint_id).get("implies", {})))#maps stores the constraint and the implication 
        return maps

    def _apply_meta_rules(
        self,
        problem: ExtractedProblem,
        quantities: dict[str, Quantity],
        fired: list[str],
    ) -> None:
        text = problem.raw_text.lower()
        if "free fall" in text or "thrown upward" in text or "dropped" in text:
            self._apply_implied_values("meta_free_fall_a_minus_g", {"a": -9.8}, quantities, fired, override=True)
        if "u" not in quantities and ("from rest" in text or "starts from rest" in text or "dropped" in text):
            self._apply_implied_values("meta_from_rest_u_zero", {"u": 0.0}, quantities, fired)

    def _apply_implied_values(
        self,
        constraint_id: str,
        implied: dict[str, float],
        quantities: dict[str, Quantity],
        fired: list[str],
        override: bool = False,
    ) -> None:
        changed = False
        for symbol, value in implied.items():
            if symbol not in quantities or override:
                quantities[symbol] = Quantity(symbol, float(value), UNIT_BY_SYMBOL.get(symbol, ""), constraint_id)
                changed = True
        if changed and constraint_id not in fired:
            fired.append(constraint_id)

    def _has_potential_equation(self, problem: ExtractedProblem) -> bool:#makes sure the known quantities we have are enough for an equation
        known = set(problem.quantities)
        equations = self.graph.nodes_by_type("equation")
        for _ in range(20):
            new_sym = False
            for _, attrs in equations:
                symbols = set(attrs["symbols"])
                unknowns = symbols - known
                if len(unknowns) == 1:
                    sym = next(iter(unknowns))
                    if sym not in known:
                        known.add(sym)
                        new_sym = True
            if problem.target in known:
                return True
            if not new_sym:
                break
        return False

    def _find_enabled_equations(
        self, knowns: set[str], equations: list[tuple[str, dict]]
    ) -> list[tuple[str, dict, str]]:
        enabled = []
        for node_id, attrs in equations:
            symbols = set(attrs["symbols"])
            unknowns = symbols - knowns
            if len(unknowns) == 1:
                sym = next(iter(unknowns))
                enabled.append((node_id, attrs, sym))
        return enabled

    def _solve_equation(
        self, node_id: str, attrs: dict, problem: ExtractedProblem, solve_for: str | None = None
    ) -> SolutionStep:
        target = solve_for or problem.target
        sympy_symbols = {symbol: sp.Symbol(symbol) for symbol in ("u", "v", "a", "t", "s")}#kinematics-only symbol whitelist
        left_text, right_text = attrs["expression"].split("=")
        equation = sp.Eq(
            sp.sympify(left_text.strip(), locals=sympy_symbols),
            sp.sympify(right_text.strip(), locals=sympy_symbols),
        )
        symbols = set(attrs["symbols"])
        known_symbols = symbols - {target}
        if not known_symbols.issubset(problem.quantities):
            raise UnderConstrainedError(f"{node_id} lacks required knowns.")

        substitutions = {
            sympy_symbols[symbol]: quantity.value
            for symbol, quantity in problem.quantities.items()
            if symbol in known_symbols
        }
        solved = sp.solve(equation.subs(substitutions), sympy_symbols[target])
        if not solved:
            raise ContradictionError(f"Could not solve {node_id}.")

        numeric_roots = [root for root in solved if root.is_number]
        if not numeric_roots:
            raise UnderConstrainedError(f"{node_id} produced symbolic roots.")

        value = self._choose_root(target, numeric_roots)
        law_nodes = self.graph.incoming(node_id, "expressed_as") or self.graph.outgoing(node_id, "derives_from")
        return SolutionStep(
            equation_node=node_id,
            law_node=law_nodes[0] if law_nodes else "law_unknown",
            equation=attrs["expression"],
            substitution={str(symbol): float(value) for symbol, value in substitutions.items()},
            solved_symbol=target,
            value=round(value, 4),
            unit=UNIT_BY_SYMBOL.get(target, ""),
            raw_value=float(value),
        )

    def _choose_root(self, unknown: str, solved: list[sp.Expr]) -> float:
        numeric = [float(value) for value in solved]
        if unknown == "v" and len(numeric) > 1:
            positive = [value for value in numeric if value >= 0]
            if positive:
                return min(positive)
        return numeric[0]
