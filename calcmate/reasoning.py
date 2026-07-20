from __future__ import annotations

from dataclasses import dataclass, replace

import sympy as sp

from calcmate.constants import CANONICAL_SYMBOLS, UNIT_BY_SYMBOL
from calcmate.knowledge_graph import PhysicsKnowledgeGraph
from calcmate.models import ExtractedProblem, PhaseTrace, Quantity, RetrievedCase, SolutionStep, UnitValidation
from calcmate.overlay import Overlay
from calcmate.text_patterns import compile_phrase, compile_phrase_set
from calcmate.unit_validation import UnitValidator

# Base accepted phrasings for meta-rule fallbacks, compiled into regex so
# variants ("dropping", "started from rest") are caught without listing
# every inflected form as its own literal string.
_FREE_FALL_PATTERN = compile_phrase_set(["free fall", "thrown upward", "thrown up", "dropped"])
_FROM_REST_PATTERN = compile_phrase_set(["from rest", "dropped"])

# Most equation "applicable_when" edges in the graph are informational (e.g.
# eq_v_u_at -> constraint_from_rest just documents a common scenario; the
# equation itself is a general identity that must stay usable even when no
# such constraint fired). eq_time_of_flight is the one genuine exception: it
# assumes a projectile that returns to its launch height under constant
# gravity, so treating it as always-available (like the other equations of
# motion) lets it fire on any ordinary problem where u and a happen to be
# known, fabricating a bogus "t" and corrupting the answer. Gate it
# explicitly on the constraint that actually justifies its physics.
_CONDITIONAL_EQUATIONS: dict[str, str] = {
    "eq_time_of_flight": "constraint_free_fall_upward_a_minus_g",
}


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


class PhysicsReasoner:
    """Layer 2. Deterministic graph traversal, SymPy solving, and validation."""

    def __init__(self, graph: PhysicsKnowledgeGraph, unit_validator: UnitValidator | None = None):
        self.graph = graph
        self.unit_validator = unit_validator or UnitValidator()

    def solve(
        self,
        problem: ExtractedProblem,
        overlay: Overlay,
        retrieved_cases: list[RetrievedCase] | None = None,
    ) -> ReasoningResult:
        trace: list[PhaseTrace] = []
        try:
            working_set = self.resolve_domain(problem, trace)#all the equations and constraints pertaining to that domain are loaded
            constrained_problem, constraints_fired = self.resolve_constraints(problem, retrieved_cases or [], trace)#all possible implications are made and whether a suitable equation is present or not is also verified,the modified problem with all new implications are returned
            raw_steps = self.solve_with_sympy(constrained_problem, working_set, trace, constraints_fired)#return SolutionStep(equation_node,law_node,equation,substitution,solved_symbol=problem.target,value,unit)
            steps = self.reconstruct_solution_path(raw_steps, overlay, trace)#redo the steps based on the overlay
            unit_validation = self.validate_units(steps[-1], trace)#validates units using dimensionality checking using pint
            if not unit_validation.is_valid:
                raise ReasoningError(unit_validation.message)
        except ReasoningError as exc:
            # Attach whatever phases did complete so a caller falling back to
            # case-based solving (or narration) can show why the graph solver
            # itself couldn't reach the target.
            exc.phase_trace = trace
            raise
        return ReasoningResult(
            problem=constrained_problem,
            constraints_fired=constraints_fired,
            steps=steps,
            law_nodes=[step.law_node for step in steps],
            unit_validation=unit_validation,
            phase_trace=trace,
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
        if not self._has_potential_equation(constrained, fired):#makes sure the known quantities we have are enough for an equation
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
        constraints_fired: list[str] | None = None,
    ) -> list[SolutionStep]:
        MAX_ITER = 20
        knowns: set[str] = set(problem.quantities.keys())
        fired = set(constraints_fired or [])
        solution_steps: list[SolutionStep] = []

        for iteration in range(MAX_ITER):                              # Stop 3: safety limit
            enabled = self._find_enabled_equations(knowns, working_set["equations"], fired)

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
                problem = replace(
                    problem,
                    quantities={
                        **problem.quantities,
                        sym: Quantity(sym, step.value, step.unit, source_text="derived"),
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
    ) -> list[SolutionStep]:
        # The last step always solves the target; earlier steps are intermediates.
        target_sym = steps[-1].solved_symbol
        intermediate = [s for s in steps if s.solved_symbol != target_sym]
        answer_candidates = [s for s in steps if s.solved_symbol == target_sym]

        allowed = [s for s in answer_candidates if s.equation_node not in overlay.blocked_methods]
        if not allowed:
            raise ReasoningError("All candidate equations were blocked by overlay.")

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

    def validate_units(self, step: SolutionStep, trace: list[PhaseTrace]) -> UnitValidation:#validates units using dimensionality checking using pint
        validation = self.unit_validator.validate(step)
        trace.append(
            PhaseTrace(
                phase="7_unit_validation",
                status="ok" if validation.is_valid else "blocked",
                detail=validation.message,
            )
        )
        return validation

    def _apply_trigger_constraints(
        self,
        problem: ExtractedProblem,
        quantities: dict[str, Quantity],
        fired: list[str],
    ) -> None:#get the constraints and imply values for the specific trigger phrases
        raw_text = problem.raw_text.lower()
        for node_id, attrs in self.graph.nodes_by_type("constraint"):
            constraint_triggers = attrs.get("trigger_phrases", [])
            if not self._constraint_trigger_matches(constraint_triggers, raw_text):
                continue
            self._apply_implied_values(node_id, attrs.get("implies", {}), quantities, fired, override=True)

    def _constraint_trigger_matches(self, constraint_triggers: list[str], raw_text: str) -> bool:
        # A constraint fires only if one of its accepted trigger phrases is
        # actually present in the raw problem text (inflection-tolerant
        # regex, not an exact string match against one accepted spelling).
        # Deliberately NOT matched against the LLM's self-reported
        # trigger_phrases alone - the extractor is instructed to only copy
        # phrases present in the text, but that's a prompt instruction, not
        # a guarantee, and a hallucinated phrase here would silently
        # override an already-correctly-extracted value (e.g. forcing
        # a=0 via constraint_constant_speed on a problem that never said
        # "uniform motion"). Requiring the raw text itself doesn't lose any
        # legitimate match: a correctly-copied trigger phrase is by
        # definition present in the text, so the same regex against the
        # text catches it anyway.
        for trigger in constraint_triggers:
            pattern = compile_phrase(trigger)
            if pattern.search(raw_text):
                return True
        return False

    def _apply_case_suggested_constraints(
        self,
        problem: ExtractedProblem,
        retrieved_cases: list[RetrievedCase],
        quantities: dict[str, Quantity],
        fired: list[str],
    ) -> None:
        # Symbol/domain overlap alone is a weak signal - e.g. any problem
        # targeting "s" with "u" among its knowns structurally resembles the
        # canonical max-height case even with no free-fall wording at all.
        # Only actually apply a retrieved case's suggested constraint if the
        # constraint's own trigger phrases are corroborated by this
        # problem's text, the same bar _apply_trigger_constraints uses.
        current_signature = set(problem.quantities)
        raw_text = problem.raw_text.lower()
        for case in retrieved_cases:
            if case.score < 2 or case.unknown != problem.target:#condition to accept implications from a similar case
                continue
            if not current_signature & case.known_symbols:#set intersection btwn the question quantities and the retrieved case symbols...if not even one matches then continue
                continue
            for constraint_id, value_map in self._constraint_maps_for_case(case):#get the relevant constraints and respective implications for the retrieved cases
                constraint_triggers = self.graph.node(constraint_id).get("trigger_phrases", [])
                if not self._constraint_trigger_matches(constraint_triggers, raw_text):
                    continue
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
        if _FREE_FALL_PATTERN.search(text):
            self._apply_implied_values("meta_free_fall_a_minus_g", {"a": -9.8}, quantities, fired, override=True)
        if "u" not in quantities and _FROM_REST_PATTERN.search(text):
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

    def _has_potential_equation(self, problem: ExtractedProblem, constraints_fired: list[str] | None = None) -> bool:#makes sure the known quantities we have are enough for an equation
        known = set(problem.quantities)
        fired = set(constraints_fired or [])
        equations = self.graph.nodes_by_type("equation")
        for _ in range(20):
            new_sym = False
            for node_id, attrs in equations:
                if not self._equation_conditions_met(node_id, fired):
                    continue
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

    def _equation_conditions_met(self, node_id: str, fired_constraints: set[str]) -> bool:
        required = _CONDITIONAL_EQUATIONS.get(node_id)
        return required is None or required in fired_constraints

    def _find_enabled_equations(
        self, knowns: set[str], equations: list[tuple[str, dict]], fired_constraints: set[str] | None = None
    ) -> list[tuple[str, dict, str]]:
        fired = fired_constraints or set()
        enabled = []
        for node_id, attrs in equations:
            if not self._equation_conditions_met(node_id, fired):
                continue
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
        sympy_symbols = {symbol: sp.Symbol(symbol) for symbol in CANONICAL_SYMBOLS}
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
        )

    def _choose_root(self, unknown: str, solved: list[sp.Expr]) -> float:
        numeric = [float(value) for value in solved]
        if unknown == "v" and len(numeric) > 1:
            positive = [value for value in numeric if value >= 0]
            if positive:
                return min(positive)
        return numeric[0]
