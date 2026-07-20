from __future__ import annotations

import os
from typing import Protocol

from calcmate.case_graph import load_all_cases
from calcmate.constants import UNIT_BY_SYMBOL
from calcmate.extraction import DSPyPhysicsExtractor
from calcmate.fallback_solver import CaseFallbackSolver
from calcmate.knowledge_graph import load_default_graph
from calcmate.models import (
    AttemptLog,
    ExtractedProblem,
    PhaseTrace,
    Quantity,
    RetrievedCase,
    Solution,
    SolutionStep,
    UnitValidation,
)
from calcmate.narration import Narrator
from calcmate.overlay import load_overlay
from calcmate.postgres_logging import AttemptLogger, NoopAttemptLogger
from calcmate.reasoning import PhysicsReasoner, ReasoningError, ReasoningResult
from calcmate.retrieval import CaseRetriever, FaissCaseRetriever, InMemoryCaseRetriever, TfidfCaseRetriever
from calcmate.units import convert as convert_si_value


class Extractor(Protocol):
    def extract(self, text: str) -> ExtractedProblem:
        ...


class CalcMatePipeline:
    """Nine-phase CalcMate workflow.

    The LLM-facing phases are extraction and narration. The graph solver
    always gets the first attempt using only trigger-phrase and meta-rule
    constraints; case retrieval is deferred and only runs if that attempt
    raises a ReasoningError, at which point it feeds the case-fallback
    solver. Graph traversal, SymPy, unit validation, and logging remain
    deterministic system components throughout.
    """

    def __init__(
        self,
        extractor: Extractor | None = None,
        retriever: CaseRetriever | None = None,
        narrator: Narrator | None = None,
        attempt_logger: AttemptLogger | None = None,
    ) -> None:
        graph = load_default_graph()
        # Loaded once and shared: the retriever's seed search and the
        # fallback solver's node2vec expansion must draw from the exact
        # same case pool, or "similar case" would mean different things
        # at each stage.
        cases = load_all_cases()
        self.extractor = extractor or DSPyPhysicsExtractor()
        self.retriever = retriever or self._default_retriever(cases)
        self.reasoner = PhysicsReasoner(graph)
        self.fallback_solver = CaseFallbackSolver(graph, cases_by_id={case["case_id"]: case for case in cases})
        self.narrator = narrator or Narrator()
        self.attempt_logger = attempt_logger or NoopAttemptLogger()

    def _default_retriever(self, cases: list[dict]) -> CaseRetriever:
        backend = os.environ.get("CALCMATE_RETRIEVAL_BACKEND", "tfidf").lower()
        if backend == "faiss":
            try:
                return FaissCaseRetriever()
            except Exception:
                return InMemoryCaseRetriever()
        if backend == "memory":
            return InMemoryCaseRetriever()
        try:
            return TfidfCaseRetriever(cases=cases)
        except Exception:
            return InMemoryCaseRetriever()

    def solve(self, text: str, overlay_id: str = "ncert") -> Solution:
        overlay = load_overlay(overlay_id)#loading the overlay
        phase_trace: list[PhaseTrace] = []#this list is used to track which values you got in which phase?

        extracted = self._phase_1_extract(text, phase_trace)#return ExtractedProblem(raw_text=text,quantities=quantities,target=target,trigger_phrases=triggers,domain_hint=domain_hint,)  this is wat is store inside extracted and phase list contains a few details of this extraction
        extracted = self._phase_1b_si_normalize(extracted, phase_trace)#convert every known's value into SI before anything downstream touches it

        # Retrieval is deferred: the graph solver gets first crack at the
        # problem using only trigger-phrase/meta-rule constraints. Similar
        # cases are only fetched if that fails - see the "5a_..." trace
        # entry below for why.
        retrieved_cases: list[RetrievedCase] = []
        fallback_case_id: str | None = None
        try:
            reasoning = self.reasoner.solve(extracted, overlay, retrieved_cases) ##return ReasoningResult(problem,constraints_fired,steps,law_nodesunit_validation,phase_trace,)
        except ReasoningError as exc:
            phase_trace.extend(getattr(exc, "phase_trace", []))
            phase_trace.append(
                PhaseTrace(
                    phase="5a_graph_solver_failed",
                    status="blocked",
                    detail=(
                        f"Knowledge graph could not solve for {extracted.target!r} ({exc}). "
                        "Retrieving similar cases to fall back on."
                    ),
                )
            )
            retrieved_cases = self._phase_2_retrieve(extracted, phase_trace)
            reasoning, fallback_case_id = self._phase_5b_case_fallback(extracted, retrieved_cases, phase_trace, exc)

        reasoning = self._apply_requested_output_unit(reasoning)
        phase_trace.extend(reasoning.phase_trace)# a copy of the phase trace
        narration = self._phase_8_narrate(reasoning, overlay, phase_trace)#explanation for answer has been fetched from llm or fallback method
        solution = self._build_solution(
            reasoning, overlay.overlay_id, retrieved_cases, narration, phase_trace, fallback_case_id
        )#jst a class to structure the answer
        self._phase_9_log(solution, phase_trace)#the final log
        return solution

    def _phase_5b_case_fallback(
        self,
        extracted: ExtractedProblem,
        retrieved_cases,
        phase_trace: list[PhaseTrace],
        original_error: ReasoningError,
    ) -> tuple[ReasoningResult, str | None]:
        """The graph solver couldn't reach the target on its own. Try to
        borrow a solving method from a structurally similar solved case
        instead of failing outright."""
        fallback_trace: list[PhaseTrace] = []
        result = self.fallback_solver.solve(extracted, retrieved_cases, fallback_trace)
        phase_trace.extend(fallback_trace)
        if result is None:
            raise original_error

        unit_validation = self.reasoner.unit_validator.validate(result.steps[-1])
        reasoning = ReasoningResult(
            problem=extracted,
            constraints_fired=result.applied_constraints,
            steps=result.steps,
            law_nodes=[step.law_node for step in result.steps],
            unit_validation=unit_validation,
            phase_trace=[],
            was_under_constrained=True,
        )
        return reasoning, result.source_case_id

    def _phase_1_extract(self, text: str, trace: list[PhaseTrace]) -> ExtractedProblem:
        extracted = self.extractor.extract(text) #return ExtractedProblem(raw_text=text,quantities=quantities,target=target,trigger_phrases=triggers,domain_hint=domain_hint,)  this is wat is store inside extracted
        trace.append(
            PhaseTrace(
                phase="1_extraction",
                status="ok",
                detail=f"Extracted knowns {sorted(extracted.quantities)} and target {extracted.target}.",
            )
        )
        return extracted

    def _phase_1b_si_normalize(self, extracted: ExtractedProblem, trace: list[PhaseTrace]) -> ExtractedProblem:
        """Every equation in the graph is expressed in SI. Extraction only
        parses whatever unit the question used (km, hours, kmph, ...) - if
        those values were substituted straight into the equations without
        converting them first, the result would be numerically wrong while
        still carrying the *symbol's* canonical SI unit as a label (e.g.
        "300 km / 5 hours" naively substituted as 300/5 and labeled "m/s").
        This phase makes the "solve in SI" step actually true, not just
        true for problems that already happened to use SI units.
        """
        converted: dict[str, Quantity] = {}
        notes: list[str] = []
        for symbol, quantity in extracted.quantities.items():
            si_unit = UNIT_BY_SYMBOL.get(symbol)
            if not si_unit or quantity.unit == si_unit:
                converted[symbol] = quantity
                continue
            si_value = convert_si_value(quantity.value, quantity.unit, si_unit)
            if si_value is None:
                # Unrecognized unit - leave it; unit validation downstream
                # will flag the mismatch rather than silently treating the
                # raw value as if it were already SI.
                converted[symbol] = quantity
                continue
            converted[symbol] = replace(quantity, value=round(si_value, 6), unit=si_unit)
            notes.append(f"{symbol}: {quantity.value:g} {quantity.unit} -> {si_value:g} {si_unit}")

        trace.append(
            PhaseTrace(
                phase="1b_si_input_normalization",
                status="ok",
                detail="Converted to SI: " + "; ".join(notes) if notes else "All knowns were already in SI.",
            )
        )
        return replace(extracted, quantities=converted)

    def _phase_2_retrieve(self, extracted: ExtractedProblem, trace: list[PhaseTrace]):#only called after the graph solver fails (see 5a_graph_solver_failed) - gets similar cases to feed the case-fallback solver
        cases = self.retriever.retrieve(extracted, top_k=3)
        trace.append(
            PhaseTrace(
                phase="2_hybrid_retrieval",
                status="ok",
                detail=f"Retrieved {len(cases)} similar case(s).",
            )
        )
        return cases#contains the top 3-5 matches

    # Regex-detected "in <unit>" phrasing for each unit the answer might be
    # requested in. New spellings map onto the same handful of canonical
    # units without each needing its own accepted-string entry elsewhere.
    _OUTPUT_UNIT_PATTERNS: list[tuple[str, str]] = [
        (r"\bin\s+(?:kilometers?|kilometres?|km)\s*(?:per|/)\s*(?:hour|h)\b", "km/h"),
        (r"\bin\s+kmph\b", "km/h"),
        (r"\bin\s+(?:kilometers?|kilometres?|km)\s*(?:per|/)\s*(?:minute|min)\b", "km/min"),
        (r"\bin\s+(?:centimeters?|centimetres?|cm)\s*(?:per|/)\s*(?:second|s)\b", "cm/s"),
        (r"\bin\s+(?:millimeters?|millimetres?|mm)\b", "mm"),
        (r"\bin\s+(?:centimeters?|centimetres?|cm)\b", "cm"),
        (r"\bin\s+(?:kilometers?|kilometres?|km)\b", "km"),
        (r"\bin\s+(?:meters?|metres?)\s*(?:per|/)\s*(?:second|s)\b", "m/s"),
        (r"\bin\s+(?:meters?|metres?)\b", "m"),
        (r"\bin\s+(?:milliseconds?|ms)\b", "ms"),
        (r"\bin\s+(?:minutes?|min)\b", "min"),
        (r"\bin\s+(?:hours?|hrs?|h)\b", "h"),
    ]

    def _apply_requested_output_unit(self, reasoning: ReasoningResult) -> ReasoningResult:
        """Phase 7b: the target was already solved entirely in SI (phase 5).
        This step only converts that SI answer into whatever unit the
        question asked for - it never re-derives the answer.
        """
        requested_unit = self._requested_output_unit(reasoning.problem.raw_text)
        if requested_unit is None:
            return reasoning

        final_step = reasoning.steps[-1]
        converted_step = self._convert_step_unit(final_step, requested_unit)
        if converted_step is None:
            trace = list(reasoning.phase_trace)
            trace.append(
                PhaseTrace(
                    phase="7b_output_unit_conversion",
                    status="skipped",
                    detail=f"Requested {requested_unit}, but no SI conversion is registered from {final_step.unit}.",
                )
            )
            return replace(reasoning, phase_trace=trace)

        steps = list(reasoning.steps)
        steps[-1] = converted_step
        trace = list(reasoning.phase_trace)
        trace.append(
            PhaseTrace(
                phase="7b_output_unit_conversion",
                status="ok",
                detail=(
                    f"Solved {final_step.value:g} {final_step.unit} (SI) and converted "
                    f"to requested unit: {converted_step.value:g} {requested_unit}."
                ),
            )
        )
        unit_validation = UnitValidation(
            expected_unit=reasoning.unit_validation.expected_unit,
            actual_unit=converted_step.unit,
            is_valid=True,
            message=f"Final answer converted from SI ({final_step.unit}) to requested unit {requested_unit}.",
        )
        return replace(reasoning, steps=steps, unit_validation=unit_validation, phase_trace=trace)

    def _requested_output_unit(self, text: str) -> str | None:
        lowered = text.lower()
        for pattern, unit in self._OUTPUT_UNIT_PATTERNS:
            if re.search(pattern, lowered):
                return unit
        return None

    def _convert_step_unit(self, step: SolutionStep, requested_unit: str) -> SolutionStep | None:
        converted_value = convert_si_value(step.value, step.unit, requested_unit)
        if converted_value is None:
            return None
        return replace(step, value=round(converted_value, 4), unit=requested_unit)

    def _phase_8_narrate(self, reasoning: ReasoningResult, overlay, trace: list[PhaseTrace]) -> str:
        narration = self.narrator.narrate(
            reasoning.problem,
            overlay,
            reasoning.constraints_fired,
            reasoning.steps,
        )
        trace.append(
            PhaseTrace(
                phase="8_narration",
                status="ok",
                detail="Generated narration from verified solution payload.",
            )
        )
        return narration

    def _phase_9_log(self, solution: Solution, trace: list[PhaseTrace]) -> None:
        self.attempt_logger.log_attempt(#attempt log
            AttemptLog(
                problem_text=solution.problem.raw_text,
                overlay_id=solution.overlay_id,
                law_nodes_used=solution.law_nodes,
                constraints_fired=solution.applied_constraints,
                was_under_constrained=solution.was_under_constrained,
                was_contradiction=solution.was_contradiction,
            )
        )
        trace.append(
            PhaseTrace(
                phase="9_output_and_logging",
                status="ok",
                detail="Prepared output and recorded attempt log.",
            )
        )

    def _build_solution(
        self,
        reasoning: ReasoningResult,
        overlay_id: str,
        retrieved_cases,
        narration: str,
        phase_trace: list[PhaseTrace],
        fallback_case_id: str | None = None,
    ) -> Solution:
        answer = reasoning.steps[-1]
        return Solution(
            problem=reasoning.problem,
            overlay_id=overlay_id,
            applied_constraints=reasoning.constraints_fired,
            steps=reasoning.steps,
            answer_symbol=answer.solved_symbol,
            answer_value=answer.value,
            answer_unit=answer.unit,
            law_nodes=reasoning.law_nodes,
            narration=narration,
            retrieved_cases=retrieved_cases,
            unit_validation=reasoning.unit_validation,
            phase_trace=phase_trace,
            was_under_constrained=reasoning.was_under_constrained,
            was_contradiction=reasoning.was_contradiction,
            fallback_case_id=fallback_case_id,
        )
