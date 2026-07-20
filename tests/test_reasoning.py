import unittest

from calcmate.models import ExtractedProblem, Quantity
from calcmate.pipeline import CalcMatePipeline


class FixtureExtractor:
    def extract(self, text: str) -> ExtractedProblem:
        if "maximum height" in text:
            return ExtractedProblem(
                raw_text=text,
                target="s",
                trigger_phrases=["maximum height", "thrown upward"],
                quantities={"u": Quantity("u", 20, "m/s", "initial velocity 20 m/s")},
            )
        if "starts from rest" in text:
            return ExtractedProblem(
                raw_text=text,
                target="v",
                trigger_phrases=["starts from rest"],
                quantities={
                    "u": Quantity("u", 0, "m/s", "starts from rest"),
                    "a": Quantity("a", 2, "m/s^2", "accelerates at 2 m/s^2"),
                    "t": Quantity("t", 5, "s", "for 5 s"),
                },
            )
        return ExtractedProblem(
            raw_text=text,
            target="v",
            trigger_phrases=[],
            quantities={
                "u": Quantity("u", 10, "m/s", "initial velocity 10 m/s"),
                "a": Quantity("a", 2, "m/s^2", "accelerates at 2 m/s^2"),
                "s": Quantity("s", 50, "m", "over 50 m"),
            },
        )


class ReasoningTests(unittest.TestCase):
    def test_max_height_uses_graph_constraints_and_third_equation(self):
        pipeline = CalcMatePipeline(extractor=FixtureExtractor())
        solution = pipeline.solve(
            "A ball is thrown upward with initial velocity 20 m/s. Find the maximum height.",
            "jee",
        )

        self.assertEqual(solution.answer_symbol, "s")
        self.assertEqual(solution.answer_value, 20.4082)
        self.assertIn("constraint_max_height_v_zero", solution.applied_constraints)
        self.assertIn("constraint_free_fall_upward_a_minus_g", solution.applied_constraints)
        self.assertEqual(solution.steps[0].equation_node, "eq_v2_u2_2as")

    def test_ncert_prefers_first_equation_when_possible(self):
        pipeline = CalcMatePipeline(extractor=FixtureExtractor())
        solution = pipeline.solve(
            "A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find the final velocity.",
            "ncert",
        )

        self.assertEqual(solution.answer_symbol, "v")
        self.assertEqual(solution.answer_value, 10)
        self.assertEqual(solution.steps[0].equation_node, "eq_v_u_at")
        self.assertIn("Given:", solution.narration)

    def test_jee_overlay_keeps_concise_narration(self):
        pipeline = CalcMatePipeline(extractor=FixtureExtractor())
        solution = pipeline.solve(
            "A train moving at initial velocity 10 m/s accelerates at 2 m/s^2 over 50 m. Find the final velocity.",
            "jee",
        )

        self.assertEqual(solution.answer_symbol, "v")
        self.assertEqual(solution.steps[0].equation_node, "eq_v2_u2_2as")
        self.assertEqual(solution.narration.strip(), "v = 17.3205 m/s.")

    def test_nine_phase_trace_and_unit_validation_are_returned(self):
        pipeline = CalcMatePipeline(extractor=FixtureExtractor())
        solution = pipeline.solve(
            "A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find the final velocity.",
            "ncert",
        )

        phases = [trace.phase for trace in solution.phase_trace]
        self.assertIn("1_extraction", phases)
        self.assertIn("5_sympy_solving", phases)
        self.assertIn("7_unit_validation", phases)
        self.assertIn("8_narration", phases)
        self.assertIn("9_output_and_logging", phases)
        self.assertIsNotNone(solution.unit_validation)
        self.assertTrue(solution.unit_validation.is_valid)

        # The graph solver reached the target on trigger/meta-rule
        # constraints alone, so retrieval and case fallback should never
        # have run.
        self.assertNotIn("2_hybrid_retrieval", phases)
        self.assertNotIn("5a_graph_solver_failed", phases)
        self.assertNotIn("5b_case_fallback_solving", phases)
        self.assertIsNone(solution.fallback_case_id)

    def test_retrieval_and_case_fallback_only_run_after_graph_solver_fails(self):
        class UnderConstrainedExtractor:
            def extract(self, text: str) -> ExtractedProblem:
                # Only "u" is known and the text carries no trigger phrases,
                # so neither trigger constraints (4a) nor meta-rules (4c)
                # can supply the missing symbols - the graph solver must
                # fail before retrieval is allowed to run at all.
                return ExtractedProblem(
                    raw_text="A ball has initial velocity 20 m/s. Find the maximum height.",
                    target="s",
                    trigger_phrases=[],
                    quantities={"u": Quantity("u", 20, "m/s", "initial velocity 20 m/s")},
                )

        pipeline = CalcMatePipeline(extractor=UnderConstrainedExtractor())
        solution = pipeline.solve(
            "A ball has initial velocity 20 m/s. Find the maximum height.",
            "ncert",
        )

        phases = [trace.phase for trace in solution.phase_trace]
        self.assertIn("5a_graph_solver_failed", phases)
        self.assertIn("2_hybrid_retrieval", phases)
        self.assertIn("5b_case_fallback_solving", phases)

        # Retrieval must appear only after the graph-solver-failed marker.
        failed_index = phases.index("5a_graph_solver_failed")
        retrieval_index = phases.index("2_hybrid_retrieval")
        self.assertLess(failed_index, retrieval_index)


if __name__ == "__main__":
    unittest.main()
