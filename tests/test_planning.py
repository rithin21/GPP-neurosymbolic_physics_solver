import unittest

from calcmate.knowledge_graph import load_default_graph
from calcmate.models import ExtractedProblem, Quantity
from calcmate.pipeline import CalcMatePipeline
from calcmate.planning import SolutionPlan
from calcmate.reasoning import PhysicsReasoner


class FixtureExtractor:
    def extract(self, text: str) -> ExtractedProblem:
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


class FakePlanner:
    def __init__(self, ordered):
        self.ordered = ordered

    def plan(self, problem, graph_context, retrieved_cases):
        return SolutionPlan(ordered_equations=self.ordered, strategy_note="fake plan")


class PlanningTests(unittest.TestCase):
    def test_order_by_plan_reorders_enabled_equations(self):
        reasoner = PhysicsReasoner(load_default_graph())
        enabled = [
            ("eq_s_ut_half_at2", {}, "s"),
            ("eq_v_u_at", {}, "v"),
        ]
        plan = SolutionPlan(ordered_equations=["eq_v_u_at"])
        ordered = reasoner._order_by_plan(enabled, plan)
        self.assertEqual(ordered[0][0], "eq_v_u_at")

    def test_planner_records_phase_and_preserves_correctness(self):
        pipeline = CalcMatePipeline(
            extractor=FixtureExtractor(),
            planner=FakePlanner(["eq_v_u_at", "eq_s_ut_half_at2"]),
        )
        solution = pipeline.solve(
            "A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find the final velocity.",
            "ncert",
        )
        phases = {trace.phase: trace.status for trace in solution.phase_trace}
        self.assertIn("4e_planning", phases)
        self.assertEqual(phases["4e_planning"], "ok")
        self.assertEqual(solution.answer_value, 10)
        self.assertEqual(solution.steps[0].equation_node, "eq_v_u_at")

    def test_null_planner_marks_planning_skipped(self):
        pipeline = CalcMatePipeline(extractor=FixtureExtractor())
        solution = pipeline.solve(
            "A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find the final velocity.",
            "ncert",
        )
        phases = {trace.phase: trace.status for trace in solution.phase_trace}
        self.assertEqual(phases["4e_planning"], "skipped")


if __name__ == "__main__":
    unittest.main()
