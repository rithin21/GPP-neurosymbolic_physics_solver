import unittest

from calcmate.extraction import DSPyPhysicsExtractor


class ExtractionNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.extractor = DSPyPhysicsExtractor.__new__(DSPyPhysicsExtractor)

    def test_find_speed_uses_plain_grade_6_7_target(self):
        problem = self.extractor._to_problem(
            "A bus travels 18 km in half an hour. Find speed.",
            {
                "knowns_raw": {"distance": 18, "time": 0.5},
                "units": {"distance": "km", "time": "h"},
                "source_text": {"distance": "18 km", "time": "half an hour"},
                "unknown_hint": "v",
                "domain_hint": "kinematics",
                "matched_trigger_phrases": [],
            },
        )

        self.assertEqual(problem.target, "speed")
        self.assertIn("travels", problem.trigger_phrases)
        self.assertEqual(set(problem.quantities), {"distance", "time"})

    def test_find_distance_and_find_time_use_plain_targets(self):
        distance_problem = self.extractor._to_problem(
            "A child walks at constant speed for 10 seconds. Find distance.",
            {
                "knowns_raw": {"speed": 2, "time": 10},
                "units": {"speed": "m/s", "time": "s"},
                "unknown_hint": "s",
                "matched_trigger_phrases": ["constant speed"],
            },
        )
        time_problem = self.extractor._to_problem(
            "A runner covers 100 m at uniform motion. Find time.",
            {
                "knowns_raw": {"distance": 100, "speed": 5},
                "units": {"distance": "m", "speed": "m/s"},
                "unknown_hint": "t",
                "matched_trigger_phrases": [],
            },
        )

        self.assertEqual(distance_problem.target, "distance")
        self.assertIn("constant speed", distance_problem.trigger_phrases)
        self.assertIn("walks", distance_problem.trigger_phrases)
        self.assertEqual(time_problem.target, "time")
        self.assertIn("covers", time_problem.trigger_phrases)
        self.assertIn("uniform motion", time_problem.trigger_phrases)

    def test_rest_phrases_are_recognized_without_changing_acceleration_schema(self):
        grade_problem = self.extractor._to_problem(
            "A cart starts from rest, moves, and comes to rest during the journey. Find time.",
            {
                "knowns_raw": {"distance": 20, "speed": 4},
                "units": {"distance": "m", "speed": "m/s"},
                "unknown_hint": "time",
                "matched_trigger_phrases": [],
            },
        )
        acceleration_problem = self.extractor._to_problem(
            "A car starts from rest and accelerates at 2 m/s^2 for 5 s. Find the final velocity.",
            {
                "knowns_raw": {"u": 0, "a": 2, "t": 5},
                "units": {"u": "m/s", "a": "m/s^2", "t": "s"},
                "unknown_hint": "v",
                "matched_trigger_phrases": ["starts from rest"],
            },
        )

        self.assertEqual(grade_problem.target, "time")
        self.assertIn("starts from rest", grade_problem.trigger_phrases)
        self.assertIn("comes to rest", grade_problem.trigger_phrases)
        self.assertIn("journey", grade_problem.trigger_phrases)
        self.assertIn("moves", grade_problem.trigger_phrases)
        self.assertEqual(acceleration_problem.target, "v")
        self.assertEqual(set(acceleration_problem.quantities), {"u", "a", "t"})


if __name__ == "__main__":
    unittest.main()
