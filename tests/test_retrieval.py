import json
import tempfile
import unittest
from pathlib import Path

from calcmate.retrieval import load_cases_jsonl


class RetrievalLoaderTests(unittest.TestCase):
    def test_loads_existing_domain_format(self):
        case = self._load_single_case(
            {
                "case_id": "domain_case",
                "problem_text": "A car starts from rest.",
                "known_symbols": ["u", "a", "t"],
                "unknown": "v",
                "domain": "kinematics",
                "constraints_fired": ["constraint_from_rest"],
                "implied_values": {"u": 0},
                "equations_used": ["eq_v_u_at"],
                "law_nodes": ["law_kinematics_uniform_acceleration"],
            }
        )

        self.assertEqual(case.domain, "kinematics")
        self.assertEqual(case.known_symbols, {"u", "a", "t"})

    def test_loads_new_chapter_format_as_domain(self):
        case = self._load_single_case(
            {
                "case_id": "chapter_case",
                "problem_text": "A bus travels for 2 hours.",
                "known_symbols": ["speed", "time"],
                "unknown": "distance",
                "chapter": "kinematics",
                "constraints_fired": ["constraint_constant_speed"],
                "implied_values": {"a": 0},
                "equations_used": ["eq_s_vt"],
                "law_nodes": ["law_uniform_motion"],
            }
        )

        self.assertEqual(case.domain, "kinematics")
        self.assertEqual(case.equations_used, ["eq_s_vt"])

    def _load_single_case(self, data):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.jsonl"
            path.write_text(json.dumps(data) + "\n", encoding="utf-8")
            cases = load_cases_jsonl(path)
        self.assertEqual(len(cases), 1)
        return cases[0]


if __name__ == "__main__":
    unittest.main()
