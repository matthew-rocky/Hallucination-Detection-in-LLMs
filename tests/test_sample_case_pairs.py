import unittest

from data.sample_cases import list_cases, get_sample_pair, sample_pairs_for
from ui.method_descriptions import METHOD_ORDER


class SampleCasePairTests(unittest.TestCase):
    def test_method_pair(self):
        for method_name in METHOD_ORDER:
            pair = sample_pairs_for(method_name)
            self.assertIsNotNone(pair["low"], method_name)
            self.assertIsNotNone(pair["high"], method_name)
            self.assertEqual(pair["low"]["risk_level"], "low")
            self.assertEqual(pair["high"]["risk_level"], "high")
            self.assertIn(method_name, pair["low"]["method_targets"])
            self.assertIn(method_name, pair["high"]["method_targets"])

    def test_pair_lookup(self):
        self.assertEqual(
            get_sample_pair("Internal-Signal Baseline", "low")["id"],
            "internal_baseline_low_tokyo",
        )
        self.assertEqual(
            get_sample_pair("Internal-Signal Baseline", "high")["id"],
            "internal_baseline_high_marwick",
        )
        self.assertEqual(
            get_sample_pair("CoVe-Style Verification", "high")["id"],
            "cove_high_midtown",
        )

    def test_demo_ids_unique(self):
        cases = list_cases()
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        pair_ids = {case["pair_id"] for case in cases}
        self.assertEqual(len(pair_ids), len(METHOD_ORDER))


if __name__ == "__main__":
    unittest.main()
