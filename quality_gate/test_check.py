import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from check import DEFAULT_INPUT, evaluate


class GateTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads(DEFAULT_INPUT.read_text())

    def test_archive_has_no_eligible_candidate(self):
        result = evaluate(self.document)
        self.assertEqual(result["eligible_layers"], [])
        self.assertEqual(len(result["per_layer"]), 4)
        self.assertEqual(result["mechanism"], "unresolved")

    def test_threshold_is_strict(self):
        self.document["measurements"][0]["kl"] = 0.1
        self.document["measurements"][1]["kl"] = 0.0999
        result = evaluate(self.document)
        self.assertEqual(result["eligible_layers"], [20])
        self.assertEqual(result["mechanism"], "unresolved")

    def test_invalid_kl_values_abort_the_report(self):
        for kl in (None, "0.05", True, -0.05, float("nan"), float("inf"), -float("inf")):
            with self.subTest(kl=kl):
                self.document["measurements"][0]["kl"] = kl
                with self.assertRaises(ValueError):
                    evaluate(self.document)

    def test_incomplete_or_unknown_layers_abort_the_report(self):
        measurements = self.document["measurements"]
        for rows in (
            [],
            measurements[:-1],
            measurements + [measurements[0]],
            measurements + [{"layer": 21, "kl": 0.0}],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evaluate({"measurements": rows})

    def test_cli_rejects_malformed_and_duplicate_fields_without_a_report(self):
        with tempfile.TemporaryDirectory() as directory:
            input_file = Path(directory) / "invalid.json"
            for raw in ("{", '{"measurements": [], "measurements": []}'):
                with self.subTest(raw=raw):
                    input_file.write_text(raw)
                    run = subprocess.run(
                        [sys.executable, str(DEFAULT_INPUT.with_name("check.py")), str(input_file)],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(run.returncode, 2)
                    self.assertEqual(run.stdout, "")
                    self.assertIn("KL gate stopped", run.stderr)
