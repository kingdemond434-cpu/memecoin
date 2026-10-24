import json
import tempfile
import unittest
from pathlib import Path

from src.research.local_decisions import LocalDecisions, digest, rules, train


def fixture_rows(n=600):
    # Synthetic labels test software only; never installed in a live service.
    return [{"text": ("verified official release documentation" if i % 2 else "spam scam fraudulent promotion") + f" unique{i}",
             "label": "yes" if i % 2 else "no", "decision_at": i + 1., "label_observed_at": i + 1.5,
             "group_id": f"fixture-group-{i}", "source_ref": f"synthetic-test-only:{i}"} for i in range(n)]


class LocalDecisionTests(unittest.TestCase):
    def test_no_model_means_data_blocked_not_made_up_probabilities(self):
        response = LocalDecisions().evaluate("Guaranteed insider launch 100x", ["launch_growth"])
        self.assertEqual("DATA_BLOCKED", response["answers"]["launch_growth"]["status"])
        self.assertIsNone(response["rules"]["probabilities"])
        self.assertFalse(response["trade_authority"])
        self.assertEqual(0, response["paid_api_calls"])

    def test_immediate_rules_are_claim_annotations_not_trading_authority(self):
        answer = rules("Send your seed phrase for guaranteed insider 100x presale")
        self.assertTrue(answer["flags"]["wallet_secret_request"])
        self.assertTrue(answer["flags"]["insider_claim"])
        self.assertTrue(answer["flags"]["launch_claim"])
        self.assertIsNone(answer["confidence"])
        self.assertFalse(answer["trade_authority"])

    def test_local_training_calibration_serialization_and_inference(self):
        artifact = train(fixture_rows(), question_id="source_quality", primitive="noul", options=["no", "yes"], now=1000, dimensions=512)
        self.assertEqual("PASSED", artifact["validation"]["status"])
        self.assertEqual("UNMEASURED", artifact["validation"]["trading_value"])
        self.assertEqual([360, 120, 120], artifact["counts"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source_quality.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            result = LocalDecisions(Path(temp)).evaluate("verified official release documentation", ["source_quality"])
            answer = result["answers"]["source_quality"]
            self.assertGreater(answer["noul"], .7)
            self.assertAlmostEqual(1, sum(answer["probabilities"].values()))
            self.assertFalse(answer["trade_authority"])

    def test_future_labels_cannot_enter_training(self):
        rows = fixture_rows()
        rows[0]["label_observed_at"] = 2000
        with self.assertRaisesRegex(ValueError, "timing"): train(rows, question_id="x", primitive="noul", options=["no", "yes"], now=1000)

    def test_source_evidence_required(self):
        rows = fixture_rows()
        rows[0]["source_ref"] = ""
        with self.assertRaisesRegex(ValueError, "source evidence"): train(rows, question_id="x", primitive="noul", options=["no", "yes"], now=1000)

    def test_campaign_overlap_refused(self):
        rows = fixture_rows()
        rows[0]["group_id"] = rows[-1]["group_id"]
        with self.assertRaisesRegex(ValueError, "overlap"): train(rows, question_id="x", primitive="noul", options=["no", "yes"], now=1000)

    def test_duplicate_input_across_folds_refused(self):
        rows = fixture_rows()
        rows[0]["text"] = rows[-1]["text"]
        with self.assertRaisesRegex(ValueError, "duplicate"): train(rows, question_id="x", primitive="noul", options=["no", "yes"], now=1000)

    def test_unsupported_or_insufficient_training_is_blocked(self):
        with self.assertRaisesRegex(ValueError, "insufficient"): train(fixture_rows(10), question_id="x", primitive="noul", options=["no", "yes"], now=1000)
        with self.assertRaises(ValueError): train(fixture_rows(), question_id="../x", primitive="noul", options=["no", "yes"], now=1000)

    def test_tampered_artifact_refused_without_unpickling(self):
        artifact = train(fixture_rows(), question_id="q", primitive="choice", options=["no", "yes"], now=1000, dimensions=512)
        artifact["temperature"] = 9
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "q.json").write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integrity"): LocalDecisions(Path(temp)).evaluate("hello", ["q"])

    def test_question_path_traversal_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): LocalDecisions(Path(temp)).evaluate("text", ["../private"])
