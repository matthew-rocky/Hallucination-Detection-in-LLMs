import unittest
from unittest.mock import patch

from detectors.signal import SignalConfig, _init_hf, get_signal_status, run_signal_detector


FAKE_FEATURE_RECORD = {
    "features": {
        "mean_negative_log_prob": 1.2,
        "token_log_prob_std": 0.3,
        "entropy_mean": 2.1,
        "entropy_std": 0.4,
        "mean_token_probability": 0.45,
        "top2_margin_mean": 0.2,
        "num_answer_tokens": 4.0,
        "selected_layers": [5],
        "hidden_norm_mean": 8.0,
        "hidden_norm_var": 0.2,
        "hidden_drift_mean": 0.05,
        "layer_centroid_dispersion": 0.03,
    },
    "tokens": ["Tokyo"],
    "answer_token_count": 4,
}
READY_STATUS = {
    "python_executable": "C:/Python/python.exe",
    "torch_installed": True,
    "torch_version": "2.5.1",
    "transformers_installed": True,
    "transformers_version": "4.50.2",
    "device": "cpu",
    "backend_available": True,
    "backend_status": "available",
    "backend_status_label": "HF backend active",
    "backend_error": None,
    "backend_model_name": "distilgpt2",
    "local_files_only": False,
}
OFFLINE_STATUS = {
    "python_executable": "C:/Python/python.exe",
    "torch_installed": False,
    "torch_version": None,
    "transformers_installed": False,
    "transformers_version": None,
    "device": "cpu",
    "backend_available": False,
    "backend_status": "unavailable",
    "backend_status_label": "HF backend unavailable",
    "backend_error": "torch import failed: No module named 'torch'",
    "backend_model_name": "distilgpt2",
    "local_files_only": False,
}


class SignalBackendTests(unittest.TestCase):
    def tearDown(self):
        _init_hf.cache_clear()

    @patch("detectors.signal._package_version", return_value=None)
    @patch("detectors.signal._import_torch", side_effect=ImportError("No module named 'torch'"))
    def test_no_torch_status(self, _mock_import_torch, _mock_package_version):
        _init_hf.cache_clear()
        status = get_signal_status(SignalConfig(model_name="distilgpt2"))

        self.assertFalse(status["backend_available"])
        self.assertEqual(status["backend_status"], "unavailable")
        self.assertFalse(status["torch_installed"])
        self.assertFalse(status["transformers_installed"])
        self.assertEqual(status["device"], "cpu")
        self.assertIn("torch import failed", status["backend_error"])

    @patch("detectors.signal._extract_features")
    @patch("detectors.signal.get_signal_status", return_value=OFFLINE_STATUS)
    def test_detector_offline(self, _mock_status, mock_extract):
        result = run_signal_detector(
            question="What is the capital of Japan?",
            answer="The capital of Japan is Tokyo.",
            method_name="Internal-Signal Baseline",
            mode="uncertainty_baseline",
        )

        mock_extract.assert_not_called()
        self.assertTrue(result["metadata"]["fallback_mode"])
        self.assertFalse(result["metadata"]["backend_available"])
        self.assertEqual(result["metadata"]["backend_status"], "unavailable")
        self.assertEqual(result["metadata"]["result_origin"], "deterministic_fallback_approximation")

    @patch("detectors.signal._extract_features", return_value=FAKE_FEATURE_RECORD)
    @patch("detectors.signal.get_signal_status", return_value=READY_STATUS)
    def test_detector_backend(self, _mock_status, mock_extract):
        result = run_signal_detector(
            question="What is the capital of Japan?",
            answer="The capital of Japan is Tokyo.",
            method_name="Internal-Signal Baseline",
            mode="uncertainty_baseline",
        )

        self.assertGreaterEqual(mock_extract.call_count, 1)
        self.assertEqual(result["implementation_status"], "implemented")
        self.assertEqual(result["metadata"]["backend_status"], "available")
        self.assertEqual(result["metadata"]["backend_status_label"], "HF backend active")
        self.assertEqual(result["metadata"]["result_origin"], "full_backend_scoring")
        self.assertFalse(result["metadata"].get("fallback_mode", False))


if __name__ == "__main__":
    unittest.main()
