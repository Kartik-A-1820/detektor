from __future__ import annotations

import unittest
from unittest import mock

from utils.architecture_compatibility import collect_compatibility_matrix


class ArchitectureCompatibilityTests(unittest.TestCase):
    def test_collect_compatibility_matrix_collects_cpu_and_cuda_results(self) -> None:
        fake_cfg = {
            "data": {"num_classes": 4},
        }
        fake_summary = {
            "model_profile": "comet",
        }
        fake_hardware = {
            "device": "cuda",
            "gpu_name": "GTX",
            "free_vram_gb": 3.0,
            "free_ram_gb": 8.0,
            "cpu_count": 8,
        }

        with (
            mock.patch("utils.architecture_compatibility.resolve_training_config", return_value=(fake_cfg, fake_summary)),
            mock.patch("utils.architecture_compatibility.detect_hardware_profile", return_value=fake_hardware),
            mock.patch(
                "utils.architecture_compatibility.probe_profile_training_capacity",
                side_effect=lambda profile_key, **kwargs: {
                    "profile": profile_key,
                    "device": kwargs["device_name"],
                    "compatible": True,
                    "compatibility": "supported",
                    "max_batch_size": 8 if kwargs["device_name"] == "cuda" else 4,
                    "model_info": {"trainable_params": 123},
                },
            ),
        ):
            matrix = collect_compatibility_matrix(profiles=["firefly", "comet"], img_size=512)

        self.assertEqual(matrix["recommended_profile"], "comet")
        self.assertEqual(matrix["profiles"], ["firefly", "comet"])
        self.assertIn("cpu", matrix["results"])
        self.assertIn("cuda", matrix["results"])
        self.assertEqual(len(matrix["results"]["cpu"]), 2)
        self.assertEqual(len(matrix["results"]["cuda"]), 2)
        self.assertEqual(matrix["results"]["cuda"][0]["max_batch_size"], 8)


if __name__ == "__main__":
    unittest.main()
