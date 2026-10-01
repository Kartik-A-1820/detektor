"""Tests for the benchmark framework (fast; uses the smallest profile and tiny workloads)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.common import BenchContext
from benchmarks.report import compare, flatten_metrics, render_markdown
from benchmarks.runner import main as bench_main
from benchmarks.suites import FAST, SUITES, resolve
from benchmarks.synthetic import CLASS_NAMES, generate_dataset
from benchmarks.timing import PeakMemory, summarize, time_callable


class TimingTests(unittest.TestCase):
    def test_summarize_percentiles(self) -> None:
        stats = summarize([float(v) for v in range(1, 101)])
        self.assertEqual(stats["n"], 100)
        self.assertAlmostEqual(stats["p50_ms"], 50.5, places=1)
        self.assertGreater(stats["p99_ms"], stats["p95_ms"])
        self.assertGreater(stats["fps"], 0)
        self.assertEqual(summarize([]), {"n": 0})

    def test_time_callable_counts_runs(self) -> None:
        calls = {"n": 0}

        def fn() -> None:
            calls["n"] += 1

        out = time_callable(fn, warmup=2, runs=5)
        self.assertEqual(len(out), 5)
        self.assertEqual(calls["n"], 7)

    def test_peak_memory_reports_growth(self) -> None:
        with PeakMemory("cpu") as mem:
            blob = bytearray(64 * 1024 * 1024)
            blob[::4096] = b"x" * len(blob[::4096])
            import time

            time.sleep(0.05)
        self.assertIsNotNone(mem.peak_mb)
        self.assertGreater(mem.delta_mb or 0.0, 20.0)
        del blob


class SuiteRegistryTests(unittest.TestCase):
    def test_resolve_aliases_and_dedup(self) -> None:
        self.assertEqual(resolve(["all"]), list(SUITES))
        self.assertEqual(resolve(["fast"]), FAST)
        self.assertEqual(resolve(["latency", "latency", "memory"]), ["latency", "memory"])
        with self.assertRaises(KeyError):
            resolve(["nope"])


class SyntheticDatasetTests(unittest.TestCase):
    def test_generation_is_deterministic_and_valid(self) -> None:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ya = generate_dataset(a, train=6, val=3, img_size=96, seed=7)
            yb = generate_dataset(b, train=6, val=3, img_size=96, seed=7)
            self.assertEqual(len(list((Path(a) / "train" / "images").glob("*.png"))), 6)
            self.assertEqual(len(list((Path(a) / "val" / "labels").glob("*.txt"))), 3)
            la = (Path(a) / "train" / "labels" / "train_0000.txt").read_text()
            lb = (Path(b) / "train" / "labels" / "train_0000.txt").read_text()
            self.assertEqual(la, lb)
            self.assertTrue(la.strip())  # at least one instance
            for line in la.strip().splitlines():
                parts = line.split()
                self.assertIn(int(parts[0]), range(len(CLASS_NAMES)))
                self.assertTrue(all(0.0 <= float(v) <= 1.0 for v in parts[1:]))
            self.assertTrue(ya.exists() and yb.exists())


    def test_relative_output_dir_yields_absolute_yaml_paths(self) -> None:
        # Regression: a relative --output-dir produced a YAML whose paths were resolved twice by train.py.
        import os

        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                yaml_path = generate_dataset("rel/data", train=2, val=1, img_size=64)
            finally:
                os.chdir(cwd)
            import yaml

            cfg = yaml.safe_load(yaml_path.read_text())
            self.assertTrue(Path(cfg["train"]).is_absolute() and Path(cfg["train"]).exists())
            self.assertTrue(Path(cfg["val"]).is_absolute() and Path(cfg["val"]).exists())


class SuitesSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = BenchContext(
            profiles=["firefly"], img_sizes=[128], img_size=128, batch_sizes=[1, 2], warmup=1, runs=2,
            output_dir=Path(self.tmp.name), quick=True,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_complexity(self) -> None:
        from benchmarks.suites import complexity

        row = complexity.run(self.ctx)["rows"][0]
        self.assertGreater(row["params_m"], 0.5)
        self.assertGreater(row["gflops"], 0.0)
        self.assertAlmostEqual(row["gflops"], row["gmacs"] * 2, places=2)

    def test_latency_and_throughput(self) -> None:
        from benchmarks.suites import latency, throughput

        lat = latency.run(self.ctx)["rows"][0]
        self.assertEqual(lat["forward"]["n"], 2)
        self.assertIn("postprocess_p50_ms", lat["predict_dense"])
        thr = throughput.run(self.ctx)["rows"]
        self.assertEqual([r["batch"] for r in thr], [1, 2])
        self.assertTrue(all(r["images_per_s"] > 0 for r in thr))

    def test_startup_loads_non_default_proto_k_checkpoint(self) -> None:
        # Regression: firefly (proto_k=16) checkpoints must load without passing proto_k.
        from benchmarks.suites import startup

        row = startup.run(self.ctx)["rows"][0]
        self.assertGreater(row["checkpoint_mb"], 1.0)
        self.assertGreater(row["load_ms"]["mean_ms"], 0.0)

    def test_memory_measure_profile_and_subprocess_suite(self) -> None:
        from benchmarks.suites import memory

        row = memory.measure_profile("firefly", 2, 96, "cpu", 2)
        self.assertGreater(row["inference_peak_mb"], row["process_baseline_mb"])
        self.assertGreater(row["model_weights_mb"], 4.0)
        self.assertEqual(row["train_batch"], 2)
        self.assertGreater(row["train_peak_mb"], 0)
        # the real suite measures each profile in a fresh child process
        out = memory.run(self.ctx)
        self.assertNotIn("error", out["rows"][0], out["rows"][0])
        self.assertEqual(out["rows"][0]["profile"], "firefly")

    def test_unavailable_suites_skip_cleanly(self) -> None:
        from benchmarks.suites import accuracy, robustness

        self.assertIn("skipped", accuracy.run(self.ctx))
        self.assertIn("skipped", robustness.run(self.ctx))


class ReportAndCompareTests(unittest.TestCase):
    @staticmethod
    def _results(p50: float, ips: float, map50: float) -> dict:
        return {
            "environment": {"device": "cpu"},
            "suites": {
                "latency": {"rows": [{"profile": "firefly", "img_size": 320, "forward": {"p50_ms": p50},
                                      "predict_default": {"p50_ms": p50 + 2, "p95_ms": p50 + 5}}]},
                "throughput": {"rows": [{"profile": "firefly", "img_size": 320, "batch": 4, "images_per_s": ips}]},
                "e2e": {"metrics": {"map50": map50}},
            },
        }

    def test_flatten_directions(self) -> None:
        flat = flatten_metrics(self._results(10.0, 100.0, 0.5))
        self.assertEqual(flat["latency/firefly@320/forward_p50_ms"], (10.0, "lower"))
        self.assertEqual(flat["throughput/firefly@320/b4/images_per_s"], (100.0, "higher"))
        self.assertEqual(flat["e2e/map50"], (0.5, "higher"))

    def test_compare_flags_regressions_and_improvements(self) -> None:
        base = self._results(10.0, 100.0, 0.50)
        slower = self._results(20.0, 60.0, 0.30)  # everything got worse
        md, regs = compare(base, slower, threshold_pct=15)
        self.assertGreaterEqual(len(regs), 3)
        self.assertIn("regression", md)
        faster = self._results(5.0, 200.0, 0.80)
        _, regs = compare(base, faster, threshold_pct=15)
        self.assertEqual(regs, [])
        _, regs = compare(base, self._results(10.5, 98.0, 0.49), threshold_pct=15)
        self.assertEqual(regs, [])  # within noise

    def test_markdown_survives_failed_and_skipped_suites(self) -> None:
        results = {
            "environment": {"device": "cpu", "torch": "x"},
            "suites": {
                "latency": {"status": "error", "error": "boom"},
                "robustness": {"status": "skipped", "skipped": "needs weights", "description": "d", "rows": []},
            },
        }
        md = render_markdown(results)
        self.assertIn("Suite failed", md)
        self.assertIn("Skipped", md)


class RunnerCliTests(unittest.TestCase):
    def test_list_and_quick_run_end_to_end(self) -> None:
        self.assertEqual(bench_main(["list"]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            code = bench_main([
                "run", "--suites", "complexity,latency", "--profiles", "firefly", "--img-sizes", "128",
                "--quick", "--device", "cpu", "--output-dir", tmp, "--tag", "t",
            ])
            self.assertEqual(code, 0)
            run_dir = next(Path(tmp).iterdir())
            data = json.loads((run_dir / "results.json").read_text())
            self.assertEqual(data["suites"]["complexity"]["status"], "ok")
            self.assertEqual(data["schema_version"], 1)
            self.assertIn("Inference latency", (run_dir / "report.md").read_text(encoding="utf-8"))
            self.assertEqual(bench_main(["compare", str(run_dir / "results.json"), str(run_dir / "results.json")]), 0)


if __name__ == "__main__":
    unittest.main()
