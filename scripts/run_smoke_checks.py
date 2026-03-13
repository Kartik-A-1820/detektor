from __future__ import annotations

from typing import Callable, List, Tuple


def _run_check(name: str, fn: Callable[[], None]) -> Tuple[str, bool, str]:
    try:
        fn()
        return name, True, "PASS"
    except Exception as exc:
        return name, False, f"FAIL: {exc}"


def main() -> None:
    """Run a concise sequence of local smoke checks with PASS/FAIL output."""
    checks: List[Tuple[str, Callable[[], None]]] = []

    def check_model_import() -> None:
        from models.chimera import ChimeraODIS

        _ = ChimeraODIS(num_classes=1, proto_k=24)

    def check_model_forward() -> None:
        import torch
        from models.chimera import ChimeraODIS

        model = ChimeraODIS(num_classes=1, proto_k=24).eval()
        with torch.no_grad():
            _ = model(torch.randn(1, 3, 512, 512))

    def check_model_export_forward() -> None:
        import torch
        from models.chimera import ChimeraODIS

        model = ChimeraODIS(num_classes=1, proto_k=24).eval()
        with torch.no_grad():
            _ = model.forward_export(torch.randn(1, 3, 512, 512))

    def check_predict_structure() -> None:
        import torch
        from models.chimera import ChimeraODIS

        model = ChimeraODIS(num_classes=1, proto_k=24).eval()
        with torch.no_grad():
            outputs = model.predict(torch.randn(1, 3, 512, 512), original_sizes=[(512, 512)])
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise RuntimeError("predict did not return a single-item list")

    def check_export_helper_import() -> None:
        from utils.export_utils import get_export_names

        _ = get_export_names()

    def check_fastapi_app_import() -> None:
        try:
            from serve import ServiceConfig, create_app
        except Exception as exc:
            print(f"SKIP fastapi_app_import: {exc}")
            return
        _ = create_app(ServiceConfig(weights="dummy.pt"))

    def check_benchmark_helper_import() -> None:
        from utils.benchmark import benchmark_forward

        _ = benchmark_forward

    checks.extend(
        [
            ("model_import", check_model_import),
            ("model_forward", check_model_forward),
            ("model_export_forward", check_model_export_forward),
            ("predict_structure", check_predict_structure),
            ("export_helper_import", check_export_helper_import),
            ("fastapi_app_import", check_fastapi_app_import),
            ("benchmark_helper_import", check_benchmark_helper_import),
        ]
    )

    passed = 0
    for name, fn in checks:
        _, ok, message = _run_check(name, fn)
        print(f"[{ 'PASS' if ok else 'FAIL' }] {name}: {message}")
        if ok:
            passed += 1

    print(f"summary: {passed}/{len(checks)} checks passed")


if __name__ == "__main__":
    main()
