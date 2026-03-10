from export import export_onnx


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Detektor to ONNX")
    parser.add_argument("--config", type=str, default="configs/chimera_s_512.yaml", help="Path to the base model config YAML")
    parser.add_argument("--weights", type=str, default="runs/chimera/chimera_last.pt", help="Path to model weights or checkpoint")
    parser.add_argument("--output", type=str, default="exports/chimera_odis.onnx", help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--dynamic-batch", action="store_true", help="Enable dynamic batch axis in the exported model")
    parser.add_argument("--check-parity", action="store_true", help="Run optional PyTorch vs ONNX parity check after export")
    args = parser.parse_args()

    export_onnx(
        config_path=args.config,
        weights=args.weights,
        output_path=args.output,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
        check_parity=args.check_parity,
    )
