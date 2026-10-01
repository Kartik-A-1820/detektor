# Exporting Models

Export a checkpoint to ONNX (works for every architecture profile; the model is rebuilt from the checkpoint's own
metadata):

```bash
python export.py --weights runs/chimera/chimera_best.pt --output exports/chimera_odis.onnx --check-parity
```

`--check-parity` runs the exported graph in ONNX Runtime and compares it with PyTorch; the command exits with status 2
if the outputs differ, so it can gate a release pipeline. Options and details: [TOOLS.md](../reference/TOOLS.md#onnx-export).

Compatibility alias:

```bash
python -m scripts.export_onnx --weights runs/chimera/chimera_best.pt
```

The exported graph contains the network only (class, box, objectness, mask-coefficient maps and prototypes); decoding,
NMS and mask assembly are done by `ChimeraODIS.predict`. On CPU, ONNX Runtime is typically 1.6–3× faster than eager
PyTorch for the forward pass — see [BENCHMARKS.md](../BENCHMARKS.md).
