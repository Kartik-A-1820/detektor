# Exporting Models

Export ONNX:

```bash
python export.py --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt --output exports/chimera_odis.onnx
```

Compatibility alias:

```bash
python -m scripts.export_onnx --config configs/chimera_s_512.yaml --weights runs/chimera/chimera_best.pt
```
