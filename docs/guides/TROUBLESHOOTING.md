# Troubleshooting

## Training Issues

**Non-finite loss:**
```bash
python train.py --config configs/chimera_s_512.yaml --data-yaml data.yaml --debug-loss
```
Check which component fails and see `docs/internal/OPTIMIZER_LOSS_BASELINE.md`.

**Out of memory:**
- Reduce `batch_size` in config
- Lower `img_size` (e.g., 416 or 384)
- Disable AMP: `amp: false`
- Increase `grad_accum` for gradient accumulation

**Checkpoint mismatch:**
- Use `--num-classes` to override auto-detection
- Ensure checkpoint matches model architecture

## Inference Issues

**Wrong num_classes:**
```bash
python infer.py --weights checkpoint.pt --source image.jpg --num-classes 4
```

**No detections:**
- Lower `--conf-thresh` (default: 0.25)
- Check if model was trained on similar data
- Verify image format is supported

## API Issues

**Port already in use:**
```bash
python serve.py --weights checkpoint.pt --port 8080
```

**CUDA out of memory:**
```bash
python serve.py --weights checkpoint.pt --device cpu
```

# Performance Tips

## For GTX 1650 Ti 4GB

**Recommended config:**
```yaml
train:
  img_size: 512
  batch_size: 8
  amp: true
  grad_accum: 1
  vram_cap: 0.80
```

**For faster training:**
- Use SGD: `optimizer: "sgd"`, `lr: 0.01`
- Reduce warmup: `warmup_epochs: 1`
- Disable EMA (saves memory)

**For better accuracy:**
- Use AdamW: `optimizer: "adamw"`, `lr: 0.002`
- Enable EMA: `--ema`
- Increase epochs: `epochs: 100`
