# API examples

Use these after completing the [root setup](../README.md#quick-start) and training or obtaining a compatible checkpoint. Commands below run from the repository root. Replace `photo.jpg` with an image you can share; output filenames are examples, not bundled assets.

| File | What it shows |
| --- | --- |
| [`api_client.py`](api_client.py) | Calling the REST API from Python (single + batch, API key, drawing boxes) |
| [`sample_val_metrics.json`](sample_val_metrics.json) | Shape of the validation metrics output |

```bash
python serve.py --weights runs/chimera/chimera_best.pt          # terminal 1
python examples/api_client.py photo.jpg --save annotated.jpg    # terminal 2
```

`curl` equivalent:

```bash
curl -X POST "http://localhost:8000/v1/predict?conf_thresh=0.3" -H "X-API-Key: $DETEKTOR_API_KEY" -F "image=@photo.jpg"
```

For authenticated deployments, set the server API key according to [configuration](../docs/reference/CONFIGURATION.md) and use the same key in the client. Please include server/client versions and a minimal non-sensitive request when reporting an issue. Focused API examples and error-handling improvements are welcome through [CONTRIBUTING.md](../CONTRIBUTING.md).
