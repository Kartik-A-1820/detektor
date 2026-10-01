# Examples

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
