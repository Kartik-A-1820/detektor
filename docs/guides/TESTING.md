# Running Tests

```bash
python -m pytest                       # whole suite (≈1–2 min on CPU, no GPU or network needed)
python -m pytest tests/test_api.py -x  # one file, stop at first failure
make test                              # same, via the Makefile
ruff check .                           # lint (also enforced in CI)
```

The commands below use the standard-library `unittest` runner, which also works; `pytest` is recommended.

Detektor includes comprehensive unit, integration, and regression tests.

## Fast Tests (Unit Tests Only)

Run lightweight unit tests for quick validation:

```bash
python -m unittest discover -s tests -p "test_box_ops.py"
python -m unittest discover -s tests -p "test_ciou.py"
python -m unittest discover -s tests -p "test_mask_ops.py"
python -m unittest discover -s tests -p "test_schemas.py"
python -m unittest discover -s tests -p "test_config.py"
```

Or run all unit tests:

```bash
python -m unittest tests.test_box_ops tests.test_ciou tests.test_mask_ops tests.test_schemas tests.test_config
```

## Full Test Suite

Run all tests including integration and regression tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Test Categories

**Unit Tests:**
- `test_box_ops.py` - Box decoding and flattening operations
- `test_ciou.py` - CIoU loss and IoU helpers
- `test_mask_ops.py` - Mask composition, cropping, and resizing
- `test_schemas.py` - API schema serialization and validation
- `test_config.py` - Configuration parsing and dataset YAML handling

**Integration Tests:**
- `test_integration.py` - End-to-end workflows (inference, reporting, validation)
- `test_api.py` - FastAPI endpoint testing

**Regression Tests:**
- `test_regression.py` - Schema stability and no-NaN guarantees

**Serving, UI and benchmarks:**
- `test_api_security.py` - API-key auth, CORS, payload limit (413), request IDs, Prometheus output
- `test_upload_validation.py` - content-type leniency, decompression-bomb guard
- `test_ui.py` - console rendering helpers, runtime callbacks, interface construction
- `test_benchmarks.py` - timing/statistics, suite smoke tests, report + regression compare, runner CLI

**Smoke Tests:**
- `test_model.py` - Model architecture smoke tests
- `test_predict.py` - Prediction format validation
- `test_export.py` - ONNX export smoke tests

## Running Specific Test Classes

```bash
python -m unittest tests.test_box_ops.TestBoxOps
python -m unittest tests.test_ciou.TestCIoU.test_ciou_loss_identical_boxes
```

## Test Coverage Notes

- **Unit tests** are fast (<1s each) and cover core helper functions
- **Integration tests** may take longer and test full workflows
- **Regression tests** ensure API stability and no-NaN guarantees
- All tests are designed to run locally without GPU requirements
