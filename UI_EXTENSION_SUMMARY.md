# Detektor UI Extension Summary

## Overview
Extended the Detektor UI with three new tabs: **Training**, **Validation**, and **Dataset Check**. These tabs work in both in-process mode (when runtime is provided) and remote backend mode.

## Changes Made

### 1. `serve.py` - New API Endpoints

Added five new endpoints to the FastAPI server:

#### Training Endpoints
- **POST `/v1/train/start`** - Start a training run in a background thread
  - Parameters: `data_yaml`, `config_path`, `epochs`, `batch_size`, `lr`, `model_profile`, `focal_loss_gamma`, `out_dir`, `run_val`
  - Returns: `job_id`, `status`, `out_dir`
  - Stores job state in module-level `TRAINING_JOBS` dict
  - Runs training in a daemon thread using `threading.Thread`

- **GET `/v1/train/status/{job_id}`** - Get training job status and logs
  - Returns: `job_id`, `status`, `log_tail` (last 20 lines), `metrics`, `out_dir`, `error`

- **POST `/v1/train/stop/{job_id}`** - Request cancellation of a running training job
  - Returns: `job_id`, `status`, `message`

#### Validation Endpoint
- **POST `/v1/validate/run`** - Run validation synchronously
  - Parameters: `weights`, `data_yaml`, `conf_thresh`, `iou_thresh`, `output_dir`
  - Falls back to active checkpoint if weights not provided
  - Returns: `status`, `metrics` (JSON-serialized)

#### Dataset Check Endpoint
- **POST `/v1/dataset/check`** - Run dataset validation
  - Parameters: `data_yaml`, `output_dir`
  - Returns: `status`, `summary`, `issues` (list of validation issues)

### 2. `ui/app.py` - Extended UI

#### New Runtime Methods
Added to `DetektorUIRuntime` class:
- `start_training()` - Calls `/v1/train/start` on localhost
- `get_training_status()` - Calls `/v1/train/status/{job_id}`
- `stop_training()` - Calls `/v1/train/stop/{job_id}`
- `run_validation()` - Calls `/v1/validate/run`
- `check_dataset()` - Calls `/v1/dataset/check`

#### New Helper Functions
- `_post_json()` - POST JSON payload and return parsed response
- `_get_json()` - GET URL and return parsed JSON response
- `_start_training_remote()` - Start training via remote backend
- `_refresh_training_status_remote()` - Refresh training status
- `_stop_training_remote()` - Stop training via remote backend
- `_plot_live_loss()` - Build live loss plot from metrics
- `_run_validation_remote()` - Run validation via remote backend
- `_extract_per_class_rows()` - Extract per-class metrics table
- `_build_gate_html()` - Build promotion gate HTML (OPEN/BLOCKED)
- `_run_dataset_check_remote()` - Run dataset check via remote backend

#### New UI Tabs

**Training Tab** (`_build_training_tab_inprocess()`)
- Inputs: data_yaml, config_yaml (optional), epochs (1-100), batch_size (1-32), lr, model_profile (dropdown), focal_loss_gamma (0-3), out_dir, run_val checkbox
- Buttons: Start Training, Stop Training, Refresh Status
- Outputs: status textbox, log textbox (last 30 lines), metrics JSON, live loss plot
- Uses `gr.State()` to track current job_id

**Validation Tab** (`_build_validation_tab_inprocess()`)
- Inputs: weights path (optional, uses active checkpoint), data_yaml, conf_thresh slider, iou_thresh slider, output_dir
- Button: Run Validation
- Outputs: metrics JSON, per-class table (class, precision, recall, f1, ap50), status text, promotion gate HTML
- Gate criteria: mAP50 ≥ 0.5 and recall ≥ 0.5
- Gate HTML: green "✅ GATE: OPEN" or red "🚫 GATE: BLOCKED"

**Dataset Check Tab** (`_build_dataset_check_tab_inprocess()`)
- Inputs: data_yaml path, output_dir
- Button: Run Check
- Outputs: summary JSON, issues table (severity, category, message, file), status text

#### Restructured Interface
- Moved existing inference UI into a "Inference" tab
- Added three new tabs: Training, Validation, Dataset Check
- All tabs use `gr.Tabs()` for organization
- Remote mode (`_build_remote_interface()`) also includes all four tabs

### 3. CSS Enhancements
Added new CSS classes:
- `.det-gate-open` - Green text for open promotion gate
- `.det-gate-blocked` - Red text for blocked promotion gate

### 4. Constants
- `_MODEL_PROFILES` - List of model profiles: firefly, comet, nova, pulsar, quasar, supernova

## Implementation Details

### Training Job Management
- Jobs stored in `TRAINING_JOBS` dict with thread-safe access via `TRAINING_JOBS_LOCK`
- Each job has: `job_id`, `status`, `log_buffer` (deque, maxlen=500), `metrics`, `out_dir`, `started_at`, `stop_requested`, `thread`
- Training runs in background thread, doesn't block API
- Log buffer captures training output for UI display

### Error Handling
- All endpoints have try/except blocks with proper error responses
- Validation endpoint converts torch.Tensor and numpy arrays to JSON-serializable types
- Dataset check endpoint handles missing splits gracefully

### Lazy Imports
- Training, validation, and dataset check modules imported lazily inside endpoint functions to avoid circular imports

### Promotion Gate Logic
- Gate opens when: mAP50 ≥ 0.5 AND (recall ≥ 0.5 OR recall is None)
- Displays metrics values alongside gate status
- Color-coded: green for OPEN, red for BLOCKED

## Testing Recommendations

1. **In-process mode**: Start serve.py with `--ui` flag and test all tabs
2. **Remote mode**: Start serve.py without `--ui`, then run `ui/app.py` standalone
3. **Training**: Test start, status refresh, and stop operations
4. **Validation**: Test with and without weights parameter (should use active checkpoint)
5. **Dataset Check**: Test with valid and invalid data.yaml files

## Backward Compatibility

- All existing functionality preserved
- Existing inference UI moved into "Inference" tab
- No breaking changes to existing API endpoints
- New endpoints are additive only

## Production Quality Features

- Thread-safe job storage with RLock
- Proper error handling and logging
- JSON serialization for torch/numpy types
- Timeout handling for long-running operations
- Status polling via Refresh button (not auto-polling to avoid load)
- Daemon threads for background training
- Lazy imports to avoid circular dependencies
