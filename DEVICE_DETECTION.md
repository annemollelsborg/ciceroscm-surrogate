# Cross-Platform Device Detection

This repository supports automatic device detection to run on all hardware platforms:
- **NVIDIA GPUs** (CUDA)
- **Apple Silicon GPUs** (MPS - Metal Performance Shaders)
- **CPU** (fallback)

## Quick Start

The repository uses automatic device detection by default. No changes to how the code is usual run:

```bash
# Training will auto-detect the best available device
python src/train.py

# Speed tests will benchmark all available devices
python src/speed_test.py

# MARL experiments will use the best device
python src/marl_experiment.py
```

## How It Works

### Automatic Device Selection Priority

When device is set to `"auto"` (the default), the system selects devices in this priority order:

1. **CUDA** - NVIDIA GPU (best performance for large models)
2. **MPS** - Apple Silicon GPU (optimal for M-series Macs)
3. **CPU** - Universal fallback (works everywhere)

### Configuration Files

All configuration files use `device: auto` for automatic detection:

- `config/train.yaml` - Training pipeline configuration
- `config/marl.yaml` - Multi-agent RL configuration

A device can still be specified manually if needed:

```yaml
general:
  device: cuda      # Force CUDA
  device: mps       # Force MPS (Apple Silicon)
  device: cpu       # Force CPU
  device: cuda:1    # Use specific GPU
  device: auto      # Auto-detect (recommended)
```

## Using Device Detection in Code

### Python Scripts

```python
from src.utils.device_utils import get_device, print_device_info

# Auto-detect best device
device = get_device("auto", verbose=True)
print_device_info(device)

# Or specify a device with fallback
device = get_device("cuda")  # Falls back to CPU if CUDA unavailable
```

### Jupyter Notebooks

The notebook `notebooks/rnn_based_surrogates.ipynb` has been updated to use automatic device detection:

```python
from src.utils.device_utils import get_device, print_device_info

# Auto-detect and display device info
device = get_device("auto", verbose=True)
print_device_info(device)
```

This will display information like:

```
============================================================
Device Information
============================================================
Device: mps
Name: Apple Silicon GPU (MPS)
Supports Amp: False
Supports Half: False
============================================================
```

## Device Utility Functions

The new `src/utils/device_utils.py` module provides device management:

Core functions:

- **`get_device(device, verbose=True)`** - Get the best available PyTorch device
- **`print_device_info(device)`** - Display device information
- **`configure_device_optimizations(device, verbose=True)`** - Apply device-specific optimizations

Feature detection:

- **`supports_amp(device)`** - Check if Automatic Mixed Precision is supported
- **`supports_half_precision(device)`** - Check if FP16 is beneficial
- **`should_pin_memory(device)`** - Check if pinned memory should be used

Training utilities:

- **`get_autocast_context(device, enabled=True)`** - Get appropriate autocast context
- **`get_grad_scaler(device, enabled=True)`** - Get gradient scaler for AMP
- **`synchronize_device(device)`** - Synchronize device for accurate timing

Device information:

- **`get_device_info(device)`** - Get detailed device specifications (dict)

## Updated Files

The following files have been updated to support automatic device detection:

Core modules:

- `src/utils/device_utils.py` - **New** centralized device detection utility
- `src/train.py` - Training pipeline with auto device detection
- `src/marl_env_step.py` - MARL engines with device detection
- `src/utils/model_utils.py` - Model instantiation with auto device
- `src/utils/data_utils.py` - Device-aware data loaders
- `src/speed_test.py` - Multi-device benchmarking

Configuration files:

- `config/train.yaml` - Changed `device: mps` → `device: auto`
- `config/marl.yaml` - Changed `surrogate_device: cuda` → `surrogate_device: auto`

Notebooks:

- `notebooks/rnn_based_surrogates.ipynb` - Auto device detection in all cells