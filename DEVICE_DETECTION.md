# Cross-Platform Device Detection

This repository now supports automatic device detection to run seamlessly on all hardware platforms:
- **NVIDIA GPUs** (CUDA)
- **Apple Silicon GPUs** (MPS - Metal Performance Shaders)
- **CPU** (fallback)

## Quick Start

The repository now uses automatic device detection by default. Simply run your code as usual:

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

All configuration files now use `device: auto` for automatic detection:

- `config/train.yaml` - Training pipeline configuration
- `config/marl.yaml` - Multi-agent RL configuration

You can still manually specify a device if needed:

```yaml
general:
  device: cuda      # Force CUDA
  device: mps       # Force MPS (Apple Silicon)
  device: cpu       # Force CPU
  device: cuda:1    # Use specific GPU
  device: auto      # Auto-detect (recommended)
```

### Device-Specific Optimizations

The system automatically enables appropriate optimizations for each device:

#### CUDA (NVIDIA GPUs)
- ✅ Automatic Mixed Precision (AMP) - on Volta+ GPUs (compute capability ≥ 7.0)
- ✅ Half Precision (FP16) inference
- ✅ cuDNN benchmark mode
- ✅ TensorFloat-32 (TF32) matmul on Ampere+ GPUs
- ✅ Pinned memory for faster CPU→GPU transfers

#### MPS (Apple Silicon)
- ✅ High precision matmul
- ❌ AMP not supported yet
- ❌ Half precision disabled (compatibility issues)
- ❌ Pinned memory not beneficial

#### CPU
- ✅ High precision matmul
- ❌ AMP not beneficial for performance
- ❌ Half precision slower than FP32
- ❌ Pinned memory not applicable

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

The new `src/utils/device_utils.py` module provides comprehensive device management:

### Core Functions

- **`get_device(device, verbose=True)`** - Get the best available PyTorch device
- **`print_device_info(device)`** - Display detailed device information
- **`configure_device_optimizations(device, verbose=True)`** - Apply device-specific optimizations

### Feature Detection

- **`supports_amp(device)`** - Check if Automatic Mixed Precision is supported
- **`supports_half_precision(device)`** - Check if FP16 is beneficial
- **`should_pin_memory(device)`** - Check if pinned memory should be used

### Training Utilities

- **`get_autocast_context(device, enabled=True)`** - Get appropriate autocast context
- **`get_grad_scaler(device, enabled=True)`** - Get gradient scaler for AMP
- **`synchronize_device(device)`** - Synchronize device for accurate timing

### Device Information

- **`get_device_info(device)`** - Get detailed device specifications (dict)

## Updated Files

The following files have been updated to support cross-platform execution:

### Core Modules
- ✅ `src/utils/device_utils.py` - **New** centralized device detection utility
- ✅ `src/train.py` - Training pipeline with auto device detection
- ✅ `src/marl_env_step.py` - MARL engines with device detection
- ✅ `src/utils/model_utils.py` - Model instantiation with auto device
- ✅ `src/utils/data_utils.py` - Device-aware data loaders
- ✅ `src/speed_test.py` - Multi-device benchmarking

### Configuration Files
- ✅ `config/train.yaml` - Changed `device: mps` → `device: auto`
- ✅ `config/marl.yaml` - Changed `surrogate_device: cuda` → `surrogate_device: auto`

### Notebooks
- ✅ `notebooks/rnn_based_surrogates.ipynb` - Auto device detection in all cells

## Examples

### Example 1: Training on Any Device

```bash
# Automatically uses CUDA, MPS, or CPU based on availability
python src/train.py
```

Output on Mac M-series:
```
INFO:__main__:Auto-detected MPS device (Apple Silicon GPU)
============================================================
Device Information
============================================================
Device: mps
Name: Apple Silicon GPU (MPS)
Supports Amp: False
Supports Half: False
============================================================
INFO:__main__:Enabled MPS optimizations: high precision matmul
```

Output on NVIDIA GPU system:
```
INFO:__main__:Auto-detected CUDA device: NVIDIA RTX 4090 (1 GPU(s) available)
============================================================
Device Information
============================================================
Device: cuda
Name: NVIDIA RTX 4090
Compute Capability: (8, 9)
Total Memory Gb: 24.0
Multi Processor Count: 128
Supports Amp: True
Supports Half: True
============================================================
INFO:__main__:Enabled CUDA optimizations: cuDNN benchmark, high precision matmul
```

### Example 2: Speed Testing All Devices

```bash
python src/speed_test.py
```

This will automatically benchmark:
- CPU (always)
- CUDA (if available)
- MPS (if available on Apple Silicon)

### Example 3: Manual Device Selection

```python
from src.utils.device_utils import get_device

# Force CUDA with fallback to CPU
device = get_device("cuda", verbose=True)

# Force CPU even if GPU available
device = get_device("cpu", verbose=True)

# Force specific GPU
device = get_device("cuda:1", verbose=True)
```

### Example 4: Model Instantiation

```python
from src.utils.model_utils import instantiate_model

# Auto-detect device (recommended)
model = instantiate_model(
    model_type="tcn",
    n_gas=40,
    hidden=128,
    num_layers=5,
    device="auto",  # or omit to use default "auto"
)

# Specify device
model = instantiate_model(
    model_type="tcn",
    n_gas=40,
    hidden=128,
    num_layers=5,
    device="mps",
)
```

## Migration Guide

If you have existing code that uses hardcoded devices, here's how to migrate:

### Before (Mac-specific)
```python
device = "mps"
model = model.to(device)
```

### After (Cross-platform)
```python
from src.utils.device_utils import get_device

device = get_device("auto")
model = model.to(device)
```

### Before (CUDA-specific with manual checks)
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
```

### After (Cross-platform with auto-optimization)
```python
from src.utils.device_utils import get_device, supports_amp

device = get_device("auto")
use_amp = supports_amp(device)
```

## Performance Notes

### CUDA (NVIDIA GPUs)
- Best performance for large models and batch sizes
- Supports TensorFloat-32 (TF32) on Ampere+ GPUs for 8x speedup
- AMP provides ~2x speedup on Volta+ GPUs with minimal accuracy loss

### MPS (Apple Silicon)
- Excellent performance on M1/M2/M3 Macs
- Typically 5-15x faster than CPU on Apple Silicon
- No AMP support yet, but still very fast
- Best for development and smaller models

### CPU
- Universal compatibility
- Adequate for small models and inference
- Good for debugging and testing
- Slower than GPU options

## Troubleshooting

### Issue: "CUDA device requested but not available"
**Solution**: The system will automatically fall back to CPU. No action needed. If you want to force a specific device, check that it's available first.

### Issue: MPS not being detected on Mac
**Solution**: Ensure you have:
- macOS 12.3+ (Monterey)
- Apple Silicon Mac (M1/M2/M3)
- PyTorch 1.12+ with MPS support

### Issue: Different results on different devices
**Solution**: This is expected due to:
- Floating-point precision differences
- Different optimization kernels
- Results should be similar within numerical tolerance

## Advanced Usage

### Custom Device Configuration

```python
from src.utils.device_utils import (
    get_device,
    configure_device_optimizations,
    get_autocast_context,
    get_grad_scaler,
)

# Initialize
device = get_device("auto")
configure_device_optimizations(device)

# Training loop with device-aware features
scaler = get_grad_scaler(device, enabled=True)
autocast_ctx = get_autocast_context(device, enabled=True)

for batch in dataloader:
    with autocast_ctx:
        output = model(batch)
        loss = criterion(output, target)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
```

### Benchmarking Specific Devices

```python
from src.utils.device_utils import get_device, synchronize_device
import time

device = get_device("cuda")

# Accurate timing
synchronize_device(device)
start = time.perf_counter()

# ... model inference ...

synchronize_device(device)
elapsed = time.perf_counter() - start
print(f"Inference time: {elapsed:.4f}s")
```

## Benefits of This Implementation

1. **Zero configuration needed** - Works out of the box on any system
2. **Optimal performance** - Automatically uses best available hardware
3. **Development flexibility** - Easy to switch between devices for testing
4. **Production ready** - Handles device availability gracefully
5. **Future proof** - Easy to add support for new accelerators (e.g., AMD ROCm, Intel Arc)

## Future Enhancements

Potential future improvements:

- [ ] Add support for AMD GPUs (ROCm)
- [ ] Add support for Intel Arc GPUs
- [ ] Implement distributed multi-GPU training
- [ ] Add device performance profiling
- [ ] Create device-specific model optimization presets
