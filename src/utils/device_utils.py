"""
Device detection and management utilities for cross-platform PyTorch deployment.

This module provides automatic device detection that works across:
- NVIDIA GPUs (CUDA)
- Apple Silicon GPUs (MPS)
- CPU fallback

It also handles device-specific optimizations and features.
"""

import torch
import logging
from typing import Optional, Literal

logger = logging.getLogger(__name__)


def get_device(
    device: Optional[str] = None,
    verbose: bool = True
) -> torch.device:
    """
    Get the best available PyTorch device.

    Device selection priority:
    1. If device is specified and available, use it
    2. If device is "auto" or None, auto-detect in order: CUDA > MPS > CPU
    3. Fall back to CPU if requested device unavailable

    Args:
        device: Device string ('cuda', 'mps', 'cpu', 'cuda:0', 'auto', or None)
                If None or 'auto', automatically selects best available device
        verbose: Whether to log device selection information

    Returns:
        torch.device: The selected device

    Examples:
        >>> device = get_device()  # Auto-detect best device
        >>> device = get_device('cuda')  # Use CUDA if available
        >>> device = get_device('cuda:1')  # Use specific GPU
    """
    # Auto-detect if not specified or explicitly set to "auto"
    if device is None or device == "auto":
        if torch.cuda.is_available():
            selected_device = torch.device("cuda")
            if verbose:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_count = torch.cuda.device_count()
                logger.info(f"Auto-detected CUDA device: {gpu_name} ({gpu_count} GPU(s) available)")
        elif torch.backends.mps.is_available():
            selected_device = torch.device("mps")
            if verbose:
                logger.info("Auto-detected MPS device (Apple Silicon GPU)")
        else:
            selected_device = torch.device("cpu")
            if verbose:
                logger.info("No GPU detected, using CPU")
        return selected_device

    # Parse the requested device
    requested_device = torch.device(device)
    device_type = requested_device.type

    # Validate requested device is available
    if device_type == "cuda":
        if not torch.cuda.is_available():
            logger.warning(f"CUDA device '{device}' requested but CUDA not available. Falling back to CPU.")
            return torch.device("cpu")

        # Check if specific GPU index is valid
        if requested_device.index is not None:
            if requested_device.index >= torch.cuda.device_count():
                logger.warning(
                    f"CUDA device index {requested_device.index} requested but only "
                    f"{torch.cuda.device_count()} GPU(s) available. Using cuda:0."
                )
                return torch.device("cuda:0")

        if verbose:
            gpu_name = torch.cuda.get_device_name(requested_device.index or 0)
            logger.info(f"Using requested CUDA device: {gpu_name}")
        return requested_device

    elif device_type == "mps":
        if not torch.backends.mps.is_available():
            logger.warning("MPS device requested but not available. Falling back to CPU.")
            return torch.device("cpu")

        if verbose:
            logger.info("Using requested MPS device (Apple Silicon GPU)")
        return requested_device

    elif device_type == "cpu":
        if verbose:
            logger.info("Using requested CPU device")
        return requested_device

    else:
        logger.warning(f"Unknown device type '{device_type}'. Falling back to CPU.")
        return torch.device("cpu")


def supports_amp(device: torch.device) -> bool:
    """
    Check if Automatic Mixed Precision (AMP) is supported on the device.

    AMP is supported on:
    - CUDA devices with compute capability >= 7.0 (Volta and newer)
    - CPU (but not recommended for performance)

    Args:
        device: PyTorch device to check

    Returns:
        bool: True if AMP is supported and recommended
    """
    if device.type == "cuda":
        # Check compute capability for efficient AMP support
        if torch.cuda.is_available():
            # Compute capability 7.0+ has Tensor Cores for efficient mixed precision
            capability = torch.cuda.get_device_capability(device.index or 0)
            return capability[0] >= 7
        return False

    # MPS doesn't support AMP yet
    # CPU supports AMP but it's not beneficial for performance
    return False


def supports_half_precision(device: torch.device) -> bool:
    """
    Check if half precision (FP16) inference is supported and beneficial.

    Args:
        device: PyTorch device to check

    Returns:
        bool: True if half precision is supported
    """
    # CUDA always supports half precision
    if device.type == "cuda":
        return True

    # MPS supports half precision but may have compatibility issues
    # Better to test or use with caution
    if device.type == "mps":
        return False  # Conservative default

    # CPU supports half but it's slower than float32
    return False


def should_pin_memory(device: torch.device) -> bool:
    """
    Determine if pinned memory should be used for DataLoaders.

    Pinned memory enables faster CPU-to-GPU transfers but uses more RAM.
    Only beneficial when transferring data to CUDA devices.

    Args:
        device: PyTorch device that will receive the data

    Returns:
        bool: True if pinned memory should be used
    """
    return device.type == "cuda"


def get_autocast_context(device: torch.device, enabled: bool = True):
    """
    Get the appropriate autocast context for the device.

    Args:
        device: PyTorch device
        enabled: Whether autocast should be enabled

    Returns:
        Context manager for autocast or nullcontext
    """
    from contextlib import nullcontext

    if not enabled or not supports_amp(device):
        return nullcontext()

    if device.type == "cuda":
        return torch.amp.autocast("cuda")

    return nullcontext()


def get_grad_scaler(device: torch.device, enabled: bool = True) -> Optional[torch.amp.GradScaler]:
    """
    Get the appropriate gradient scaler for mixed precision training.

    Args:
        device: PyTorch device
        enabled: Whether scaling should be enabled

    Returns:
        GradScaler if AMP is supported, else None
    """
    if enabled and supports_amp(device):
        if device.type == "cuda":
            return torch.amp.GradScaler("cuda")

    return None


def configure_device_optimizations(device: torch.device, verbose: bool = True):
    """
    Configure device-specific optimizations.

    Args:
        device: PyTorch device to optimize for
        verbose: Whether to log optimization settings
    """
    if device.type == "cuda":
        # Enable cuDNN benchmark mode for optimal conv algorithm selection
        torch.backends.cudnn.benchmark = True

        # Use TF32 for better performance on Ampere+ GPUs (A100, RTX 30xx, etc.)
        # TF32 provides good balance between speed and accuracy
        torch.set_float32_matmul_precision("high")

        if verbose:
            logger.info("Enabled CUDA optimizations: cuDNN benchmark, high precision matmul")

    elif device.type == "mps":
        # MPS-specific optimizations
        # Currently limited options, but set matmul precision for consistency
        torch.set_float32_matmul_precision("high")

        if verbose:
            logger.info("Enabled MPS optimizations: high precision matmul")

    else:
        # CPU optimizations
        # Set reasonable matmul precision
        torch.set_float32_matmul_precision("high")

        if verbose:
            logger.info("Using CPU with high precision matmul")


def synchronize_device(device: torch.device):
    """
    Synchronize the device to ensure all operations are complete.
    Useful for accurate timing measurements.

    Args:
        device: PyTorch device to synchronize
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        # MPS synchronization
        torch.mps.synchronize()
    # CPU operations are synchronous by default


def get_device_info(device: torch.device) -> dict:
    """
    Get detailed information about the device.

    Args:
        device: PyTorch device to query

    Returns:
        dict: Device information including name, memory, capabilities
    """
    info = {
        "device_type": device.type,
        "device_index": device.index,
    }

    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index or 0
        info.update({
            "name": torch.cuda.get_device_name(idx),
            "compute_capability": torch.cuda.get_device_capability(idx),
            "total_memory_gb": torch.cuda.get_device_properties(idx).total_memory / 1e9,
            "multi_processor_count": torch.cuda.get_device_properties(idx).multi_processor_count,
            "supports_amp": supports_amp(device),
            "supports_half": supports_half_precision(device),
        })

    elif device.type == "mps":
        info.update({
            "name": "Apple Silicon GPU (MPS)",
            "supports_amp": supports_amp(device),
            "supports_half": supports_half_precision(device),
        })

    elif device.type == "cpu":
        info.update({
            "name": "CPU",
            "supports_amp": False,
            "supports_half": False,
        })

    return info


def print_device_info(device: torch.device):
    """
    Print detailed device information in a formatted way.

    Args:
        device: PyTorch device to display info for
    """
    info = get_device_info(device)

    print(f"\n{'='*60}")
    print(f"Device Information")
    print(f"{'='*60}")
    print(f"Device: {device}")

    for key, value in info.items():
        if key not in ["device_type", "device_index"]:
            formatted_key = key.replace("_", " ").title()
            print(f"{formatted_key}: {value}")

    print(f"{'='*60}\n")
