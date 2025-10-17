"""
Model utilities for device management, checkpointing, and model analysis
"""

import os
import random
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel


def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU)

    Args:
        prefer_mps: Prefer Apple Silicon MPS over CPU

    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available() and prefer_mps:
        device = torch.device("mps")
        print("✓ Using Apple Silicon MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU (training will be slow)")

    return device


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic operations (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    print(f"✓ Set random seed to {seed}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
    }


def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """
    Print detailed model summary

    Args:
        model: PyTorch model
        model_name: Name to display
    """
    params = count_parameters(model)

    print(f"\n📊 {model_name} Summary:")
    print(f"  Total Parameters: {params['total']:,}")
    print(f"  Trainable Parameters: {params['trainable']:,}")
    print(f"  Non-trainable Parameters: {params['non_trainable']:,}")

    # Estimate model size (FP32)
    size_mb = (params['total'] * 4) / (1024 ** 2)  # 4 bytes per float32
    print(f"  Estimated Size (FP32): {size_mb:.2f} MB")
    print(f"  Estimated Size (INT4): {size_mb / 8:.2f} MB")

    # Print layer info
    print(f"\n  Model Architecture:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"    {name}: {num_params:,} parameters")


def save_model(
    model: PreTrainedModel,
    output_dir: str,
    tokenizer: Optional[Any] = None,
    config: Optional[Dict] = None,
):
    """
    Save model checkpoint

    Args:
        model: Hugging Face model
        output_dir: Output directory
        tokenizer: Optional tokenizer to save
        config: Optional config dict to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save_pretrained(output_dir)
    print(f"✓ Model saved to {output_dir}")

    # Save tokenizer
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
        print(f"✓ Tokenizer saved to {output_dir}")

    # Save config
    if config is not None:
        import json
        config_path = output_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✓ Config saved to {config_path}")


def load_model(
    model_path: str,
    model_class: type,
    device: Optional[torch.device] = None,
) -> PreTrainedModel:
    """
    Load model checkpoint

    Args:
        model_path: Path to model directory
        model_class: Model class (e.g., AutoModelForSequenceClassification)
        device: Device to load model on

    Returns:
        Loaded model
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = model_class.from_pretrained(model_path)

    if device is not None:
        model = model.to(device)

    print(f"✓ Model loaded from {model_path}")
    return model


def calculate_model_flops(model: nn.Module, input_shape: tuple) -> int:
    """
    Estimate FLOPs (Floating Point Operations) for model

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, seq_length)

    Returns:
        Estimated FLOPs
    """
    # Simplified FLOP estimation for Transformer
    # Actual calculation would require detailed profiling

    total_params = sum(p.numel() for p in model.parameters())

    # Rough estimate: 2 * params * batch_size * seq_length
    batch_size, seq_length = input_shape
    estimated_flops = 2 * total_params * batch_size * seq_length

    return estimated_flops


def estimate_inference_time(
    model: nn.Module,
    input_ids: torch.Tensor,
    num_runs: int = 100,
    warmup: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Estimate model inference time

    Args:
        model: PyTorch model
        input_ids: Sample input tensor
        num_runs: Number of inference runs
        warmup: Number of warmup runs
        device: Device to run on

    Returns:
        Dictionary with timing statistics (ms)
    """
    import time

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    input_ids = input_ids.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            start = time.perf_counter()
            _ = model(input_ids)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    return {
        "mean_ms": np.mean(times),
        "median_ms": np.median(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "std_ms": np.std(times),
    }


def print_inference_stats(timing_stats: Dict[str, float]):
    """Print formatted inference timing statistics"""
    print(f"\n⚡ Inference Time Statistics (over {100} runs):")
    print(f"  Mean: {timing_stats['mean_ms']:.2f} ms")
    print(f"  Median: {timing_stats['median_ms']:.2f} ms")
    print(f"  Min: {timing_stats['min_ms']:.2f} ms")
    print(f"  Max: {timing_stats['max_ms']:.2f} ms")
    print(f"  Std Dev: {timing_stats['std_ms']:.2f} ms")


def get_model_memory_usage(model: nn.Module) -> Dict[str, float]:
    """
    Calculate model memory usage

    Args:
        model: PyTorch model

    Returns:
        Dictionary with memory usage in MB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / (1024 ** 2)

    return {
        "params_mb": param_size / (1024 ** 2),
        "buffers_mb": buffer_size / (1024 ** 2),
        "total_mb": size_mb,
    }


def check_model_output_shape(
    model: nn.Module,
    input_ids: torch.Tensor,
    expected_output_shape: tuple,
):
    """
    Verify model output shape

    Args:
        model: PyTorch model
        input_ids: Sample input
        expected_output_shape: Expected output shape

    Raises:
        AssertionError if output shape doesn't match
    """
    model.eval()

    with torch.no_grad():
        output = model(input_ids)

        if hasattr(output, 'logits'):
            output_shape = output.logits.shape
        else:
            output_shape = output.shape

    assert output_shape == expected_output_shape, \
        f"Output shape {output_shape} doesn't match expected {expected_output_shape}"

    print(f"✓ Model output shape verified: {output_shape}")


def freeze_layers(model: nn.Module, num_layers_to_freeze: int):
    """
    Freeze bottom N layers of model

    Args:
        model: PyTorch model
        num_layers_to_freeze: Number of layers to freeze from bottom
    """
    frozen_count = 0

    for name, param in model.named_parameters():
        # Check if this is an encoder layer
        if "encoder.layer" in name or "transformer.layer" in name:
            layer_num = int(name.split(".")[2])
            if layer_num < num_layers_to_freeze:
                param.requires_grad = False
                frozen_count += param.numel()

    print(f"✓ Froze {frozen_count:,} parameters in bottom {num_layers_to_freeze} layers")


def unfreeze_all_layers(model: nn.Module):
    """Unfreeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Unfroze all parameters ({trainable_params:,} trainable)")
