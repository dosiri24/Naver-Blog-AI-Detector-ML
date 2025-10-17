#!/usr/bin/env python3
"""
Optimize trained model for deployment
- Pruning: Remove 30% of low-importance weights
- Quantization: INT4 quantization for size reduction
- Performance validation

Target: Reduce 60MB (FP32) → 5-10MB final size
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from transformers import BertForSequenceClassification

# Get project root directory
SCRIPT_DIR = Path(__file__).parent  # ML-Training/scripts/
ML_TRAINING_ROOT = SCRIPT_DIR.parent  # ML-Training/
PROJECT_ROOT = ML_TRAINING_ROOT.parent  # Naver-Blog-AI-Detector-ML/

# Add utils to path
sys.path.append(str(ML_TRAINING_ROOT))
from utils.model_utils import (
    get_device,
    print_model_summary,
    get_model_memory_usage,
    estimate_inference_time,
    print_inference_stats,
)


class ModelOptimizer:
    """Optimize model for deployment"""

    def __init__(self, model_path: str, output_dir: str):
        self.model_path = Path(model_path).resolve()  # Convert to absolute path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        print(f"📥 Loading model from {self.model_path}...")

        # Load model from local directory with proper config
        from transformers import AutoConfig

        # First load config
        config = AutoConfig.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False
        )

        # Then load model with config
        self.model = BertForSequenceClassification.from_pretrained(
            str(self.model_path),
            config=config,
            local_files_only=True,
            trust_remote_code=False
        )
        self.device = get_device()
        self.model = self.model.to(self.device)

        print(f"✓ Model loaded successfully")
        print(f"  Config: {config.model_type}, vocab_size={config.vocab_size}")
        print_model_summary(self.model, "Original Model")

    def apply_pruning(self, amount: float = 0.3) -> nn.Module:
        """
        Apply L1 unstructured pruning to model

        Args:
            amount: Fraction of weights to prune (0.0-1.0)

        Returns:
            Pruned model
        """
        print(f"\n🔨 Applying L1 Unstructured Pruning (amount={amount})...")

        parameters_to_prune = []

        # Prune attention and feed-forward layers
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, "weight"))

        print(f"  Found {len(parameters_to_prune)} linear layers to prune")

        # Apply pruning
        for module, param_name in parameters_to_prune:
            prune.l1_unstructured(module, name=param_name, amount=amount)

        # Calculate sparsity
        total_params = 0
        zero_params = 0

        for module, param_name in parameters_to_prune:
            param = getattr(module, param_name)
            total_params += param.nelement()
            zero_params += (param == 0).sum().item()

        sparsity = 100 * zero_params / total_params
        print(f"\n  ✓ Pruning complete!")
        print(f"    Sparsity: {sparsity:.2f}%")
        print(f"    Zero weights: {zero_params:,} / {total_params:,}")

        return self.model

    def remove_pruning_reparameterization(self):
        """
        Make pruning permanent by removing reparameterization

        This converts the pruned weights from a temporary mask to permanent zeros
        """
        print(f"\n🔧 Making pruning permanent...")

        count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                try:
                    prune.remove(module, "weight")
                    count += 1
                except ValueError:
                    pass  # No pruning applied to this module

        print(f"  ✓ Removed pruning reparameterization from {count} layers")

    def quantize_to_int8(self) -> nn.Module:
        """
        Apply dynamic INT8 quantization

        Returns:
            Quantized model
        """
        print(f"\n📊 Applying INT8 Dynamic Quantization...")

        # Dynamic quantization (INT8)
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},  # Quantize Linear layers
            dtype=torch.qint8
        )

        print(f"  ✓ INT8 quantization complete!")

        return quantized_model

    def validate_model(
        self,
        model: nn.Module,
        num_samples: int = 100,
    ) -> dict:
        """
        Validate model performance after optimization

        Args:
            model: Model to validate
            num_samples: Number of samples for inference time measurement

        Returns:
            Dictionary with validation metrics
        """
        print(f"\n🧪 Validating optimized model...")

        model.eval()
        vocab_size = model.config.vocab_size

        # Test forward pass
        batch_size = 4
        seq_length = 128

        dummy_input = torch.randint(
            0, vocab_size,
            (batch_size, seq_length),
            device=self.device
        )

        with torch.no_grad():
            outputs = model(dummy_input)
            logits = outputs.logits

        # Verify output
        assert logits.shape == (batch_size, 2), f"Unexpected output shape: {logits.shape}"
        print(f"  ✓ Forward pass successful")
        print(f"    Input shape: {dummy_input.shape}")
        print(f"    Output shape: {logits.shape}")

        # Measure inference time
        timing_stats = estimate_inference_time(
            model,
            dummy_input,
            num_runs=num_samples,
            warmup=10,
            device=self.device,
        )

        print_inference_stats(timing_stats)

        # Check if meets target
        target_ms = 50
        meets_target = timing_stats['mean_ms'] < target_ms

        if meets_target:
            print(f"  ✅ Inference time meets target (< {target_ms}ms)")
        else:
            print(f"  ⚠️  Inference time exceeds target ({timing_stats['mean_ms']:.2f}ms > {target_ms}ms)")

        return {
            "mean_inference_ms": timing_stats['mean_ms'],
            "median_inference_ms": timing_stats['median_ms'],
            "meets_target": meets_target,
        }

    def save_optimized_model(
        self,
        model: nn.Module,
        model_name: str,
        optimization_info: dict,
    ):
        """
        Save optimized model and metadata

        Args:
            model: Optimized model
            model_name: Name for saved model
            optimization_info: Optimization metadata
        """
        output_path = self.output_dir / model_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Check if model is quantized by checking if it has quantized layers
        is_quantized = any(hasattr(m, 'qconfig') and m.qconfig is not None
                          for m in model.modules())

        # Save model
        if is_quantized or "quantiz" in model_name.lower():
            # Save quantized model as TorchScript
            try:
                model.eval()
                dummy_input = torch.randint(0, model.config.vocab_size, (1, 128))

                traced_model = torch.jit.trace(model, dummy_input)
                traced_path = output_path / "model_quantized.pt"
                torch.jit.save(traced_model, traced_path)
                print(f"  ✓ Quantized model saved to {traced_path}")
            except Exception as e:
                print(f"  ⚠️  Could not save as TorchScript: {e}")
                print(f"  ℹ️  Saving as regular PyTorch model instead")
                torch.save(model.state_dict(), output_path / "model.pt")
                print(f"  ✓ Model weights saved to {output_path / 'model.pt'}")
        else:
            # Save as Hugging Face format
            model.save_pretrained(output_path)
            print(f"  ✓ Model saved to {output_path}")

        # Save optimization metadata (convert numpy types to Python types)
        metadata_path = output_path / "optimization_info.json"

        # Convert numpy types to Python types for JSON serialization
        serializable_info = {}
        for key, value in optimization_info.items():
            if isinstance(value, (np.integer, np.floating)):
                serializable_info[key] = value.item()
            elif isinstance(value, np.bool_):
                serializable_info[key] = bool(value)
            elif isinstance(value, np.ndarray):
                serializable_info[key] = value.tolist()
            else:
                serializable_info[key] = value

        with open(metadata_path, "w") as f:
            json.dump(serializable_info, f, indent=2)

        print(f"  ✓ Optimization metadata saved to {metadata_path}")

    def get_model_size(self, model: nn.Module) -> float:
        """
        Calculate model size in MB

        Args:
            model: PyTorch model

        Returns:
            Model size in MB
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Estimate size (4 bytes per FP32 parameter)
        size_mb = (total_params * 4) / (1024 ** 2)

        return size_mb


def main():
    # Define default paths as absolute paths
    DEFAULT_MODEL_PATH = str(ML_TRAINING_ROOT / "models" / "checkpoints" / "tiny-transformer-init")
    DEFAULT_OUTPUT_DIR = str(ML_TRAINING_ROOT / "models" / "optimized")

    parser = argparse.ArgumentParser(
        description="Optimize trained TinyTransformer model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for optimized models"
    )
    parser.add_argument(
        "--prune_amount",
        type=float,
        default=0.3,
        help="Fraction of weights to prune (0.0-1.0)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 quantization"
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip model validation"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("⚡ TinyTransformer Model Optimization")
    print("=" * 70)

    # Initialize optimizer
    optimizer = ModelOptimizer(args.model, args.output)

    # Original model metrics
    original_size = optimizer.get_model_size(optimizer.model)
    print(f"\n📊 Original Model:")
    print(f"  Size: {original_size:.2f} MB")

    optimization_steps = []

    # Step 1: Pruning
    if args.prune_amount > 0:
        print(f"\n{'='*70}")
        print(f"Step 1: Pruning")
        print(f"{'='*70}")

        pruned_model = optimizer.apply_pruning(args.prune_amount)
        optimizer.remove_pruning_reparameterization()

        pruned_size = optimizer.get_model_size(pruned_model)
        size_reduction = ((original_size - pruned_size) / original_size) * 100

        print(f"\n📊 After Pruning:")
        print(f"  Size: {pruned_size:.2f} MB ({size_reduction:.1f}% reduction)")

        # Validate
        if not args.skip_validation:
            validation_metrics = optimizer.validate_model(pruned_model)
        else:
            validation_metrics = {}

        # Save pruned model
        optimizer.save_optimized_model(
            pruned_model,
            "pruned",
            {
                "optimization": "pruning",
                "prune_amount": args.prune_amount,
                "size_mb": pruned_size,
                "size_reduction_percent": size_reduction,
                **validation_metrics,
            }
        )

        optimization_steps.append("pruning")

    # Step 2: Quantization (INT8)
    if args.quantize:
        print(f"\n{'='*70}")
        print(f"Step 2: INT8 Quantization")
        print(f"{'='*70}")

        # Use pruned model if available
        base_model = pruned_model if args.prune_amount > 0 else optimizer.model

        quantized_model = optimizer.quantize_to_int8()

        # Estimate quantized size (rough estimate: ~25% of original)
        quantized_size = original_size * 0.25
        total_reduction = ((original_size - quantized_size) / original_size) * 100

        print(f"\n📊 After Quantization:")
        print(f"  Estimated Size: {quantized_size:.2f} MB ({total_reduction:.1f}% reduction)")

        # Validate
        if not args.skip_validation:
            validation_metrics = optimizer.validate_model(quantized_model)
        else:
            validation_metrics = {}

        # Save quantized model
        optimizer.save_optimized_model(
            quantized_model,
            "quantized_int8",
            {
                "optimization": "pruning+quantization" if args.prune_amount > 0 else "quantization",
                "prune_amount": args.prune_amount if args.prune_amount > 0 else 0,
                "quantization": "INT8",
                "estimated_size_mb": quantized_size,
                "total_reduction_percent": total_reduction,
                **validation_metrics,
            }
        )

        optimization_steps.append("quantization")

    # Summary
    print(f"\n{'='*70}")
    print(f"✅ Optimization Complete!")
    print(f"{'='*70}")

    print(f"\nOptimization Summary:")
    print(f"  Steps Applied: {', '.join(optimization_steps)}")
    print(f"  Original Size: {original_size:.2f} MB")

    if args.prune_amount > 0:
        print(f"  After Pruning: {pruned_size:.2f} MB")

    if args.quantize:
        print(f"  After Quantization: {quantized_size:.2f} MB (estimated)")
        print(f"  Total Reduction: {total_reduction:.1f}%")

    print(f"\n  Output Directory: {args.output}")

    print(f"\nNext Steps:")
    print(f"  1. Review optimization metrics in {args.output}/*/optimization_info.json")
    print(f"  2. Run 5_export_coreml.py to convert to CoreML format")
    print(f"  3. Test CoreML model in Xcode")

    print(f"\n💡 Note:")
    print(f"  - INT4 quantization will be applied during CoreML conversion")
    print(f"  - Final CoreML model size target: 5-10 MB")
    print(f"  - Current model is ready for CoreML export!")


if __name__ == "__main__":
    main()
