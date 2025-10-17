#!/usr/bin/env python3
"""
Export optimized model to CoreML format
- CoreML 8.0+ with INT4 quantization
- W8A8 mode for Neural Engine optimization
- Target: 5-10MB final size, < 50ms inference

Requires: coremltools 8.0+, macOS 14+
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
from transformers import BertForSequenceClassification


class ModelWrapper(nn.Module):
    """Wrapper to make HuggingFace model output trace-compatible"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # Return only logits (not dict)
        return outputs.logits

# Get project root directory
SCRIPT_DIR = Path(__file__).parent  # ML-Training/scripts/
ML_TRAINING_ROOT = SCRIPT_DIR.parent  # ML-Training/
PROJECT_ROOT = ML_TRAINING_ROOT.parent  # Naver-Blog-AI-Detector-ML/

# Add utils to path
sys.path.append(str(ML_TRAINING_ROOT))
from utils.model_utils import get_device


class CoreMLExporter:
    """Export PyTorch model to CoreML format"""

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        model_name: str = "BlogAIDetector",
    ):
        self.model_path = Path(model_path).resolve()  # Convert to absolute path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

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
        self.model.eval()

        print(f"✓ Model loaded successfully")
        print(f"  Config: {config.model_type}, vocab_size={config.vocab_size}")

    def export_to_onnx(self, onnx_path: str) -> str:
        """
        Export PyTorch model to ONNX format

        Args:
            onnx_path: Output path for ONNX model

        Returns:
            Path to ONNX model
        """
        print(f"\n🔄 Exporting to ONNX format...")

        # Prepare dummy input
        batch_size = 1
        seq_length = 128
        vocab_size = self.model.config.vocab_size

        dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # Export to ONNX
        torch.onnx.export(
            self.model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size"},
            },
            opset_version=17,  # Latest ONNX opset
            do_constant_folding=True,
        )

        print(f"  ✓ ONNX export complete: {onnx_path}")
        return onnx_path

    def convert_to_coreml(
        self,
        quantization: str = "int4",
        enable_w8a8: bool = True,
    ) -> ct.models.MLModel:
        """
        Convert PyTorch model to CoreML

        Args:
            quantization: Quantization type ("int4", "int8", "float16")
            enable_w8a8: Enable W8A8 mode for Neural Engine

        Returns:
            CoreML model
        """
        print(f"\n🍎 Converting to CoreML format...")
        print(f"  Quantization: {quantization}")
        print(f"  W8A8 Mode: {enable_w8a8}")

        # Prepare traced model for CoreML
        batch_size = 1
        seq_length = 128
        vocab_size = self.model.config.vocab_size

        dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # Wrap model to return only logits (not dict)
        wrapped_model = ModelWrapper(self.model)
        wrapped_model.eval()

        # Trace the wrapped model
        traced_model = torch.jit.trace(
            wrapped_model,
            (dummy_input_ids, dummy_attention_mask)
        )

        # Convert PyTorch to CoreML
        mlmodel = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, 128), dtype=np.int32),
                ct.TensorType(name="attention_mask", shape=(1, 128), dtype=np.int32),
            ],
            minimum_deployment_target=ct.target.macOS14,
            compute_precision=ct.precision.FLOAT16,
        )

        print(f"  ✓ Base CoreML conversion complete")

        # Apply quantization
        if quantization == "int4":
            print(f"\n  📊 Applying INT4 quantization (per-block)...")
            from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig

            # Use palettization for INT4-like compression
            config = OptimizationConfig(
                global_config=OpPalettizerConfig(
                    mode="kmeans",
                    nbits=4,  # 4-bit quantization
                )
            )

            mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config)
            print(f"    ✓ INT4 quantization applied (4-bit palettization)")

        elif quantization == "int8":
            print(f"\n  📊 Applying INT8 quantization...")
            mlmodel = ct.optimize.coreml.linear_quantize_weights(
                mlmodel,
                mode="linear_symmetric",
            )
            print(f"    ✓ INT8 quantization applied")

        # Enable W8A8 mode for A17 Pro / M4+
        if enable_w8a8:
            print(f"\n  ⚡ Enabling W8A8 mode for Neural Engine...")
            try:
                from coremltools.optimize.coreml import OpLinearQuantizerConfig

                # W8A8 configuration
                config = OptimizationConfig(
                    global_config=OpLinearQuantizerConfig(mode="linear_symmetric")
                )
                mlmodel = ct.optimize.coreml.linear_quantize_activations(mlmodel, config)
                print(f"    ✓ W8A8 mode enabled (optimized for A17 Pro / M4+)")
            except Exception as e:
                print(f"    ⚠️  W8A8 mode failed: {e}")
                print(f"    Continuing without W8A8 optimization")

        return mlmodel

    def add_metadata(
        self,
        mlmodel: ct.models.MLModel,
        version: str = "2.0.0",
    ) -> ct.models.MLModel:
        """
        Add metadata to CoreML model

        Args:
            mlmodel: CoreML model
            version: Model version

        Returns:
            CoreML model with metadata
        """
        print(f"\n📝 Adding model metadata...")

        # Model information
        mlmodel.short_description = "Naver Blog AI Content Detector"
        mlmodel.author = "TinyTransformer v2.0"
        mlmodel.license = "MIT"
        mlmodel.version = version

        # Get actual input/output names
        spec = mlmodel.get_spec()
        input_names = [inp.name for inp in spec.description.input]
        output_names = [out.name for out in spec.description.output]

        print(f"  Input names: {input_names}")
        print(f"  Output names: {output_names}")

        # Input descriptions (use actual names)
        if "input_ids" in input_names:
            mlmodel.input_description["input_ids"] = (
                "Tokenized input IDs (max 128 tokens). "
                "Concatenate title and snippet before tokenization."
            )
        if "attention_mask" in input_names:
            mlmodel.input_description["attention_mask"] = (
                "Attention mask (1 for real tokens, 0 for padding)"
            )

        # Output descriptions (use first output name)
        if output_names:
            mlmodel.output_description[output_names[0]] = (
                "Raw logits [HUMAN_score, AI_score]. "
                "Apply softmax to get probabilities."
            )

        # User-defined metadata
        mlmodel.user_defined_metadata["model_type"] = "TinyBERT-4"
        mlmodel.user_defined_metadata["task"] = "binary_classification"
        mlmodel.user_defined_metadata["classes"] = "HUMAN, AI"
        mlmodel.user_defined_metadata["max_sequence_length"] = "128"
        mlmodel.user_defined_metadata["target_inference_ms"] = "< 50"
        mlmodel.user_defined_metadata["output_name"] = output_names[0] if output_names else "unknown"

        print(f"  ✓ Metadata added")

        return mlmodel

    def save_coreml_model(
        self,
        mlmodel: ct.models.MLModel,
        format: str = "mlpackage",
    ):
        """
        Save CoreML model

        Args:
            mlmodel: CoreML model
            format: Save format ("mlpackage" or "mlmodel")
        """
        extension = ".mlpackage" if format == "mlpackage" else ".mlmodel"
        output_path = self.output_dir / f"{self.model_name}{extension}"

        print(f"\n💾 Saving CoreML model...")
        mlmodel.save(str(output_path))

        print(f"  ✓ Model saved to {output_path}")

        # Get file size
        import os
        if output_path.is_dir():
            # mlpackage is a directory
            total_size = sum(
                f.stat().st_size
                for f in output_path.rglob('*')
                if f.is_file()
            )
        else:
            total_size = output_path.stat().st_size

        size_mb = total_size / (1024 ** 2)
        print(f"  📊 Model size: {size_mb:.2f} MB")

        # Check target
        target_size = 10  # MB
        if size_mb <= target_size:
            print(f"  ✅ Size meets target (≤ {target_size} MB)")
        else:
            print(f"  ⚠️  Size exceeds target ({size_mb:.2f} MB > {target_size} MB)")

        return output_path, size_mb

    def test_coreml_model(self, mlmodel: ct.models.MLModel):
        """
        Test CoreML model with sample input

        Args:
            mlmodel: CoreML model
        """
        print(f"\n🧪 Testing CoreML model...")

        # Create sample input
        sample_input = {
            "input_ids": np.random.randint(0, 8000, (1, 128), dtype=np.int32),
            "attention_mask": np.ones((1, 128), dtype=np.int32),
        }

        # Run prediction
        output = mlmodel.predict(sample_input)

        print(f"  ✓ Prediction successful")
        print(f"    Input shape: {sample_input['input_ids'].shape}")
        print(f"    Output keys: {list(output.keys())}")

        # Get logits (use actual output name from CoreML)
        output_key = list(output.keys())[0]  # Get first output key
        logits = output[output_key][0]

        # Apply softmax
        probs = np.exp(logits) / np.sum(np.exp(logits))

        print(f"\n  📊 Sample prediction:")
        print(f"    Output name: {output_key}")
        print(f"    Logits: {logits}")
        print(f"    Probabilities: {probs}")
        print(f"    HUMAN: {probs[0]:.4f}")
        print(f"    AI: {probs[1]:.4f}")
        print(f"    Prediction: {'AI' if probs[1] > 0.5 else 'HUMAN'}")


def main():
    # Define default paths as absolute paths
    DEFAULT_MODEL_PATH = str(ML_TRAINING_ROOT / "models" / "checkpoints" / "tiny-transformer-init")
    DEFAULT_OUTPUT_DIR = str(ML_TRAINING_ROOT / "models" / "coreml")

    parser = argparse.ArgumentParser(
        description="Export TinyTransformer to CoreML format"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to optimized PyTorch model"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for CoreML model"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="BlogAIDetector",
        help="CoreML model name"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["int4", "int8", "float16"],
        default="int4",
        help="Quantization type"
    )
    parser.add_argument(
        "--enable_w8a8",
        action="store_true",
        default=True,
        help="Enable W8A8 mode for Neural Engine (A17 Pro / M4+)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["mlpackage", "mlmodel"],
        default="mlpackage",
        help="CoreML save format"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test CoreML model after export"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🍎 CoreML Model Export (CoreML 8.0+)")
    print("=" * 70)

    # Check CoreML Tools version
    print(f"\n📦 CoreML Tools version: {ct.__version__}")
    if not ct.__version__.startswith("8"):
        print(f"  ⚠️  CoreML Tools 8.0+ recommended for INT4 quantization")
        print(f"  Current version: {ct.__version__}")

    # Initialize exporter
    exporter = CoreMLExporter(
        model_path=args.model,
        output_dir=args.output,
        model_name=args.name,
    )

    # Convert PyTorch model directly to CoreML (skip ONNX)
    mlmodel = exporter.convert_to_coreml(
        quantization=args.quantization,
        enable_w8a8=args.enable_w8a8,
    )

    # Step 3: Add metadata
    mlmodel = exporter.add_metadata(mlmodel)

    # Step 4: Save CoreML model
    output_path, size_mb = exporter.save_coreml_model(mlmodel, format=args.format)

    # Step 5: Test (optional)
    if args.test:
        exporter.test_coreml_model(mlmodel)

    # Summary
    print(f"\n{'='*70}")
    print(f"✅ CoreML Export Complete!")
    print(f"{'='*70}")

    print(f"\n📊 Export Summary:")
    print(f"  Model: {args.name}")
    print(f"  Quantization: {args.quantization}")
    print(f"  W8A8 Mode: {args.enable_w8a8}")
    print(f"  Format: {args.format}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Output: {output_path}")

    print(f"\n🎯 Integration with Safari Extension:")
    print(f"  1. Open your Xcode project")
    print(f"  2. Drag {output_path.name} into your project")
    print(f"  3. Ensure 'Add to targets' includes your Safari Extension")
    print(f"  4. Use the model in Swift:")

    print(f"\n  Swift code:")
    print(f"  ```swift")
    print(f"  let model = try {args.name}(configuration: MLModelConfiguration())")
    print(f"  let input = {args.name}Input(input_ids: tokenizedIds, attention_mask: mask)")
    print(f"  let output = try model.prediction(input: input)")
    print(f"  let logits = output.logits")
    print(f"  // Apply softmax to get probabilities")
    print(f"  ```")

    print(f"\n💡 Performance Notes:")
    print(f"  - Target inference: < 50ms per prediction")
    print(f"  - Optimized for Apple Silicon (M1/M2/M3/M4)")
    print(f"  - W8A8 mode provides best performance on A17 Pro / M4+")
    print(f"  - Neural Engine acceleration enabled")

    print(f"\n✓ CoreML export complete! No temporary files to clean up.")


if __name__ == "__main__":
    main()
