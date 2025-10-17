#!/usr/bin/env python3
"""
Build TinyTransformer model architecture
Based on TinyBERT-4 (2024-2025 research)

Target: 15M parameters, 5-10MB final size (INT4)
"""

import argparse
import sys
from pathlib import Path
import yaml

import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    AutoConfig,
    AutoModelForSequenceClassification,
)

# Get project root directory
SCRIPT_DIR = Path(__file__).parent  # ML-Training/scripts/
PROJECT_ROOT = SCRIPT_DIR.parent  # ML-Training/

# Add utils to path
sys.path.append(str(PROJECT_ROOT))
from utils.model_utils import print_model_summary, get_device


class TinyTransformerConfig:
    """Configuration for TinyTransformer model"""

    def __init__(self, config_path: str = None):
        if config_path:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
                arch = config_dict.get("architecture", {})
        else:
            # Default TinyBERT-4 configuration
            arch = {}

        # Model architecture (TinyBERT-4 based)
        self.vocab_size = arch.get("vocab_size", 8000)
        self.hidden_size = arch.get("hidden_size", 312)
        self.num_hidden_layers = arch.get("num_hidden_layers", 4)
        self.num_attention_heads = arch.get("num_attention_heads", 12)
        self.intermediate_size = arch.get("intermediate_size", 1200)
        self.max_position_embeddings = arch.get("max_position_embeddings", 128)
        self.type_vocab_size = arch.get("type_vocab_size", 2)
        self.hidden_dropout_prob = arch.get("hidden_dropout_prob", 0.1)
        self.attention_probs_dropout_prob = arch.get("attention_probs_dropout_prob", 0.1)

        # Classification
        self.num_labels = 2  # AI / HUMAN
        self.classifier_dropout = arch.get("classifier_dropout", 0.1)

        # Initialization
        self.initializer_range = arch.get("initializer_range", 0.02)
        self.layer_norm_eps = arch.get("layer_norm_eps", 1e-12)

    def to_bert_config(self) -> BertConfig:
        """Convert to Hugging Face BertConfig"""
        return BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            num_labels=self.num_labels,
            classifier_dropout=self.classifier_dropout,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
        )


def create_tiny_transformer(
    config: TinyTransformerConfig,
    pretrained: bool = False,
) -> BertForSequenceClassification:
    """
    Create TinyTransformer model for binary classification

    Args:
        config: TinyTransformerConfig instance
        pretrained: Whether to load pretrained weights (not applicable for custom config)

    Returns:
        BertForSequenceClassification model
    """
    # Convert to BERT config
    bert_config = config.to_bert_config()

    # Create model from config (from scratch)
    model = BertForSequenceClassification(bert_config)

    print(f"\n✓ Created TinyTransformer model from scratch")
    print(f"  Architecture: TinyBERT-4 based")
    print(f"  Vocab Size: {config.vocab_size:,}")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Layers: {config.num_hidden_layers}")
    print(f"  Attention Heads: {config.num_attention_heads}")
    print(f"  Max Sequence Length: {config.max_position_embeddings}")

    return model


def verify_model_architecture(model: nn.Module):
    """
    Verify model architecture and dimensions

    Args:
        model: PyTorch model to verify
    """
    print("\n🔍 Verifying Model Architecture...")

    # Get model config
    config = model.config

    # Verify embedding dimensions
    expected_embedding_dim = config.hidden_size
    actual_embedding_dim = model.bert.embeddings.word_embeddings.embedding_dim

    assert actual_embedding_dim == expected_embedding_dim, \
        f"Embedding dim mismatch: {actual_embedding_dim} != {expected_embedding_dim}"

    print(f"  ✓ Embedding dimension: {actual_embedding_dim}")

    # Verify number of layers
    num_layers = len(model.bert.encoder.layer)
    assert num_layers == config.num_hidden_layers, \
        f"Layer count mismatch: {num_layers} != {config.num_hidden_layers}"

    print(f"  ✓ Transformer layers: {num_layers}")

    # Verify attention heads
    first_layer = model.bert.encoder.layer[0]
    num_heads = first_layer.attention.self.num_attention_heads

    assert num_heads == config.num_attention_heads, \
        f"Attention heads mismatch: {num_heads} != {config.num_attention_heads}"

    print(f"  ✓ Attention heads: {num_heads}")

    # Verify classifier
    classifier_in_features = model.classifier.in_features
    classifier_out_features = model.classifier.out_features

    assert classifier_in_features == config.hidden_size, \
        f"Classifier input mismatch: {classifier_in_features} != {config.hidden_size}"
    assert classifier_out_features == config.num_labels, \
        f"Classifier output mismatch: {classifier_out_features} != {config.num_labels}"

    print(f"  ✓ Classifier: {classifier_in_features} → {classifier_out_features}")

    print("\n✅ Model architecture verified successfully!")


def test_model_forward_pass(model: nn.Module, vocab_size: int = 8000):
    """
    Test model forward pass with dummy input

    Args:
        model: PyTorch model
        vocab_size: Vocabulary size for dummy input
    """
    print("\n🧪 Testing Forward Pass...")

    device = next(model.parameters()).device
    batch_size = 4
    seq_length = 128

    # Create dummy input
    dummy_input_ids = torch.randint(
        0, vocab_size,
        (batch_size, seq_length),
        device=device
    )
    dummy_attention_mask = torch.ones(
        (batch_size, seq_length),
        device=device
    )
    dummy_labels = torch.randint(
        0, 2,
        (batch_size,),
        device=device
    )

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            labels=dummy_labels,
        )

    # Verify output shape
    logits = outputs.logits
    expected_shape = (batch_size, 2)  # Binary classification

    assert logits.shape == expected_shape, \
        f"Output shape mismatch: {logits.shape} != {expected_shape}"

    print(f"  ✓ Input shape: {dummy_input_ids.shape}")
    print(f"  ✓ Output shape: {logits.shape}")
    print(f"  ✓ Loss: {outputs.loss.item():.4f}")

    # Verify probabilities
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, device=device)), \
        "Probabilities don't sum to 1"

    print(f"  ✓ Probabilities sum to 1.0")
    print(f"  ✓ Sample prediction: {probs[0].cpu().numpy()}")

    print("\n✅ Forward pass test successful!")


def save_model_architecture(
    model: nn.Module,
    output_dir: str,
    model_name: str = "tiny-transformer",
):
    """
    Save model architecture and configuration

    Args:
        model: PyTorch model
        output_dir: Output directory
        model_name: Model name for saving
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save_pretrained(output_dir)
    print(f"\n✓ Model saved to {output_dir}")

    # Save architecture summary
    summary_path = output_dir / "architecture_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Architecture: TinyBERT-4 based\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Vocab Size: {model.config.vocab_size:,}\n")
        f.write(f"  Hidden Size: {model.config.hidden_size}\n")
        f.write(f"  Num Layers: {model.config.num_hidden_layers}\n")
        f.write(f"  Attention Heads: {model.config.num_attention_heads}\n")
        f.write(f"  Intermediate Size: {model.config.intermediate_size}\n")
        f.write(f"  Max Seq Length: {model.config.max_position_embeddings}\n")
        f.write(f"  Num Labels: {model.config.num_labels}\n\n")

        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        f.write(f"Parameters:\n")
        f.write(f"  Total: {total_params:,}\n")
        f.write(f"  Trainable: {trainable_params:,}\n\n")

        # Size estimates
        size_fp32 = (total_params * 4) / (1024 ** 2)
        size_int4 = size_fp32 / 8

        f.write(f"Size Estimates:\n")
        f.write(f"  FP32: {size_fp32:.2f} MB\n")
        f.write(f"  INT4: {size_int4:.2f} MB\n")

    print(f"✓ Architecture summary saved to {summary_path}")


def main():
    # Define default paths as absolute paths
    DEFAULT_CONFIG_PATH = str(PROJECT_ROOT / "configs" / "tiny_transformer.yaml")
    DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "models" / "checkpoints" / "tiny-transformer-init")

    parser = argparse.ArgumentParser(
        description="Build TinyTransformer model architecture"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run forward pass test"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use (auto-detect if not specified)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🏗️  TinyTransformer Model Builder")
    print("=" * 70)

    # Load configuration
    print(f"\n📋 Loading configuration from {args.config}...")
    config = TinyTransformerConfig(args.config)

    # Create model
    print("\n🔨 Building model...")
    model = create_tiny_transformer(config)

    # Print summary
    print_model_summary(model, "TinyTransformer")

    # Verify architecture
    verify_model_architecture(model)

    # Get device
    if args.device:
        device = torch.device(args.device)
        print(f"\n📍 Using specified device: {device}")
    else:
        device = get_device()

    model = model.to(device)

    # Test forward pass
    if args.test:
        test_model_forward_pass(model, config.vocab_size)

    # Save model
    print(f"\n💾 Saving model to {args.output}...")
    save_model_architecture(model, args.output, "TinyTransformer")

    print("\n✅ Model building complete!")
    print(f"\nNext steps:")
    print(f"  1. Train tokenizer on your data")
    print(f"  2. Run 3_train_from_scratch.py to train the model")
    print(f"  3. Optimize with 4_optimize.py")
    print(f"  4. Export to CoreML with 5_export_coreml.py")


if __name__ == "__main__":
    main()
