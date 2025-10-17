#!/usr/bin/env python3
"""
Train TinyTransformer From Scratch
Training on 10,297 samples (297 real + 10,000 synthetic)

No knowledge distillation, no pretrained weights - pure from scratch training
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import json
from typing import Dict, List

import torch
import numpy as np
from transformers import (
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not installed. W&B logging disabled.")

# Get project root directory
SCRIPT_DIR = Path(__file__).parent  # ML-Training/scripts/
PROJECT_ROOT = SCRIPT_DIR.parent  # ML-Training/

# Add utils to path
sys.path.append(str(PROJECT_ROOT))
from utils.data_utils import (
    load_real_data,
    load_synthetic_data,
    DatasetPreprocessor,
    create_train_val_test_split,
    BlogDataset,
    save_processed_data,
    print_dataset_statistics,
)
from utils.model_utils import (
    get_device,
    set_seed,
    print_model_summary,
)
from utils.tokenizer import create_tokenizer


class TrainingConfig:
    """Training configuration from YAML"""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Data paths
        data_config = config.get("data", {})
        self.real_data_path = data_config.get("real_data")
        self.synthetic_data_path = data_config.get("synthetic_data")
        self.processed_data_path = data_config.get("processed_data")

        # Split ratios
        self.train_ratio = data_config.get("train_ratio", 0.8)
        self.val_ratio = data_config.get("val_ratio", 0.1)
        self.test_ratio = data_config.get("test_ratio", 0.1)

        # Training hyperparameters
        train_config = config.get("training", {})
        self.num_epochs = train_config.get("num_train_epochs", 15)
        self.batch_size = train_config.get("per_device_train_batch_size", 32)
        self.eval_batch_size = train_config.get("per_device_eval_batch_size", 64)
        self.learning_rate = train_config.get("learning_rate", 5e-4)
        self.weight_decay = train_config.get("weight_decay", 0.01)
        self.warmup_steps = train_config.get("warmup_steps", 500)
        self.gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = train_config.get("max_grad_norm", 1.0)

        # Scheduler
        self.lr_scheduler_type = train_config.get("lr_scheduler_type", "cosine")

        # Checkpointing
        checkpoint_config = config.get("checkpointing", {})
        self.output_dir = checkpoint_config.get("output_dir", "models/checkpoints")
        self.logging_steps = train_config.get("logging_steps", 50)
        self.eval_steps = train_config.get("eval_steps", 200)
        self.save_steps = train_config.get("save_steps", 200)
        self.save_total_limit = train_config.get("save_total_limit", 3)

        # Early stopping
        early_stop_config = config.get("early_stopping", {})
        self.early_stopping_patience = early_stop_config.get("patience", 3)

        # Regularization
        reg_config = config.get("regularization", {})
        self.label_smoothing = reg_config.get("label_smoothing", 0.1)

        # Seed
        self.seed = config.get("seed", 42)

        # W&B
        wandb_config = config.get("wandb", {})
        self.use_wandb = wandb_config.get("enabled", False)
        self.wandb_project = wandb_config.get("project", "naver-blog-ai-detector")


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation

    Args:
        eval_pred: EvalPrediction object with predictions and labels

    Returns:
        Dictionary of metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Accuracy
    accuracy = accuracy_score(labels, predictions)

    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='binary',
        pos_label=1,  # AI class
    )

    # ROC-AUC (using probabilities)
    probabilities = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    roc_auc = roc_auc_score(labels, probabilities)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def prepare_datasets(config: TrainingConfig, tokenizer):
    """
    Load and prepare training datasets

    Args:
        config: TrainingConfig instance
        tokenizer: Tokenizer

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    print("\n" + "=" * 70)
    print("📊 Loading and Preparing Datasets")
    print("=" * 70)

    # Load real data
    print(f"\n1️⃣ Loading real data from {config.real_data_path}")
    real_data = load_real_data(config.real_data_path)

    # Load synthetic data
    print(f"\n2️⃣ Loading synthetic data from {config.synthetic_data_path}")
    synthetic_data = load_synthetic_data(config.synthetic_data_path)

    # Combine datasets
    all_data = real_data + synthetic_data
    print(f"\n✓ Combined dataset: {len(all_data):,} samples")
    print(f"  Real: {len(real_data):,} ({len(real_data)/len(all_data)*100:.1f}%)")
    print(f"  Synthetic: {len(synthetic_data):,} ({len(synthetic_data)/len(all_data)*100:.1f}%)")

    # Preprocess
    print(f"\n3️⃣ Preprocessing and validating...")
    preprocessor = DatasetPreprocessor(min_length=10, max_length=500)

    real_samples = preprocessor.process_dataset(real_data, source="real")
    synthetic_samples = preprocessor.process_dataset(synthetic_data, source="synthetic")
    all_samples = real_samples + synthetic_samples

    print(f"\n✓ After preprocessing: {len(all_samples):,} samples")

    # Split into train/val/test
    print(f"\n4️⃣ Splitting dataset...")
    train_samples, val_samples, test_samples = create_train_val_test_split(
        all_samples,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        stratify=True,
    )

    # Save processed datasets
    processed_dir = Path(config.processed_data_path)
    save_processed_data(train_samples, processed_dir / "train.json")
    save_processed_data(val_samples, processed_dir / "val.json")
    save_processed_data(test_samples, processed_dir / "test.json")

    # Print statistics
    print_dataset_statistics(train_samples, "Training Set")
    print_dataset_statistics(val_samples, "Validation Set")
    print_dataset_statistics(test_samples, "Test Set")

    # Create PyTorch datasets
    print(f"\n5️⃣ Creating PyTorch datasets...")
    train_dataset = BlogDataset(train_samples, tokenizer, max_length=128)
    val_dataset = BlogDataset(val_samples, tokenizer, max_length=128)
    test_dataset = BlogDataset(test_samples, tokenizer, max_length=128)

    print(f"\n✅ Datasets prepared successfully!")

    return train_dataset, val_dataset, test_dataset


def train_model(
    model,
    train_dataset,
    val_dataset,
    config: TrainingConfig,
):
    """
    Train model from scratch

    Args:
        model: TinyTransformer model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: TrainingConfig

    Returns:
        Trained model
    """
    print("\n" + "=" * 70)
    print("🚀 Starting From Scratch Training")
    print("=" * 70)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        label_smoothing_factor=config.label_smoothing,
        report_to="wandb" if config.use_wandb else "none",
        logging_dir=f"{config.output_dir}/runs",
        seed=config.seed,
        fp16=torch.cuda.is_available(),  # Enable FP16 if CUDA available
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    print(f"\n📋 Training Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Warmup Steps: {config.warmup_steps}")
    print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"  Label Smoothing: {config.label_smoothing}")
    print(f"  Early Stopping Patience: {config.early_stopping_patience}")

    # Initialize W&B if enabled and available
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            name=f"tiny-transformer-from-scratch-{config.seed}",
            config={
                "architecture": "TinyBERT-4",
                "training_approach": "from_scratch",
                "dataset_size": len(train_dataset),
                "epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
            }
        )
    elif config.use_wandb and not WANDB_AVAILABLE:
        print("⚠️  W&B logging requested but wandb not available. Continuing without W&B.")

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        ],
    )

    # Print training info
    print(f"\n🎯 Training Details:")
    print(f"  Training Samples: {len(train_dataset):,}")
    print(f"  Validation Samples: {len(val_dataset):,}")
    print(f"  Total Steps: {len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps) * config.num_epochs:,}")
    print(f"  Device: {training_args.device}")

    # Train model
    print(f"\n⏱️  Training started...")
    train_result = trainer.train()

    # Print training results
    print(f"\n✅ Training completed!")
    print(f"\n📊 Training Results:")
    print(f"  Final Loss: {train_result.training_loss:.4f}")
    print(f"  Training Time: {train_result.metrics['train_runtime']:.2f}s")
    print(f"  Samples/Second: {train_result.metrics['train_samples_per_second']:.2f}")

    # Evaluate on validation set
    print(f"\n📈 Evaluating on validation set...")
    eval_results = trainer.evaluate()

    print(f"\n📊 Validation Results:")
    print(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  Precision: {eval_results['eval_precision']:.4f}")
    print(f"  Recall: {eval_results['eval_recall']:.4f}")
    print(f"  F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"  ROC-AUC: {eval_results['eval_roc_auc']:.4f}")

    # Save final model
    final_model_path = Path(config.output_dir) / "final"
    trainer.save_model(final_model_path)
    print(f"\n💾 Final model saved to {final_model_path}")

    # Save training metrics
    metrics_path = Path(config.output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics['train_runtime'],
            "eval_accuracy": eval_results['eval_accuracy'],
            "eval_precision": eval_results['eval_precision'],
            "eval_recall": eval_results['eval_recall'],
            "eval_f1": eval_results['eval_f1'],
            "eval_roc_auc": eval_results['eval_roc_auc'],
        }, f, indent=2)

    print(f"✓ Training metrics saved to {metrics_path}")

    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    return trainer.model


def main():
    # Define default paths as absolute paths
    DEFAULT_TRAINING_CONFIG = str(PROJECT_ROOT / "configs" / "training_config.yaml")
    DEFAULT_MODEL_CONFIG = str(PROJECT_ROOT / "configs" / "tiny_transformer.yaml")

    parser = argparse.ArgumentParser(
        description="Train TinyTransformer from scratch"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_TRAINING_CONFIG,
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default=DEFAULT_MODEL_CONFIG,
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="sentencepiece",
        choices=["sentencepiece", "solar", "exaone", "kobert"],
        help="Tokenizer type to use"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🎓 TinyTransformer From Scratch Training")
    print("=" * 70)

    # Load training config
    config = TrainingConfig(args.config)

    # Set random seed
    set_seed(config.seed)

    # Create tokenizer
    print(f"\n📝 Creating tokenizer ({args.tokenizer_type})...")
    tokenizer = create_tokenizer(
        tokenizer_type=args.tokenizer_type,
        vocab_size=8000,
    )

    # Save tokenizer immediately (for Swift integration)
    tokenizer_save_dir = Path(config.output_dir) / "tokenizer"
    tokenizer_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving tokenizer to {tokenizer_save_dir}...")
    tokenizer.save_pretrained(str(tokenizer_save_dir))

    # Also save vocab.txt separately for easy Swift access
    vocab_file = tokenizer_save_dir / "vocab.txt"
    print(f"✅ Tokenizer saved with vocab.txt ({tokenizer.vocab_size:,} tokens)")
    print(f"   Location: {vocab_file}")
    print(f"   ⚠️  IMPORTANT: Use this vocab.txt for Swift integration!")

    # Load model
    print(f"\n🏗️  Loading model from {args.model_config}...")
    from scripts.build_model import TinyTransformerConfig, create_tiny_transformer

    model_config = TinyTransformerConfig(args.model_config)
    model = create_tiny_transformer(model_config)

    # Print model summary
    print_model_summary(model, "TinyTransformer")

    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(config, tokenizer)

    # Train model
    model = train_model(model, train_dataset, val_dataset, config)

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("📊 Final Evaluation on Test Set")
    print("=" * 70)

    device = get_device()
    model = model.to(device)
    model.eval()

    # Simple test evaluation
    from torch.utils.data import DataLoader

    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    # Compute test metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', pos_label=1
    )
    test_roc_auc = roc_auc_score(all_labels, all_probs)

    print(f"\n📊 Test Set Results:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  ROC-AUC: {test_roc_auc:.4f}")

    # Save test metrics
    test_metrics_path = Path(config.output_dir) / "test_metrics.json"
    with open(test_metrics_path, "w") as f:
        json.dump({
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "roc_auc": test_roc_auc,
        }, f, indent=2)

    print(f"\n✓ Test metrics saved to {test_metrics_path}")

    print("\n✅ Training complete!")
    print(f"\nNext steps:")
    print(f"  1. Run 4_optimize.py to quantize and prune the model")
    print(f"  2. Run 5_export_coreml.py to convert to CoreML format")
    print(f"  3. Integrate into Safari Extension")


if __name__ == "__main__":
    main()
