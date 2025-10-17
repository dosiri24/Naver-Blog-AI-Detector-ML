#!/usr/bin/env python3
"""
Comprehensive model evaluation and reporting
- Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix analysis
- Error analysis
- Performance benchmarks
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Get project root directory
SCRIPT_DIR = Path(__file__).parent  # ML-Training/scripts/
PROJECT_ROOT = SCRIPT_DIR.parent  # ML-Training/

# Add utils to path
sys.path.append(str(PROJECT_ROOT))
from utils.data_utils import load_processed_data, BlogDataset
from utils.model_utils import (
    get_device,
    estimate_inference_time,
    get_model_memory_usage,
)
from utils.tokenizer import create_tokenizer


class ModelEvaluator:
    """Comprehensive model evaluation"""

    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        output_dir: str,
        model_type: str = "pytorch",
    ):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type

        # Load model
        print(f"📥 Loading {model_type} model from {self.model_path}...")
        self.device = get_device()

        if model_type == "pytorch":
            from transformers import BertForSequenceClassification
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            self.model = self.model.to(self.device)
            self.model.eval()
        elif model_type == "coreml":
            import coremltools as ct
            self.model = ct.models.MLModel(str(self.model_path))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"✓ Model loaded successfully")

    def load_test_data(self, tokenizer) -> BlogDataset:
        """Load test dataset"""
        print(f"\n📊 Loading test data from {self.test_data_path}...")

        samples = load_processed_data(self.test_data_path)
        print(f"✓ Loaded {len(samples):,} test samples")

        # Create dataset
        test_dataset = BlogDataset(samples, tokenizer, max_length=128)

        # Print statistics
        ai_count = sum(1 for s in samples if s.label == 1)
        human_count = len(samples) - ai_count

        print(f"\n  Distribution:")
        print(f"    AI: {ai_count:,} ({ai_count/len(samples)*100:.1f}%)")
        print(f"    HUMAN: {human_count:,} ({human_count/len(samples)*100:.1f}%)")

        return test_dataset

    def evaluate_pytorch_model(
        self,
        test_dataset: BlogDataset,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate PyTorch model

        Returns:
            Tuple of (predictions, labels, probabilities)
        """
        print(f"\n🔍 Evaluating PyTorch model...")

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"]

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs)

        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def evaluate_coreml_model(
        self,
        test_dataset: BlogDataset,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate CoreML model

        Returns:
            Tuple of (predictions, labels, probabilities)
        """
        print(f"\n🔍 Evaluating CoreML model...")

        all_preds = []
        all_labels = []
        all_probs = []

        for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
            sample = test_dataset[i]
            input_ids = sample["input_ids"].numpy().reshape(1, -1).astype(np.int32)
            attention_mask = sample["attention_mask"].numpy().reshape(1, -1).astype(np.int32)
            label = sample["labels"].item()

            # Predict
            output = self.model.predict({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })

            logits = output["logits"][0]
            probs = np.exp(logits) / np.sum(np.exp(logits))

            pred = 1 if probs[1] > 0.5 else 0

            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(probs[1])

        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict:
        """Compute comprehensive metrics"""
        print(f"\n📊 Computing metrics...")

        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', pos_label=1
        )
        roc_auc = roc_auc_score(labels, probabilities)

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()

        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Same as recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "specificity": specificity,
            "sensitivity": sensitivity,
            "false_positive_rate": fpr,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        }

        return metrics

    def print_metrics(self, metrics: Dict):
        """Print formatted metrics"""
        print(f"\n{'='*70}")
        print(f"📊 Evaluation Results")
        print(f"{'='*70}")

        print(f"\n🎯 Classification Metrics:")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  F1 Score:     {metrics['f1']:.4f}")
        print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
        print(f"  Specificity:  {metrics['specificity']:.4f}")
        print(f"  FPR:          {metrics['false_positive_rate']:.4f}")

        cm = metrics['confusion_matrix']
        print(f"\n📈 Confusion Matrix:")
        print(f"                 Predicted")
        print(f"               HUMAN    AI")
        print(f"  Actual HUMAN   {cm['true_negatives']:4d}  {cm['false_positives']:4d}")
        print(f"         AI      {cm['false_negatives']:4d}  {cm['true_positives']:4d}")

    def plot_confusion_matrix(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
    ):
        """Plot confusion matrix heatmap"""
        cm = confusion_matrix(labels, predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['HUMAN', 'AI'],
            yticklabels=['HUMAN', 'AI'],
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        output_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n  ✓ Confusion matrix saved to {output_path}")
        plt.close()

    def plot_roc_curve(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        roc_auc: float,
    ):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(labels, probabilities)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        output_path = self.output_dir / 'roc_curve.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ ROC curve saved to {output_path}")
        plt.close()

    def analyze_errors(
        self,
        test_dataset: BlogDataset,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
        max_examples: int = 10,
    ):
        """Analyze prediction errors"""
        print(f"\n🔍 Error Analysis:")

        # Find errors
        errors = predictions != labels
        error_indices = np.where(errors)[0]

        num_errors = len(error_indices)
        print(f"\n  Total Errors: {num_errors} / {len(labels)} ({num_errors/len(labels)*100:.2f}%)")

        # False positives (predicted AI, actually HUMAN)
        fp_indices = np.where((predictions == 1) & (labels == 0))[0]
        print(f"  False Positives: {len(fp_indices)} (HUMAN predicted as AI)")

        # False negatives (predicted HUMAN, actually AI)
        fn_indices = np.where((predictions == 0) & (labels == 1))[0]
        print(f"  False Negatives: {len(fn_indices)} (AI predicted as HUMAN)")

        # Save error examples
        error_examples = []

        for idx in error_indices[:max_examples]:
            sample = test_dataset.samples[idx]
            error_examples.append({
                "text": sample.text[:200] + "..." if len(sample.text) > 200 else sample.text,
                "true_label": "AI" if labels[idx] == 1 else "HUMAN",
                "predicted_label": "AI" if predictions[idx] == 1 else "HUMAN",
                "confidence": float(probabilities[idx]),
            })

        error_path = self.output_dir / 'error_examples.json'
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_examples, f, ensure_ascii=False, indent=2)

        print(f"\n  ✓ Error examples saved to {error_path}")

    def benchmark_performance(
        self,
        test_dataset: BlogDataset,
        num_runs: int = 100,
    ):
        """Benchmark inference performance"""
        print(f"\n⚡ Performance Benchmarking...")

        if self.model_type == "pytorch":
            # Create dummy input
            sample = test_dataset[0]
            dummy_input = sample["input_ids"].unsqueeze(0).to(self.device)

            timing_stats = estimate_inference_time(
                self.model,
                dummy_input,
                num_runs=num_runs,
                warmup=10,
                device=self.device,
            )

            print(f"\n  Inference Time (PyTorch):")
            print(f"    Mean:   {timing_stats['mean_ms']:.2f} ms")
            print(f"    Median: {timing_stats['median_ms']:.2f} ms")
            print(f"    Min:    {timing_stats['min_ms']:.2f} ms")
            print(f"    Max:    {timing_stats['max_ms']:.2f} ms")

            target_ms = 50
            if timing_stats['mean_ms'] < target_ms:
                print(f"    ✅ Meets target (< {target_ms}ms)")
            else:
                print(f"    ⚠️  Exceeds target ({timing_stats['mean_ms']:.2f}ms > {target_ms}ms)")

            # Memory usage
            memory = get_model_memory_usage(self.model)
            print(f"\n  Memory Usage:")
            print(f"    Parameters: {memory['params_mb']:.2f} MB")
            print(f"    Buffers:    {memory['buffers_mb']:.2f} MB")
            print(f"    Total:      {memory['total_mb']:.2f} MB")

            return timing_stats

        elif self.model_type == "coreml":
            import time

            sample = test_dataset[0]
            input_ids = sample["input_ids"].numpy().reshape(1, -1).astype(np.int32)
            attention_mask = sample["attention_mask"].numpy().reshape(1, -1).astype(np.int32)

            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = self.model.predict({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                })
                end = time.perf_counter()
                times.append((end - start) * 1000)

            print(f"\n  Inference Time (CoreML):")
            print(f"    Mean:   {np.mean(times):.2f} ms")
            print(f"    Median: {np.median(times):.2f} ms")
            print(f"    Min:    {np.min(times):.2f} ms")
            print(f"    Max:    {np.max(times):.2f} ms")

            return {"mean_ms": np.mean(times), "median_ms": np.median(times)}

    def save_evaluation_report(
        self,
        metrics: Dict,
        timing_stats: Dict = None,
    ):
        """Save comprehensive evaluation report"""
        report = {
            "model_path": str(self.model_path),
            "model_type": self.model_type,
            "test_data": str(self.test_data_path),
            "metrics": metrics,
        }

        if timing_stats:
            report["performance"] = timing_stats

        report_path = self.output_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 Evaluation report saved to {report_path}")


def main():
    # Define default paths as absolute paths
    DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "checkpoints" / "final")
    DEFAULT_TEST_DATA = str(PROJECT_ROOT / "data" / "processed" / "test.json")
    DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "models" / "evaluation")

    parser = argparse.ArgumentParser(
        description="Evaluate TinyTransformer model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to model"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=DEFAULT_TEST_DATA,
        help="Path to test data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["pytorch", "coreml"],
        default="pytorch",
        help="Model type"
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="sentencepiece",
        help="Tokenizer type"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarking"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("📊 TinyTransformer Model Evaluation")
    print("=" * 70)

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        test_data_path=args.test_data,
        output_dir=args.output,
        model_type=args.model_type,
    )

    # Create tokenizer
    print(f"\n📝 Creating tokenizer ({args.tokenizer_type})...")
    tokenizer = create_tokenizer(tokenizer_type=args.tokenizer_type)

    # Load test data
    test_dataset = evaluator.load_test_data(tokenizer)

    # Evaluate model
    if args.model_type == "pytorch":
        predictions, labels, probabilities = evaluator.evaluate_pytorch_model(
            test_dataset,
            batch_size=args.batch_size,
        )
    elif args.model_type == "coreml":
        predictions, labels, probabilities = evaluator.evaluate_coreml_model(test_dataset)

    # Compute metrics
    metrics = evaluator.compute_metrics(predictions, labels, probabilities)

    # Print results
    evaluator.print_metrics(metrics)

    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    evaluator.plot_confusion_matrix(labels, predictions)
    evaluator.plot_roc_curve(labels, probabilities, metrics['roc_auc'])

    # Error analysis
    evaluator.analyze_errors(test_dataset, predictions, labels, probabilities)

    # Performance benchmarking
    timing_stats = None
    if args.benchmark:
        timing_stats = evaluator.benchmark_performance(test_dataset)

    # Save report
    evaluator.save_evaluation_report(metrics, timing_stats)

    print(f"\n{'='*70}")
    print(f"✅ Evaluation Complete!")
    print(f"{'='*70}")

    print(f"\nResults saved to: {args.output}")
    print(f"  - evaluation_report.json")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    print(f"  - error_examples.json")


if __name__ == "__main__":
    main()
