"""
Data utilities for loading, preprocessing, and managing datasets
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


@dataclass
class BlogSample:
    """Single blog sample data structure"""
    text: str
    label: int  # 0: HUMAN, 1: AI
    source: str  # "real" or "synthetic"
    original_label: Optional[str] = None  # "HUMAN" or "AI"
    metadata: Optional[Dict] = None


class BlogDataset(Dataset):
    """PyTorch Dataset for blog classification"""

    def __init__(
        self,
        samples: List[BlogSample],
        tokenizer,
        max_length: int = 128,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Tokenize
        encoding = self.tokenizer(
            sample.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample.label, dtype=torch.long),
        }


class DatasetPreprocessor:
    """Preprocess and validate dataset"""

    def __init__(self, min_length: int = 10, max_length: int = 500):
        self.min_length = min_length
        self.max_length = max_length

    def validate_sample(self, text: str) -> bool:
        """Validate a single sample"""
        if not text or not isinstance(text, str):
            return False

        text_length = len(text.strip())

        # Length check
        if text_length < self.min_length or text_length > self.max_length:
            return False

        # Korean character check (at least some Korean)
        has_korean = any('\uac00' <= char <= '\ud7a3' for char in text)
        if not has_korean:
            return False

        return True

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def process_dataset(
        self,
        samples: List[Dict],
        source: str = "unknown"
    ) -> List[BlogSample]:
        """Process and validate entire dataset"""
        processed_samples = []

        for sample in tqdm(samples, desc=f"Processing {source} data"):
            # Handle both formats: direct "text" field or "title" + "snippet_text"
            text = sample.get("text", "")

            # If no direct "text" field, combine title and snippet_text
            if not text:
                title = sample.get("title", "")
                snippet_text = sample.get("snippet_text", "")
                if title and snippet_text:
                    text = f"{title}\n\n{snippet_text}"
                else:
                    text = title or snippet_text

            # Skip invalid samples
            if not self.validate_sample(text):
                continue

            # Preprocess text
            text = self.preprocess_text(text)

            # Convert label
            label_str = sample.get("label", "HUMAN")
            label = 1 if label_str.upper() == "AI" else 0

            blog_sample = BlogSample(
                text=text,
                label=label,
                source=source,
                original_label=label_str,
                metadata=sample.get("metadata", {}),
            )

            processed_samples.append(blog_sample)

        return processed_samples


def load_real_data(data_path: Union[str, Path]) -> List[Dict]:
    """
    Load real data from training_data.json

    Args:
        data_path: Path to training_data.json

    Returns:
        List of sample dictionaries
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data)} real samples from {data_path}")
    return data


def load_synthetic_data(synthetic_dir: Union[str, Path]) -> List[Dict]:
    """
    Load synthetic data from multiple JSON files

    Args:
        synthetic_dir: Directory containing synthetic data files

    Returns:
        List of sample dictionaries
    """
    synthetic_dir = Path(synthetic_dir)

    if not synthetic_dir.exists():
        print(f"⚠ Synthetic data directory not found: {synthetic_dir}")
        return []

    all_samples = []

    # Load all JSON files in synthetic directory
    json_files = list(synthetic_dir.glob("*.json"))

    if not json_files:
        print(f"⚠ No JSON files found in {synthetic_dir}")
        return []

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            samples = json.load(f)
            all_samples.extend(samples)

    print(f"✓ Loaded {len(all_samples)} synthetic samples from {synthetic_dir}")
    return all_samples


def create_train_val_test_split(
    samples: List[BlogSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[List[BlogSample], List[BlogSample], List[BlogSample]]:
    """
    Split dataset into train, validation, and test sets

    Args:
        samples: List of BlogSample objects
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
        stratify: Stratify by label

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)

    # Extract labels for stratification
    labels = [sample.label for sample in samples]

    # First split: train + val vs test
    train_val_samples, test_samples = train_test_split(
        samples,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels if stratify else None,
    )

    # Second split: train vs val
    train_val_labels = [sample.label for sample in train_val_samples]
    val_size = val_ratio / (train_ratio + val_ratio)

    train_samples, val_samples = train_test_split(
        train_val_samples,
        test_size=val_size,
        random_state=seed,
        stratify=train_val_labels if stratify else None,
    )

    # Print statistics
    print(f"\n📊 Dataset Split Statistics:")
    print(f"  Train: {len(train_samples):,} samples "
          f"(AI: {sum(1 for s in train_samples if s.label == 1):,}, "
          f"HUMAN: {sum(1 for s in train_samples if s.label == 0):,})")
    print(f"  Val:   {len(val_samples):,} samples "
          f"(AI: {sum(1 for s in val_samples if s.label == 1):,}, "
          f"HUMAN: {sum(1 for s in val_samples if s.label == 0):,})")
    print(f"  Test:  {len(test_samples):,} samples "
          f"(AI: {sum(1 for s in test_samples if s.label == 1):,}, "
          f"HUMAN: {sum(1 for s in test_samples if s.label == 0):,})")

    return train_samples, val_samples, test_samples


def save_processed_data(
    samples: List[BlogSample],
    output_path: Union[str, Path],
):
    """Save processed samples to JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert BlogSample to dict
    data = []
    for sample in samples:
        data.append({
            "text": sample.text,
            "label": "AI" if sample.label == 1 else "HUMAN",
            "source": sample.source,
            "metadata": sample.metadata or {},
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(data):,} samples to {output_path}")


def load_processed_data(data_path: Union[str, Path]) -> List[BlogSample]:
    """Load processed samples from JSON"""
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        label = 1 if item["label"].upper() == "AI" else 0
        samples.append(BlogSample(
            text=item["text"],
            label=label,
            source=item.get("source", "unknown"),
            original_label=item["label"],
            metadata=item.get("metadata", {}),
        ))

    return samples


def get_dataset_statistics(samples: List[BlogSample]) -> Dict:
    """Calculate dataset statistics"""
    texts = [sample.text for sample in samples]
    labels = [sample.label for sample in samples]

    # Length statistics
    lengths = [len(text) for text in texts]

    # Word count statistics
    word_counts = [len(text.split()) for text in texts]

    # Label distribution
    ai_count = sum(1 for label in labels if label == 1)
    human_count = sum(1 for label in labels if label == 0)

    stats = {
        "total_samples": len(samples),
        "ai_samples": ai_count,
        "human_samples": human_count,
        "ai_ratio": ai_count / len(samples) if samples else 0,
        "text_length": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": np.mean(lengths) if lengths else 0,
            "median": np.median(lengths) if lengths else 0,
            "std": np.std(lengths) if lengths else 0,
        },
        "word_count": {
            "min": min(word_counts) if word_counts else 0,
            "max": max(word_counts) if word_counts else 0,
            "mean": np.mean(word_counts) if word_counts else 0,
            "median": np.median(word_counts) if word_counts else 0,
            "std": np.std(word_counts) if word_counts else 0,
        },
        "source_distribution": {
            source: sum(1 for s in samples if s.source == source)
            for source in set(s.source for s in samples)
        },
    }

    return stats


def print_dataset_statistics(samples: List[BlogSample], name: str = "Dataset"):
    """Print formatted dataset statistics"""
    stats = get_dataset_statistics(samples)

    print(f"\n📊 {name} Statistics:")
    print(f"  Total Samples: {stats['total_samples']:,}")
    print(f"  AI Samples: {stats['ai_samples']:,} ({stats['ai_ratio']:.1%})")
    print(f"  HUMAN Samples: {stats['human_samples']:,} "
          f"({1 - stats['ai_ratio']:.1%})")

    print(f"\n  Text Length:")
    print(f"    Min: {stats['text_length']['min']:.0f} chars")
    print(f"    Max: {stats['text_length']['max']:.0f} chars")
    print(f"    Mean: {stats['text_length']['mean']:.1f} ± "
          f"{stats['text_length']['std']:.1f} chars")

    print(f"\n  Word Count:")
    print(f"    Min: {stats['word_count']['min']:.0f} words")
    print(f"    Max: {stats['word_count']['max']:.0f} words")
    print(f"    Mean: {stats['word_count']['mean']:.1f} ± "
          f"{stats['word_count']['std']:.1f} words")

    if stats['source_distribution']:
        print(f"\n  Source Distribution:")
        for source, count in stats['source_distribution'].items():
            print(f"    {source}: {count:,} samples")
