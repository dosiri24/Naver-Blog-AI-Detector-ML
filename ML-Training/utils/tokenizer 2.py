"""
Tokenizer utilities for Korean text processing
Supports SentencePiece and modern Korean tokenizers (Solar Pro 2, Exaone 4.0)
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Union

import sentencepiece as spm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class TokenizerWrapper:
    """
    Wrapper for different tokenizer types with unified interface
    """

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
        **kwargs,
    ):
        """Tokenize text with unified interface"""
        max_length = max_length or self.max_length

        return self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            **kwargs,
        )

    def decode(self, token_ids, skip_special_tokens: bool = True):
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, token_ids_list, skip_special_tokens: bool = True):
        """Decode batch of token IDs to text"""
        return self.tokenizer.batch_decode(
            token_ids_list,
            skip_special_tokens=skip_special_tokens
        )

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.tokenizer)

    def save_pretrained(self, output_dir: str):
        """Save tokenizer"""
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load pretrained tokenizer"""
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        return cls(tokenizer)


def create_tokenizer(
    tokenizer_type: str = "sentencepiece",
    vocab_size: int = 8000,
    model_path: Optional[str] = None,
    **kwargs,
) -> TokenizerWrapper:
    """
    Create tokenizer based on type

    Args:
        tokenizer_type: Type of tokenizer
            - "sentencepiece": Train custom SentencePiece
            - "solar": Solar Pro 2 tokenizer (Upstage)
            - "exaone": Exaone 4.0 tokenizer (LG AI Research)
            - "kobert": KoBERT tokenizer
            - "pretrained": Load from pretrained model
        vocab_size: Vocabulary size for custom tokenizers
        model_path: Path to pretrained model (for pretrained type)
        **kwargs: Additional arguments

    Returns:
        TokenizerWrapper instance
    """
    from transformers import AutoTokenizer

    if tokenizer_type == "pretrained" and model_path:
        # Load from pretrained model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"✓ Loaded pretrained tokenizer from {model_path}")

    elif tokenizer_type == "solar":
        # Solar Pro 2 tokenizer (Upstage)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "upstage/solar-pro-preview-instruct",
                trust_remote_code=True,
            )
            print("✓ Loaded Solar Pro 2 tokenizer")
        except Exception as e:
            print(f"⚠ Failed to load Solar tokenizer: {e}")
            print("  Falling back to SentencePiece")
            return create_sentencepiece_tokenizer(vocab_size)

    elif tokenizer_type == "exaone":
        # Exaone 4.0 tokenizer (LG AI Research)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
                trust_remote_code=True,
            )
            print("✓ Loaded Exaone 4.0 tokenizer")
        except Exception as e:
            print(f"⚠ Failed to load Exaone tokenizer: {e}")
            print("  Falling back to SentencePiece")
            return create_sentencepiece_tokenizer(vocab_size)

    elif tokenizer_type == "kobert":
        # KoBERT tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "monologg/kobert",
                trust_remote_code=True,
            )
            print("✓ Loaded KoBERT tokenizer")
        except Exception as e:
            print(f"⚠ Failed to load KoBERT tokenizer: {e}")
            print("  Falling back to SentencePiece")
            return create_sentencepiece_tokenizer(vocab_size)

    elif tokenizer_type == "sentencepiece":
        # Custom SentencePiece tokenizer
        return create_sentencepiece_tokenizer(vocab_size, **kwargs)

    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    return TokenizerWrapper(tokenizer)


def create_sentencepiece_tokenizer(
    vocab_size: int = 8000,
    model_prefix: str = "korean_blog_tokenizer",
    model_dir: str = "utils",
    train_data: Optional[List[str]] = None,
    **kwargs,
) -> TokenizerWrapper:
    """
    Create custom SentencePiece tokenizer for Korean

    Args:
        vocab_size: Vocabulary size
        model_prefix: Model file prefix
        model_dir: Directory to save model
        train_data: Training texts (optional)
        **kwargs: Additional SentencePiece arguments

    Returns:
        TokenizerWrapper with SentencePiece tokenizer
    """
    from transformers import PreTrainedTokenizerFast

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_prefix}.model"
    vocab_path = model_dir / f"{model_prefix}.vocab"

    # Check if model already exists
    if model_path.exists():
        print(f"✓ Loading existing SentencePiece model from {model_path}")
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(model_path),
            model_max_length=128,
        )
        return TokenizerWrapper(tokenizer)

    # Train new SentencePiece model
    if train_data is None:
        print("⚠ No training data provided for SentencePiece")
        print("  Using default Korean tokenizer instead")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
        return TokenizerWrapper(tokenizer)

    # Prepare training data file
    train_file = model_dir / "tokenizer_train.txt"
    with open(train_file, "w", encoding="utf-8") as f:
        for text in train_data:
            f.write(text + "\n")

    # SentencePiece training parameters
    sp_params = {
        "input": str(train_file),
        "model_prefix": str(model_dir / model_prefix),
        "vocab_size": vocab_size,
        "character_coverage": 0.9995,  # High coverage for Korean
        "model_type": "unigram",       # Unigram language model
        "pad_id": 0,
        "unk_id": 1,
        "bos_id": 2,
        "eos_id": 3,
        "user_defined_symbols": ["[CLS]", "[SEP]", "[MASK]"],
        **kwargs,
    }

    # Train SentencePiece model
    print(f"Training SentencePiece tokenizer with vocab_size={vocab_size}...")
    spm.SentencePieceTrainer.train(**sp_params)

    # Clean up training file
    train_file.unlink()

    # Load trained model
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(model_path),
        model_max_length=128,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )

    print(f"✓ SentencePiece tokenizer trained and saved to {model_path}")

    return TokenizerWrapper(tokenizer)


def train_tokenizer_from_data(
    data_path: str,
    vocab_size: int = 8000,
    output_dir: str = "utils",
    model_prefix: str = "korean_blog_tokenizer",
) -> TokenizerWrapper:
    """
    Train tokenizer from JSON data file

    Args:
        data_path: Path to JSON data file
        vocab_size: Vocabulary size
        output_dir: Output directory for tokenizer
        model_prefix: Model file prefix

    Returns:
        Trained TokenizerWrapper
    """
    import json

    # Load training data
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract texts
    texts = [item["text"] for item in data if "text" in item]

    print(f"Training tokenizer on {len(texts):,} samples...")

    return create_sentencepiece_tokenizer(
        vocab_size=vocab_size,
        model_prefix=model_prefix,
        model_dir=output_dir,
        train_data=texts,
    )


def analyze_tokenizer(
    tokenizer: TokenizerWrapper,
    sample_texts: List[str],
):
    """
    Analyze tokenizer performance on sample texts

    Args:
        tokenizer: TokenizerWrapper instance
        sample_texts: Sample texts to analyze
    """
    print(f"\n📊 Tokenizer Analysis:")
    print(f"  Vocabulary Size: {tokenizer.vocab_size:,}")

    token_counts = []
    for text in sample_texts:
        tokens = tokenizer(text, return_tensors=None)
        token_count = len(tokens["input_ids"])
        token_counts.append(token_count)

    import numpy as np
    print(f"\n  Token Count Statistics (on {len(sample_texts)} samples):")
    print(f"    Mean: {np.mean(token_counts):.1f} tokens")
    print(f"    Median: {np.median(token_counts):.1f} tokens")
    print(f"    Min: {np.min(token_counts)} tokens")
    print(f"    Max: {np.max(token_counts)} tokens")

    # Show sample tokenization
    if sample_texts:
        print(f"\n  Sample Tokenization:")
        sample = sample_texts[0][:100]  # First 100 chars
        tokens = tokenizer(sample, return_tensors=None)
        decoded = tokenizer.decode(tokens["input_ids"])

        print(f"    Original: {sample}")
        print(f"    Tokens: {tokens['input_ids'][:20]}...")  # First 20 tokens
        print(f"    Decoded: {decoded[:100]}")


def get_recommended_tokenizer(use_latest: bool = True) -> str:
    """
    Get recommended tokenizer type based on availability

    Args:
        use_latest: Whether to use latest Korean models (Solar/Exaone)

    Returns:
        Recommended tokenizer type string
    """
    if use_latest:
        # Try modern Korean tokenizers first
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained("upstage/solar-pro-preview-instruct")
            return "solar"
        except:
            pass

        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")
            return "exaone"
        except:
            pass

    # Fallback to reliable options
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("monologg/kobert")
        return "kobert"
    except:
        pass

    # Final fallback
    return "sentencepiece"
