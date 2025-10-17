# TinyTransformer ML Training Pipeline

> From Scratch training with synthetic data generation for Naver Blog AI Detection

## Overview

Complete ML training pipeline for building a lightweight on-device AI content detector:

- **Model**: TinyBERT-4 architecture (15M parameters, 4 layers)
- **Data**: ~1,200 samples (297 real + 891 synthetic = 3x real data)
- **Training**: From Scratch (no knowledge distillation)
- **Optimization**: INT4 quantization, pruning, W8A8 mode
- **Target**: 5-10MB CoreML model, <50ms inference

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-v2.txt
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Complete Pipeline

```bash
# Step 1: Generate synthetic data (3x real data = ~891 samples)
# With 20 parallel workers for faster generation
python scripts/1_generate_synthetic.py \
    --output data/synthetic/ \
    --multiplier 3.0 \
    --workers 20  # Parallel API calls (default: 20)

# Step 2: Build model architecture
python scripts/2_build_model.py \
    --config configs/tiny_transformer.yaml \
    --output models/checkpoints/tiny-transformer-init \
    --test

# Step 3: Train from scratch
python scripts/3_train_from_scratch.py \
    --config configs/training_config.yaml \
    --model_config configs/tiny_transformer.yaml \
    --tokenizer_type sentencepiece

# Step 4: Optimize (pruning + quantization)
python scripts/4_optimize.py \
    --model models/checkpoints/final \
    --output models/optimized \
    --prune_amount 0.3 \
    --quantize

# Step 5: Export to CoreML
python scripts/5_export_coreml.py \
    --model models/optimized/pruned \
    --output models/coreml \
    --quantization int4 \
    --enable_w8a8 \
    --test

# Step 6: Evaluate
python scripts/6_evaluate.py \
    --model models/checkpoints/final \
    --test_data data/processed/test.json \
    --output models/evaluation \
    --benchmark
```

## Project Structure

```
ML-Training/
├── configs/
│   ├── tiny_transformer.yaml      # Model architecture config
│   └── training_config.yaml       # Training hyperparameters
│
├── data/
│   ├── real/                      # Original 297 samples
│   ├── synthetic/                 # Generated 10,000 samples
│   └── processed/                 # Train/val/test splits
│       ├── train.json             # 8,237 samples
│       ├── val.json               # 1,030 samples
│       └── test.json              # 1,030 samples
│
├── scripts/
│   ├── 1_generate_synthetic.py    # Gemini Flash data generation
│   ├── 2_build_model.py           # TinyTransformer architecture
│   ├── 3_train_from_scratch.py    # From Scratch training
│   ├── 4_optimize.py              # Quantization & pruning
│   ├── 5_export_coreml.py         # CoreML conversion
│   └── 6_evaluate.py              # Comprehensive evaluation
│
├── models/
│   ├── checkpoints/               # Training checkpoints
│   ├── optimized/                 # Optimized models
│   ├── coreml/                    # CoreML exports
│   └── evaluation/                # Evaluation results
│
├── utils/
│   ├── data_utils.py              # Data processing utilities
│   ├── model_utils.py             # Model management utilities
│   └── tokenizer.py               # Tokenizer wrapper
│
├── requirements-v2.txt            # Python dependencies
└── README.md                      # This file
```

## Detailed Usage

### 1. Synthetic Data Generation

Generate synthetic samples (3x real data = ~891 samples) using Gemini Flash:

```bash
# Default: 3x real data size (auto-calculated, uses default path)
# With parallel processing for faster generation (20 concurrent API calls)
python scripts/1_generate_synthetic.py \
    --output data/synthetic/ \
    --multiplier 3.0 \
    --variations 5 \
    --temperature 0.9 \
    --workers 20

# Or specify exact number
python scripts/1_generate_synthetic.py \
    --output data/synthetic/ \
    --target_size 1000 \
    --variations 5 \
    --workers 20

# Adjust workers for API rate limits (if needed)
python scripts/1_generate_synthetic.py \
    --output data/synthetic/ \
    --multiplier 3.0 \
    --workers 10  # Reduce if hitting rate limits

# Or use custom data path
python scripts/1_generate_synthetic.py \
    --real_data /path/to/your/training_data.json \
    --output data/synthetic/ \
    --multiplier 3.0 \
    --workers 20
```

**Output**:
- `data/synthetic/all_synthetic.json` - All synthetic samples
- `data/synthetic/synthetic_paraphrase.json` - Paraphrased variations
- `data/synthetic/synthetic_style_transfer.json` - Style-transferred samples
- `data/synthetic/synthetic_zero_shot.json` - Newly generated samples

**Performance**:
- **Time**: ~5-10 minutes with 20 parallel workers (vs ~2.5 hours sequential)
- **Speed**: 20x faster with parallel processing
- **Cost**: ~$5-10 for 891 samples (297 real × 3)

### 2. Model Building

Create TinyTransformer architecture:

```bash
python scripts/2_build_model.py \
    --config configs/tiny_transformer.yaml \
    --output models/checkpoints/tiny-transformer-init \
    --test
```

**Model Specs**:
- Parameters: ~15M
- Hidden Size: 312
- **Layers: 4** (TinyBERT-4)
- Attention Heads: 12
- Max Sequence Length: 128

### 3. From Scratch Training

Train on ~1,200 samples (297 real + 891 synthetic, no pretrained weights):

```bash
python scripts/3_train_from_scratch.py \
    --config configs/training_config.yaml \
    --model_config configs/tiny_transformer.yaml \
    --tokenizer_type sentencepiece
```

**Training Config**:
- Epochs: 15
- Batch Size: 32
- Learning Rate: 5e-4
- Warmup Steps: 500
- Early Stopping: 3 epochs patience

**Expected Training Time**: 1-2 hours on GPU (with ~1,200 samples)

### 4. Model Optimization

Apply pruning and quantization:

```bash
# Pruning only
python scripts/4_optimize.py \
    --model models/checkpoints/final \
    --output models/optimized \
    --prune_amount 0.3

# Pruning + INT8 Quantization
python scripts/4_optimize.py \
    --model models/checkpoints/final \
    --output models/optimized \
    --prune_amount 0.3 \
    --quantize
```

**Size Reduction**:
- Original: 60 MB (FP32)
- After Pruning: ~42 MB
- After INT8: ~15 MB

### 5. CoreML Export

Convert to CoreML with INT4 quantization:

```bash
python scripts/5_export_coreml.py \
    --model models/optimized/pruned \
    --output models/coreml \
    --name BlogAIDetector \
    --quantization int4 \
    --enable_w8a8 \
    --format mlpackage \
    --test
```

**CoreML Features**:
- INT4 per-block quantization (CoreML 8.0+)
- W8A8 mode for A17 Pro / M4+ Neural Engine
- Final size: 5-10 MB
- macOS 14+ optimized

### 6. Model Evaluation

Comprehensive evaluation with visualizations:

```bash
# PyTorch model
python scripts/6_evaluate.py \
    --model models/checkpoints/final \
    --test_data data/processed/test.json \
    --output models/evaluation \
    --model_type pytorch \
    --benchmark

# CoreML model
python scripts/6_evaluate.py \
    --model models/coreml/BlogAIDetector.mlpackage \
    --test_data data/processed/test.json \
    --output models/evaluation \
    --model_type coreml \
    --benchmark
```

**Outputs**:
- `evaluation_report.json` - Comprehensive metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curve.png` - ROC curve plot
- `error_examples.json` - Misclassified samples

## Performance Targets

| Metric | Target | Expected |
|--------|--------|----------|
| Model Size | < 10 MB | 5-10 MB |
| Inference Time | < 50 ms | 30-45 ms |
| Accuracy | > 88% | 88-90% |
| F1 Score | > 88% | 88-90% |
| ROC-AUC | > 0.90 | 0.90-0.92 |

## Configuration Files

### Model Config (`configs/tiny_transformer.yaml`)

```yaml
architecture:
  vocab_size: 8000
  hidden_size: 312
  num_hidden_layers: 4
  num_attention_heads: 12
  intermediate_size: 1200
  max_position_embeddings: 128
```

### Training Config (`configs/training_config.yaml`)

```yaml
training:
  num_train_epochs: 15
  per_device_train_batch_size: 32
  learning_rate: 5.0e-4
  warmup_steps: 500

regularization:
  dropout: 0.1
  label_smoothing: 0.1

early_stopping:
  patience: 3
```

## Hardware Requirements

### Training
- **GPU**: NVIDIA RTX 3060+ (12GB VRAM) or Colab Pro+
- **RAM**: 16GB+
- **Storage**: 50GB+

### Inference (Safari Extension)
- **macOS**: 14.0+ (CoreML 8.0, Neural Engine)
- **Processor**: Apple Silicon (M1/M2/M3/M4) recommended
- **Optimal**: A17 Pro / M4+ (W8A8 mode support)

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in training_config.yaml
per_device_train_batch_size: 16  # Instead of 32
```

**2. Gemini API Rate Limit**
```bash
# Add delays in 1_generate_synthetic.py
time.sleep(1.0)  # Increase from 0.5
```

**3. CoreML Conversion Errors**
```bash
# Ensure coremltools 8.0+
pip install --upgrade coremltools

# Check PyTorch version compatibility
pip install torch==2.5.0
```

## Integration with Safari Extension

After CoreML export:

1. **Add Model to Xcode**:
   - Drag `BlogAIDetector.mlpackage` into Xcode project
   - Ensure target includes Safari Extension

2. **Swift Integration**:
```swift
import CoreML

// Load model
let model = try BlogAIDetector(configuration: MLModelConfiguration())

// Prepare input (tokenize title + snippet)
let input = BlogAIDetectorInput(
    input_ids: tokenizedIds,
    attention_mask: attentionMask
)

// Predict
let output = try model.prediction(input: input)
let logits = output.logits

// Get probabilities
let aiProb = exp(logits[1]) / (exp(logits[0]) + exp(logits[1]))
let prediction = aiProb > 0.5 ? "AI" : "HUMAN"
```

## Development Timeline

| Phase | Tasks | Time |
|-------|-------|------|
| 1 | Synthetic data generation | 2 days |
| 2 | Model architecture | 1 day |
| 3 | From Scratch training | 3 days |
| 4 | Optimization | 2 days |
| 5 | CoreML export | 1 day |
| 6 | Evaluation & testing | 1 day |
| **Total** | | **10-13 days** |

## Cost Estimate

- **Gemini Flash API**: $15-25 (10,000 synthetic samples)
- **GPU Training**: $0-50 (Colab free tier or Pro+)
- **Total**: $15-75

## References

### Models & Research
- [TinyBERT (2024-2025)](https://arxiv.org/abs/1910.01108) - Energy efficiency 91% improvement
- [Gemini Flash](https://ai.google.dev/gemini-api/docs/models) - Fast synthetic data generation
- [CoreML 8.0 Optimization](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)

### Korean NLP
- [Solar Pro 2](https://www.upstage.ai/solar-pro) - Upstage, 31B params
- [Exaone 4.0](https://www.lgresearch.ai/exaone) - LG AI Research
- [KoBERT](https://github.com/SKTBrain/KoBERT) - SK T-Brain

### Tools
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [CoreML Tools 8.0+](https://github.com/apple/coremltools)
- [PyTorch 2.5+](https://pytorch.org/)

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
- GitHub Issues: [Create an issue](https://github.com/...)
- Documentation: See `docs/` directory
- Model Plan: See `docs/model-plan-v2.md`

---

**Version**: 2.0.0
**Last Updated**: 2025-10-17
**Training Approach**: From Scratch
**Data Generation**: Gemini Flash
