#!/usr/bin/env python3
"""
Inspect CoreML model to find input/output names
"""

import coremltools as ct
from pathlib import Path

# Load CoreML model
model_path = Path(__file__).parent.parent / "models" / "coreml" / "BlogAIDetector.mlpackage"
model = ct.models.MLModel(str(model_path))

print("=" * 70)
print("CoreML Model Inspection")
print("=" * 70)

# Get model spec
spec = model.get_spec()

print("\n📥 Input Information:")
for inp in spec.description.input:
    print(f"  Name: {inp.name}")
    print(f"  Type: {inp.type}")
    print(f"  Shape: {inp.type.multiArrayType.shape}")
    print()

print("📤 Output Information:")
for out in spec.description.output:
    print(f"  Name: {out.name}")
    print(f"  Type: {out.type}")
    print(f"  Shape: {out.type.multiArrayType.shape}")
    print()

print("📝 Usage in Swift:")
print(f"  let input = BlogAIDetectorInput(")
for i, inp in enumerate(spec.description.input):
    print(f"      {inp.name}: ...,")
print(f"  )")
print(f"  let output = model.prediction(input: input)")

output_name = spec.description.output[0].name
print(f"  let logits = output.{output_name}  // [HUMAN_score, AI_score]")
