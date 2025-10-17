#!/usr/bin/env python3
"""
Generate synthetic training data using Google Gemini Flash

Strategy:
1. Paraphrasing: Generate 5 variations per sample
2. Style Transfer: AI ↔ HUMAN style conversion
3. Augmentation: Length, tone, word choice variations
4. Zero-shot Generation: Create new blog posts

Target: 10,000 synthetic samples from 297 real samples
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv


# Load environment variables from ML-Training/.env
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent  # ML-Training/scripts/
PROJECT_ROOT = SCRIPT_DIR.parent  # ML-Training/
ENV_PATH = PROJECT_ROOT / ".env"

# Load .env file with override=True to ignore existing env vars
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Debug: Print if .env was loaded
if ENV_PATH.exists():
    print(f"✓ Loaded .env from {ENV_PATH}")
else:
    print(f"⚠ Warning: .env file not found at {ENV_PATH}")


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation"""
    real_data_path: str
    output_dir: str
    target_size: int = 10000
    gemini_model: str = "gemini-flash-latest"
    variations_per_sample: int = 5
    temperature: float = 0.9
    max_retries: int = 3
    batch_size: int = 10
    max_workers: int = 40  # Parallel API calls


class GeminiSyntheticGenerator:
    """Generate synthetic data using Gemini Flash"""

    def __init__(self, config: SyntheticConfig):
        self.config = config

        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment.\n"
                f"Please set it in {ENV_PATH} file or export it.\n"
                f"Example: GEMINI_API_KEY=your_api_key_here"
            )

        # Debug: Print API key info (first/last 4 chars only)
        if len(api_key) > 8:
            masked_key = f"{api_key[:4]}...{api_key[-4:]}"
            print(f"✓ Found API key: {masked_key}")
        else:
            print(f"⚠ Warning: API key seems too short ({len(api_key)} chars)")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(config.gemini_model)

        print(f"✓ Initialized Gemini model: {config.gemini_model}")

    def load_real_data(self) -> List[Dict]:
        """Load real training data"""
        with open(self.config.real_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"✓ Loaded {len(data)} real samples")
        return data

    def generate_paraphrases(
        self,
        text: str,
        label: str,
        num_variations: int = 5,
    ) -> List[Dict]:
        """
        Generate paraphrased variations of text

        Args:
            text: Original text
            label: AI or HUMAN
            num_variations: Number of variations to generate

        Returns:
            List of synthetic samples
        """
        # Style-specific guidelines based on labeling criteria
        if label == "AI":
            style_guide = """**AI 스타일 특성 유지**:
1. 서사의 진정성: 전형적이고 그럴듯한 이야기 구조
2. 디테일: 일반적이고 상투적인 표현 ("맛있었다", "좋았다")
3. 비판의 균형: 형식적이고 가벼운 단점 언급
4. 언어: "환상적인", "기대를 뛰어넘는" 등 리뷰 상투어 사용
5. 어조: 일관되게 정중하고 긍정적, 감정 변화 최소화
6. 구조: 균일한 문단 길이, 키워드 반복, 정형화된 흐름
7. 목적: 정보 전달 중심, 상업적 의도 내포"""
        else:
            style_guide = """**HUMAN 스타일 특성 유지**:
1. 서사의 진정성: 예측 불가능하고 사소한 디테일이 살아있는 실제 경험
2. 디테일: 오감을 자극하는 구체적이고 감각적인 묘사
3. 비판의 균형: 실제 불편함에서 나온 진솔한 피드백
4. 언어: 개인만의 독창적이고 신선한 표현
5. 어조: 구어체 사용 ("그치만", "근데", "진짜"), 감정 변화 자연스러움
6. 구조: 불규칙적 문단, 자연스러운 흐름, 개인 사고의 전개
7. 목적: 순수한 경험 공유, 개인적 감정 표현"""

        prompt = f"""당신은 '디지털 텍스트 법의학자'로서 블로그 글의 AI/HUMAN 특성을 정확히 이해하고 있습니다.

원본 글:
{text}

원본 라벨: {label}

**임무**: 위 원본 글의 의미와 {label} 스타일 특성을 정확히 유지하면서 {num_variations}개의 자연스러운 변형을 생성하세요.

{style_guide}

**규칙**:
1. 원본의 핵심 의미와 {label} 스타일 특성 완벽히 유지
2. 단어 선택, 문장 구조를 다양하게 변경
3. 길이는 원본의 80-120% 범위
4. 위 스타일 가이드의 모든 특성 반영
5. 자연스럽고 진짜처럼 느껴지는 한국어 작성

**출력 형식** (JSON):
{{
  "variations": [
    "변형 1 텍스트",
    "변형 2 텍스트",
    ...
  ]
}}

JSON만 출력하세요. 다른 설명은 불필요합니다."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": 8192,
                }
            )

            # Parse JSON response
            response_text = response.text.strip()

            # Remove markdown code blocks more robustly
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]

            if response_text.endswith("```"):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            # Validate response is not empty
            if not response_text:
                print(f"⚠ Empty response from API")
                return []

            # Try to parse JSON with better error handling
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                print(f"⚠ JSON decode error: {json_err}")
                print(f"⚠ Response preview (first 200 chars): {response_text[:200]}")
                return []

            variations = result.get("variations", [])

            if not variations:
                print(f"⚠ No variations in response")
                return []

            # Create synthetic samples
            samples = []
            for var_text in variations[:num_variations]:
                if var_text and var_text.strip():
                    samples.append({
                        "text": var_text,
                        "label": label,
                        "source": "synthetic_paraphrase",
                        "metadata": {
                            "original_text": text[:100] + "...",
                            "generation_method": "paraphrase",
                        }
                    })

            return samples

        except Exception as e:
            print(f"⚠ Error generating paraphrases: {type(e).__name__}: {e}")
            return []

    def generate_style_transfer(
        self,
        text: str,
        source_label: str,
        target_label: str,
    ) -> Optional[Dict]:
        """
        Transfer style from AI to HUMAN or vice versa

        Args:
            text: Original text
            source_label: Original label (AI or HUMAN)
            target_label: Target label (AI or HUMAN)

        Returns:
            Synthetic sample with transferred style
        """
        if source_label == "AI":
            style_instruction = """**AI → HUMAN 변환 (7가지 핵심 기준)**:

1. **서사의 진정성**: 전형적 이야기 → 예측 불가능한 실제 경험
   - Before: "매장에 방문했고 음식을 주문했습니다"
   - After: "우연히 지나가다가 사람 많길래 들어갔는데"

2. **디테일의 구체성**: 일반적 묘사 → 오감 자극 구체적 묘사
   - Before: "음식이 맛있었습니다"
   - After: "갓 구운 빵에서 버터 향이 확 올라오더라고요"

3. **비판의 균형**: 형식적 단점 → 진솔한 불만
   - Before: "다만 가격이 조금 비싼 편입니다"
   - After: "솔직히 이 가격에 이 양이면 좀 아쉬웠어요"

4. **언어의 독창성**: 리뷰 상투어 → 개인적 표현
   - Before: "환상적인 경험", "기대를 뛰어넘는"
   - After: "완전 대박", "엄청 좋더라고"

5. **어조와 감정**: 일관된 정중함 → 자연스러운 감정 변화
   - 구어체 추가: "그치만", "근데", "진짜", "ㅋㅋ"
   - 종결어미: "~입니다" → "~네요", "~요", "~더라"

6. **구조의 자연스러움**: 균일한 문단 → 불규칙적 흐름
   - 키워드 반복 제거
   - 문단 길이 불규칙하게
   - 개인 사고 흐름 반영

7. **궁극적 목적**: 정보 전달 → 경험 공유
   - 상업적 느낌 제거
   - 개인 경험과 감정 중심으로"""
        else:
            style_instruction = """**HUMAN → AI 변환 (7가지 핵심 기준)**:

1. **서사의 진정성**: 실제 경험 → 전형적이고 그럴듯한 구조
   - Before: "우연히 지나가다가 사람 많길래 들어갔는데"
   - After: "매장을 방문하여 음식을 주문했습니다"

2. **디테일의 구체성**: 감각적 묘사 → 일반적 표현
   - Before: "갓 구운 빵에서 버터 향이 확 올라오더라고요"
   - After: "음식이 맛있었습니다"

3. **비판의 균형**: 진솔한 불만 → 형식적 단점 언급
   - Before: "솔직히 이 가격에 이 양이면 좀 아쉬웠어요"
   - After: "다만 가격이 조금 비싼 편입니다"

4. **언어의 독창성**: 개인적 표현 → 리뷰 상투어
   - Before: "완전 대박", "엄청 좋더라고"
   - After: "환상적인 경험", "기대를 뛰어넘는"

5. **어조와 감정**: 자연스러운 변화 → 일관된 정중함
   - 구어체 제거: "그치만", "근데", "진짜" → 문어체
   - 종결어미: "~네요", "~요" → "~입니다", "~습니다"

6. **구조의 인위성**: 자연스러운 흐름 → 균일하고 정형화
   - 키워드 적절히 반복
   - 문단 길이 균일하게
   - 논리적 구조 강화

7. **궁극적 목적**: 경험 공유 → 정보 전달
   - 객관적 정보 중심
   - 상업적 의도 내포 가능"""

        prompt = f"""당신은 '디지털 텍스트 법의학자'로서 AI와 HUMAN 글의 차이를 정확히 알고 있습니다.

원본 글 ({source_label} 스타일):
{text}

**임무**: 위 글을 {target_label} 스타일로 완벽히 변환하세요.

{style_instruction}

**중요**: 위 7가지 기준을 모두 적용하여, {target_label} 글의 특성을 완벽히 재현하세요.

**출력 형식** (JSON):
{{
  "converted_text": "변환된 텍스트"
}}

JSON만 출력하세요."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": 8192,
                }
            )

            response_text = response.text.strip()

            # Remove markdown code blocks more robustly
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]

            if response_text.endswith("```"):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            # Validate response is not empty
            if not response_text:
                print(f"⚠ Empty response from API")
                return None

            # Try to parse JSON with better error handling
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                # Log the problematic response for debugging
                print(f"⚠ JSON decode error: {json_err}")
                print(f"⚠ Response preview (first 200 chars): {response_text[:200]}")
                return None

            converted_text = result.get("converted_text", "")

            if converted_text and converted_text.strip():
                return {
                    "text": converted_text,
                    "label": target_label,
                    "source": "synthetic_style_transfer",
                    "metadata": {
                        "original_label": source_label,
                        "target_label": target_label,
                        "generation_method": "style_transfer",
                    }
                }
            else:
                print(f"⚠ Empty converted_text in response")
                return None

        except Exception as e:
            print(f"⚠ Error in style transfer: {type(e).__name__}: {e}")

        return None

    def generate_zero_shot(
        self,
        label: str,
        num_samples: int = 1,
    ) -> List[Dict]:
        """
        Generate completely new blog posts

        Args:
            label: AI or HUMAN
            num_samples: Number of samples to generate

        Returns:
            List of new synthetic samples
        """
        if label == "AI":
            style_guide = """**AI 스타일 가이드 (7가지 특징)**:

1. **서사의 진정성**: 전형적이고 그럴듯한 이야기
   - "매장을 방문하여", "상품을 구매했고", "서비스를 이용했습니다"
   - 예측 가능한 전개, 만들어진 느낌

2. **디테일의 구체성**: 일반적이고 상투적인 표현
   - "맛있었습니다", "좋았습니다", "추천드립니다"
   - 구체적 감각 묘사 최소화

3. **비판의 균형**: 형식적이고 가벼운 단점 언급
   - "다만 ~한 점이 아쉽습니다", "조금 ~한 편입니다"

4. **언어의 독창성**: 리뷰 상투어 사용
   - "환상적인", "기대를 뛰어넘는", "완벽한", "최고의"
   - 온라인 리뷰 전형적 표현

5. **어조와 감정**: 일관되게 정중하고 긍정적
   - "~입니다", "~습니다", "~였습니다"
   - 감정 변화 최소, 균일한 톤

6. **구조의 인위성**: 균일하고 정형화된 구조
   - 키워드 반복 (SEO 의도)
   - 균일한 문단 길이
   - 서론-본론-결론 명확

7. **궁극적 목적**: 정보 전달 및 상업적 의도
   - 객관적 정보 나열
   - 링크 유도, 광고성 느낌"""
        else:
            style_guide = """**HUMAN 스타일 가이드 (7가지 특징)**:

1. **서사의 진정성**: 예측 불가능한 실제 경험
   - "우연히 지나가다가", "친구 추천으로", "원래는 안 갈 뻔 했는데"
   - 사소한 디테일, 진짜 경험 느낌

2. **디테일의 구체성**: 오감 자극 감각적 묘사
   - "고소한 버터 향이 확 올라오더라고요"
   - "겉은 바삭한데 속은 촉촉했어요"
   - 구체적이고 생생한 감각

3. **비판의 균형**: 진솔한 불만과 피드백
   - "솔직히 별로였어요", "이 가격에 이건 좀..."
   - "처음엔 좋았는데 나중엔 실망"

4. **언어의 독창성**: 개인만의 표현
   - "완전 대박", "진짜 개쩔었어", "엄청 좋더라고"
   - 독특하고 신선한 표현

5. **어조와 감정**: 자연스러운 구어체와 감정 변화
   - "그치만", "근데", "진짜", "ㅋㅋ", "ㅠㅠ"
   - "~네요", "~요", "~더라", "~던데"
   - 설렘, 실망, 만족 등 감정 변화

6. **구조의 자연스러움**: 불규칙적이고 자유로운 흐름
   - 문단 길이 불규칙
   - 키워드 반복 없음
   - 생각의 흐름대로 전개

7. **궁극적 목적**: 순수한 경험 공유
   - 개인 경험과 감정 중심
   - 상업적 의도 없음
   - 진솔한 공유"""

        topics = [
            "맛집 후기", "여행 경험", "제품 리뷰", "일상 이야기",
            "취미 활동", "건강 정보", "요리 레시피", "영화/드라마 감상평",
            "독서 후기", "쇼핑 경험", "운동 일지", "IT 제품 사용기",
        ]

        topic = random.choice(topics)

        prompt = f"""당신은 '디지털 텍스트 법의학자'로서 AI와 HUMAN 글의 차이를 정확히 이해하고 있습니다.

**주제**: {topic}
**스타일**: {label}

{style_guide}

**임무**: 위 주제로 100-300자 길이의 블로그 스니펫 {num_samples}개를 {label} 스타일로 완벽히 생성하세요.

**중요**: 위 7가지 특징을 모두 반영하여, 진짜 {label} 글처럼 느껴지도록 작성하세요.

**출력 형식** (JSON):
{{
  "posts": [
    "블로그 글 1",
    "블로그 글 2",
    ...
  ]
}}

JSON만 출력하세요."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": 8192,
                }
            )

            response_text = response.text.strip()

            # Remove markdown code blocks more robustly
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]

            if response_text.endswith("```"):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            # Validate response is not empty
            if not response_text:
                print(f"⚠ Empty response from API")
                return []

            # Try to parse JSON with better error handling
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                print(f"⚠ JSON decode error: {json_err}")
                print(f"⚠ Response preview (first 200 chars): {response_text[:200]}")
                return []

            posts = result.get("posts", [])

            if not posts:
                print(f"⚠ No posts in response")
                return []

            samples = []
            for post_text in posts[:num_samples]:
                if post_text and post_text.strip():
                    samples.append({
                        "text": post_text,
                        "label": label,
                        "source": "synthetic_zero_shot",
                        "metadata": {
                            "topic": topic,
                            "generation_method": "zero_shot",
                        }
                    })

            return samples

        except Exception as e:
            print(f"⚠ Error generating zero-shot samples: {type(e).__name__}: {e}")
            return []

    def _process_paraphrase_sample(self, sample: Dict) -> List[Dict]:
        """Process a single sample for paraphrasing (for parallel execution)"""
        # Combine title and snippet_text
        title = sample.get("title", "")
        snippet_text = sample.get("snippet_text", "")
        text = f"{title}\n\n{snippet_text}" if title and snippet_text else (title or snippet_text)
        label = sample.get("label", "HUMAN")

        if not text.strip():
            return []

        variations = self.generate_paraphrases(
            text,
            label,
            self.config.variations_per_sample
        )
        return variations

    def _process_style_transfer_sample(self, sample: Dict) -> Optional[Dict]:
        """Process a single sample for style transfer (for parallel execution)"""
        # Combine title and snippet_text
        title = sample.get("title", "")
        snippet_text = sample.get("snippet_text", "")
        text = f"{title}\n\n{snippet_text}" if title and snippet_text else (title or snippet_text)
        label = sample.get("label", "HUMAN")
        target_label = "HUMAN" if label == "AI" else "AI"

        if not text.strip():
            return None

        return self.generate_style_transfer(text, label, target_label)

    def _save_checkpoint(self, data: List[Dict], filename: str):
        """Save checkpoint data to prevent data loss"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / filename
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"  💾 Checkpoint saved: {checkpoint_path}")

    def generate_all_synthetic_data(self) -> List[Dict]:
        """
        Generate all synthetic data using multiple strategies with parallel processing

        Returns:
            List of all synthetic samples
        """
        # Load real data
        real_data = self.load_real_data()
        all_synthetic = []

        # Strategy 1: Paraphrasing (main strategy) - PARALLEL
        print("\n📝 Strategy 1: Paraphrasing (Parallel)...")
        print(f"  Generating {self.config.variations_per_sample} variations per sample")
        print(f"  Using {self.config.max_workers} concurrent workers")

        paraphrase_results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._process_paraphrase_sample, sample): sample
                for sample in real_data
            }

            # Process completed tasks with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Paraphrasing"):
                try:
                    variations = future.result()
                    if variations:
                        paraphrase_results.extend(variations)
                except Exception as e:
                    sample = futures[future]
                    print(f"\n⚠ Error processing sample: {e}")

        all_synthetic.extend(paraphrase_results)
        print(f"  ✓ Generated {len(paraphrase_results)} paraphrased samples")

        # Save Strategy 1 checkpoint
        self._save_checkpoint(paraphrase_results, "checkpoint_1_paraphrase.json")

        # Strategy 2: Style Transfer (10% of samples) - PARALLEL
        print("\n🎨 Strategy 2: Style Transfer (Parallel)...")
        print(f"  Using {self.config.max_workers} concurrent workers")
        style_transfer_count = len(real_data) // 10
        style_samples = random.sample(real_data, min(style_transfer_count, len(real_data)))

        style_results = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self._process_style_transfer_sample, sample): sample
                for sample in style_samples
            }

            # Process completed tasks with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Style Transfer"):
                try:
                    result = future.result()
                    if result:
                        style_results.append(result)
                except Exception as e:
                    sample = futures[future]
                    print(f"\n⚠ Error processing sample: {e}")

        all_synthetic.extend(style_results)
        print(f"  ✓ Generated {len(style_results)} style-transferred samples")

        # Save Strategy 2 checkpoint
        self._save_checkpoint(all_synthetic, "checkpoint_2_after_style_transfer.json")

        # Strategy 3: Zero-shot Generation (fill remaining) - PARALLEL
        current_count = len(all_synthetic)
        remaining = self.config.target_size - current_count

        if remaining > 0:
            print(f"\n✨ Strategy 3: Zero-shot Generation (Parallel)...")
            print(f"  Generating {remaining} new samples")
            print(f"  Using {self.config.max_workers} concurrent workers")

            # Create tasks: each generates 5 samples
            num_tasks = (remaining + 4) // 5
            tasks = []
            for _ in range(num_tasks):
                # Alternate between AI and HUMAN
                label = "AI" if random.random() < 0.5 else "HUMAN"
                tasks.append((label, 5))

            zero_shot_results = []
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(self.generate_zero_shot, label, num_samples): (label, num_samples)
                    for label, num_samples in tasks
                }

                # Process completed tasks with progress bar
                for future in tqdm(as_completed(futures), total=len(futures), desc="Zero-shot"):
                    try:
                        samples = future.result()
                        if samples:
                            zero_shot_results.extend(samples)
                    except Exception as e:
                        print(f"\n⚠ Error generating zero-shot: {e}")

            all_synthetic.extend(zero_shot_results)
            print(f"  ✓ Generated {len(zero_shot_results)} zero-shot samples")

            # Save Strategy 3 checkpoint
            self._save_checkpoint(all_synthetic, "checkpoint_3_after_zero_shot.json")

        # Trim to target size
        all_synthetic = all_synthetic[:self.config.target_size]

        print(f"\n✅ Total synthetic samples: {len(all_synthetic)}")

        # Save final checkpoint before trimming
        self._save_checkpoint(all_synthetic, "checkpoint_final_trimmed.json")

        return all_synthetic

    def save_synthetic_data(self, synthetic_data: List[Dict]):
        """Save synthetic data to output directory"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save all synthetic data
        output_path = output_dir / "all_synthetic.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(synthetic_data, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Saved synthetic data to {output_path}")

        # Save by generation method
        methods = {}
        for sample in synthetic_data:
            method = sample.get("source", "unknown")
            if method not in methods:
                methods[method] = []
            methods[method].append(sample)

        for method, samples in methods.items():
            method_path = output_dir / f"{method}.json"
            with open(method_path, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            print(f"  ✓ Saved {len(samples)} samples to {method_path.name}")

        # Statistics
        print(f"\n📊 Synthetic Data Statistics:")
        print(f"  Total: {len(synthetic_data):,}")
        ai_count = sum(1 for s in synthetic_data if s["label"] == "AI")
        human_count = len(synthetic_data) - ai_count
        print(f"  AI: {ai_count:,} ({ai_count/len(synthetic_data)*100:.1f}%)")
        print(f"  HUMAN: {human_count:,} ({human_count/len(synthetic_data)*100:.1f}%)")


def main():
    # Define default output path as absolute path
    DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "data" / "synthetic")

    parser = argparse.ArgumentParser(
        description="Generate synthetic training data using Gemini Flash"
    )
    parser.add_argument(
        "--real_data",
        type=str,
        default="/Users/taesooa/Desktop/Swift/Naver-Blog-AI-Detector/Naver-Blog-AI-Detector/Naver-Blog-AI-Detector-ML/Data-Preprocessing/data/processed/training_data.json",
        help="Path to real training data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for synthetic data"
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=None,
        help="Target number of synthetic samples (default: 3x real data size)"
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=3.0,
        help="Multiplier for real data size (default: 3.0)"
    )
    parser.add_argument(
        "--variations",
        type=int,
        default=5,
        help="Number of variations per sample"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Gemini generation temperature"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-flash-latest",
        help="Gemini model to use"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of concurrent workers for parallel API calls (default: 20)"
    )

    args = parser.parse_args()

    # Load real data to calculate target size
    import json
    with open(args.real_data, 'r', encoding='utf-8') as f:
        real_data_temp = json.load(f)

    real_data_count = len(real_data_temp)

    # Calculate target size: 3x real data size if not specified
    if args.target_size is None:
        target_size = int(real_data_count * args.multiplier)
        print(f"\n📊 Auto-calculating target size:")
        print(f"  Real data: {real_data_count:,} samples")
        print(f"  Multiplier: {args.multiplier}x")
        print(f"  Target synthetic: {target_size:,} samples")
    else:
        target_size = args.target_size
        print(f"\n📊 Using specified target size: {target_size:,} samples")

    # Create config
    config = SyntheticConfig(
        real_data_path=args.real_data,
        output_dir=args.output,
        target_size=target_size,
        gemini_model=args.model,
        variations_per_sample=args.variations,
        temperature=args.temperature,
        max_workers=args.workers,
    )

    print("\n" + "=" * 70)
    print("🤖 Gemini Flash Synthetic Data Generator (Parallel)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Real Data: {config.real_data_path}")
    print(f"  Real Data Count: {real_data_count:,}")
    print(f"  Output: {config.output_dir}")
    print(f"  Target Size: {config.target_size:,} ({args.multiplier}x real data)")
    print(f"  Variations/Sample: {config.variations_per_sample}")
    print(f"  Model: {config.gemini_model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Parallel Workers: {config.max_workers}")

    # Generate synthetic data
    generator = GeminiSyntheticGenerator(config)
    synthetic_data = generator.generate_all_synthetic_data()

    # Save results
    generator.save_synthetic_data(synthetic_data)

    print("\n✅ Synthetic data generation complete!")


if __name__ == "__main__":
    main()
