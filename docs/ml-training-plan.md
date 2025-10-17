# 네이버 블로그 AI 탐지 - ML 학습 가이드

> **목표**: TF-IDF + Logistic Regression 기반 경량 AI 탐지 모델 개발

---

## 🎯 핵심 요약

### 데이터 형식
```json
[
  {
    "text": "제목 + 본문",
    "label": "AI" or "HUMAN"
  }
]
```

### 학습 파이프라인
```
데이터 로드 → Train/Test 분할 → TF-IDF + LR 학습 → CoreML 변환
```

**개발 시간**: 1일
**코드**: ~50줄
**예상 성능**: 75-80% 정확도, ~1MB 모델

---

## 📁 프로젝트 구조

```
Naver-Blog-AI-Detector-ML/
├── data/
│   └── processed/
│       └── training_data.json       # 5,000개 스니펫
│
├── scripts/
│   └── train.py                     # 학습 스크립트
│
└── models/
    └── BlogAIDetector.mlmodel       # CoreML 모델
```

---

## 🚀 학습 스크립트

### `scripts/train.py` (전체 코드)

```python
"""
블로그 AI 탐지 모델 학습
- TF-IDF + Logistic Regression
- CoreML 변환
- 실행: python scripts/train.py
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import coremltools as ct

# 재현성
np.random.seed(42)

print("=" * 50)
print("블로그 AI 탐지 모델 학습")
print("=" * 50)

# ========== 1. 데이터 로드 ==========
print("\n[1/5] 데이터 로드 중...")
with open('data/processed/training_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 텍스트와 라벨 추출
texts = [item['text'] for item in data]
labels = [1 if item['label'] == 'AI' else 0 for item in data]

print(f"  총 샘플: {len(texts)}")
print(f"  AI: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
print(f"  HUMAN: {len(labels) - sum(labels)} ({(1-sum(labels)/len(labels))*100:.1f}%)")

# ========== 2. Train/Test 분할 ==========
print("\n[2/5] Train/Test 분할 중...")
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.2,        # 80% Train, 20% Test
    stratify=labels,      # 클래스 비율 유지
    random_state=42
)

print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ========== 3. 파이프라인 구성 및 학습 ==========
print("\n[3/5] 모델 학습 중...")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,   # 상위 10,000개 단어
        ngram_range=(1, 2),   # unigram + bigram
        min_df=2,             # 최소 2개 문서에 등장
        max_df=0.9            # 90% 이상 문서면 제외
    )),
    ('clf', LogisticRegression(
        solver='liblinear',
        C=1.0,
        class_weight='balanced',  # 클래스 불균형 처리
        max_iter=1000,
        random_state=42
    ))
])

# 학습
pipeline.fit(X_train, y_train)
print("  ✓ 학습 완료")

# ========== 4. 평가 ==========
print("\n[4/5] 모델 평가 중...")

# 예측
y_pred = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)

print(f"\n정확도: {accuracy:.1%}")
print("\n분류 리포트:")
print(classification_report(
    y_test, y_pred,
    target_names=['HUMAN', 'AI'],
    digits=3
))

# ========== 5. CoreML 변환 ==========
print("[5/5] CoreML 변환 중...")

coreml_model = ct.converters.sklearn.convert(
    pipeline,
    input_features=[ct.models.datatypes.String(name="text")],
    output_feature_names=["ai_probability", "prediction"]
)

# 메타데이터
coreml_model.author = "Naver Blog AI Detector"
coreml_model.short_description = "네이버 블로그 AI 글 탐지 (TF-IDF + LR)"
coreml_model.version = "1.0.0"

# 설명 추가
coreml_model.input_description["text"] = "블로그 제목 + 본문 (100-300자)"
coreml_model.output_description["ai_probability"] = "AI 작성 확률 (0.0-1.0)"
coreml_model.output_description["prediction"] = "'AI' 또는 'HUMAN'"

# 저장
output_path = "models/BlogAIDetector.mlmodel"
coreml_model.save(output_path)

print(f"  ✓ 저장 완료: {output_path}")

# 모델 크기 확인
import os
size_kb = os.path.getsize(output_path) / 1024
print(f"  모델 크기: {size_kb:.1f} KB")

print("\n" + "=" * 50)
print("✅ 학습 완료!")
print("=" * 50)
print(f"\n다음 단계:")
print(f"1. Swift 프로젝트에 {output_path} 추가")
print(f"2. Safari 확장 통합")
print(f"3. 실제 테스트")
```

---

## 🏃 실행 방법

### 1. 환경 설정

```bash
# Python 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install scikit-learn coremltools numpy
```

### 2. 학습 실행

```bash
python scripts/train.py
```

### 3. 예상 출력

```
==================================================
블로그 AI 탐지 모델 학습
==================================================

[1/5] 데이터 로드 중...
  총 샘플: 5000
  AI: 2500 (50.0%)
  HUMAN: 2500 (50.0%)

[2/5] Train/Test 분할 중...
  Train: 4000 | Test: 1000

[3/5] 모델 학습 중...
  ✓ 학습 완료

[4/5] 모델 평가 중...

정확도: 78.5%

분류 리포트:
              precision    recall  f1-score   support

       HUMAN      0.780     0.792     0.786       500
          AI      0.790     0.778     0.784       500

    accuracy                          0.785      1000

[5/5] CoreML 변환 중...
  ✓ 저장 완료: models/BlogAIDetector.mlmodel
  모델 크기: 1024.3 KB

==================================================
✅ 학습 완료!
==================================================

다음 단계:
1. Swift 프로젝트에 models/BlogAIDetector.mlmodel 추가
2. Safari 확장 통합
3. 실제 테스트
```

---

## 📊 예상 성능

| 메트릭 | 예상 값 |
|--------|---------|
| 정확도 | 75-80% |
| 모델 크기 | ~1MB |
| 추론 속도 | 10-30ms |
| 학습 시간 | 5-10분 |

---

## 🔧 선택적 개선

### 개선 1: 한국어 토크나이저 (+2-3%p)

```python
# KoNLPy 설치 필요: pip install konlpy
from konlpy.tag import Okt

class KoreanTokenizer:
    def __init__(self):
        self.okt = Okt()

    def __call__(self, text):
        return self.okt.morphs(text)

# TfidfVectorizer에 적용
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=10000,
        tokenizer=KoreanTokenizer(),  # 추가
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )),
    ('clf', LogisticRegression(...))
])

# 예상: 78% → 80-82%
```

**주의**: KoNLPy 설치 복잡, 추론 속도 느려짐 (30ms → 100ms)

---

### 개선 2: 하이퍼파라미터 튜닝 (+1-2%p)

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'tfidf__max_features': [5000, 10000],
    'tfidf__ngram_range': [(1, 2), (1, 3)],
    'clf__C': [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=5, scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
pipeline = grid_search.best_estimator_

# 시간: 1-2시간
# 예상: 78% → 80%
```

---

### 개선 3: Float16 양자화 (크기 50% 감소)

```python
# CoreML 변환 시 추가
coreml_model = ct.convert(
    pipeline,
    compute_precision=ct.precision.FLOAT16
)

# 결과: 1MB → 0.5MB
# 정확도 손실: <0.5%
```

---

## 🎯 개발 전략

### 시나리오 1: 빠른 MVP

```
Day 1:
  오전: train.py 실행
  오후: Safari 확장 통합
  결과: 75-80% 정확도

Day 2:
  실제 테스트
  사용자 피드백 수집

Day 3+:
  정확도 부족 시 → 개선 적용
  충분하면 → 배포
```

---

### 시나리오 2: 높은 정확도 추구

```
Day 1-2:
  기본 버전 (75-80%)

Day 3-4:
  개선 1 (한국어 토크나이저) → 80-82%

Day 5:
  개선 2 (Grid Search) → 80-82%
  개선 3 (Float16) → 크기 감소

결과: 80-82% 정확도, <1MB
```

---

## ❓ FAQ

### Q1: 정말 이것만으로 작동하나요?
**A**: 네. scikit-learn과 CoreML이 복잡한 처리를 담당합니다.

### Q2: 한국어 특화가 필요한가요?
**A**: 선택사항입니다. 기본 TF-IDF도 한국어를 처리합니다.

### Q3: 75-80% 정확도로 충분한가요?
**A**: 실제 환경 테스트 후 판단하세요. MVP로는 충분할 수 있습니다.

### Q4: CoreML 변환이 실패하면?
**A**: scikit-learn과 coremltools 최신 버전 사용을 권장합니다.

---

## 📋 체크리스트

### MVP 개발 (1일)
- [ ] `train.py` 작성
- [ ] 데이터 로드 (`training_data.json`)
- [ ] 학습 실행 (5-10분)
- [ ] CoreML 변환
- [ ] 정확도 확인 (>75%)

### 선택적 개선
- [ ] 한국어 토크나이저 추가 (+2-3%p)
- [ ] Grid Search 튜닝 (+1-2%p)
- [ ] Float16 양자화 (크기 50% 감소)

### Safari 확장 통합
- [ ] `.mlmodel` 파일을 Xcode 프로젝트에 추가
- [ ] Swift에서 모델 로드
- [ ] JavaScript → Swift → CoreML 파이프라인
- [ ] 네이버 검색에서 실제 테스트

---

## 📞 다음 단계

### 1. 데이터 준비 완료 시
```bash
python scripts/train.py
```

### 2. 학습 완료 후
- Safari 확장에 `.mlmodel` 추가
- 실제 환경 테스트

### 3. 성능 개선 필요 시
- 개선 1, 2, 3 순차 적용
- 고급 모델(BERT) 고려 (v2.0)

---

**문서 버전**: 1.0
**작성일**: 2025-01-16
**목표**: 빠른 MVP 개발 및 점진적 개선
