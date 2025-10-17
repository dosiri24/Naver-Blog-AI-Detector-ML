# Naver Blog AI Detector - ML 프로젝트 계획서

> **목표**: 네이버 블로그 AI 작성 글 탐지 ML 모델 개발 및 Safari 확장 통합

---

## 프로젝트 개요

### 핵심 전략
**전체 글 스크래핑 → LLM 라벨링 → 스니펫 증강 → ML 학습 → CoreML 변환**

- 1,500개 블로그 **전체 글** 수집 (높은 라벨링 품질)
- 각 글에서 3-5개 **스니펫 추출** (5,000개 학습 데이터 구축)
- 스니펫으로 모델 학습 → **Train-Test Distribution 일치**
- **CoreML 온디바이스 추론** → 배포 최적화

### CoreML 아키텍처 선택 이유

**배포 관점에서 CoreML이 최적인 이유**:

1. ✅ **운영 비용 $0** (서버 불필요, 영구 무료)
2. ✅ **빠른 응답** (10-30ms 로컬 추론 vs 서버 100-500ms)
3. ✅ **개인정보 보호** (디바이스 내 처리, 서버 전송 없음)
4. ✅ **오프라인 작동** (네트워크 불필요)
5. ✅ **무한 확장** (사용자 증가해도 비용/성능 동일)
6. ✅ **App Store 우호적** (서버 의존성 없음)

**대안 (FastAPI 서버) 비교**:
- 서버 비용: $20-100/월 (사용자 증가 시 급증)
- 응답 속도: 100-500ms (네트워크 왕복)
- 개인정보: 서버 전송 필요 (법적 리스크)
- 배포 복잡도: 서버 구축, 도메인, HTTPS, 모니터링 등

→ **Safari 확장 특성상 CoreML이 압도적으로 유리**

### 데이터 파이프라인 전략

**문제**: 네이버 검색 결과 미리보기는 키워드 주변 문장만 보여줌
- 동일 블로그 글 → 여러 검색어 → 여러 미리보기 생성
- 실제 사용: Safari 확장은 **미리보기만** 볼 수 있음

**해결**:
1. **전체 글로 라벨링** → LLM이 전체 맥락 파악, 정확한 AI/Human 판단
2. **스니펫으로 학습** → 실제 추론 환경과 데이터 분포 일치
3. **데이터 증강** → 1,500개 → 5,000개 (1개 글 = 3-5개 샘플)

---

## 데이터 파이프라인

### 전체 흐름

```
1. 스크래핑 (1,500개 전체 글)
   ↓
2. LLM 라벨링 (1,500개 → AI/HUMAN 판정)
   ↓
3. 스니펫 증강 (1,500 × 3-5 = 5,000개)
   ↓
4. ML 학습 (5,000개 스니펫)
   ↓
5. CoreML 변환
```

---

## Phase 1: 데이터 수집 및 라벨링

### 1단계: 블로그 전체 글 스크래핑 (1,500개)

**목표**: 네이버 블로그 전체 본문 수집

**도구 선택**: Requests + BeautifulSoup (권장)
- 네이버 검색 API 활용
- 간단하고 빠름
- Rate limiting 쉬운 제어

**수집 전략**:
```python
# 키워드 다양화
general_keywords = ['맛집', '여행', '리뷰', '영화', '육아', '레시피']
ai_suspect_keywords = ['정보 정리', '완벽 가이드', 'OO 팁 모음']

# 클래스 균형 (AI:Human = 50:50)
# → 수집 키워드로 비율 조정
```

**출력**: `data/raw/blogs.json`
```json
[
  {
    "blog_id": "blog_001",
    "url": "https://blog.naver.com/...",
    "title": "제목",
    "full_text": "전체 본문 (1,000-5,000자)",
    "scraped_at": "2025-01-16T10:30:00"
  }
  // ... 1,500개
]
```

**품질 관리**:
- 중복 제거 (URL 기준)
- 최소 길이: 500자 이상
- 이미지만 있는 글 제외

---

### 2단계: LLM API 자동 라벨링 (1,500개)

**목표**: 전체 글 기준 AI/HUMAN 판정

**프롬프트 구조**:
```python
prompt = f"""
다음 네이버 블로그 글이 AI가 작성했는지 판단해주세요.

제목: {title}

본문 전체:
{full_text}

AI 작성 특징:
- 기계적이고 정형화된 구조
- 과도한 이모지 사용
- 완벽한 문법, 맞춤법
- 실제 경험보다 정보 나열

인간 작성 특징:
- 개인적 경험과 감정 표현
- 구어체, 자연스러운 오타
- 비정형적 문장 구조
- 진솔한 후기

JSON 형식으로 응답:
{{
  "label": "AI" or "HUMAN",
  "confidence": 0-100,
  "reasoning": "판단 근거..."
}}
"""
```

**LLM 선택**:
- **Claude 3.5 Haiku**: $0.25/1M tokens → ~$0.20 (1,500개)
- **GPT-4o-mini**: $0.15/1M tokens → ~$0.12 (1,500개)

**출력**: `data/labeled/labeled_blogs.json`
```json
[
  {
    "blog_id": "blog_001",
    "url": "https://blog.naver.com/...",
    "title": "제목",
    "full_text": "전체 본문",
    "label": "AI",  // ← LLM 판정
    "confidence": 92,
    "reasoning": "반복적인 표현, 정형화된 구조 확인",
    "labeled_at": "2025-01-16T11:00:00"
  }
  // ... 1,500개
]
```

**품질 검증**:
- `confidence < 60` → 검토 또는 제외
- AI/HUMAN 비율 확인 (40:60 ~ 60:40 목표)
- 샘플 100개 수동 검증 (Cohen's Kappa > 0.6)

---

## Phase 2: 스니펫 증강 및 모델 학습

### 3단계: 스니펫 추출 (5,000개)

**목표**: 실제 검색 결과 미리보기 시뮬레이션

**스니펫 추출 전략**:
```python
def extract_snippets(blog: dict, num_snippets: int = 3) -> list:
    """
    각 블로그 글에서 여러 위치의 스니펫 추출

    Args:
        blog: 전체 블로그 글 데이터
        num_snippets: 추출할 스니펫 개수 (기본 3개)

    Returns:
        스니펫 리스트 (각 100-300자)
    """
    full_text = blog['full_text']
    snippets = []

    # 방법 1: 문단 기준 샘플링 (자연스러움)
    paragraphs = [p for p in full_text.split('\n') if len(p) >= 100]

    # 위치별 샘플링 (시작, 중간, 끝)
    positions = ['beginning', 'middle', 'end']
    num_paragraphs = len(paragraphs)

    for i, position in enumerate(positions[:num_snippets]):
        if position == 'beginning':
            snippet = paragraphs[0][:300]
        elif position == 'middle':
            mid_idx = num_paragraphs // 2
            snippet = paragraphs[mid_idx][:300]
        else:  # end
            snippet = paragraphs[-1][:300]

        # 제목 + 스니펫 결합
        text = f"{blog['title']} {snippet}"

        snippets.append({
            'snippet_id': f"{blog['blog_id']}_s{i+1}",
            'blog_id': blog['blog_id'],
            'text': text,
            'label': blog['label'],  # ← 원본 라벨 상속
            'position': position
        })

    return snippets
```

**출력**: `data/processed/training_data.json`
```json
[
  {
    "snippet_id": "blog_001_s1",
    "blog_id": "blog_001",
    "text": "제목: ... 본문 처음 부분 (200자)",
    "label": "AI",
    "position": "beginning"
  },
  {
    "snippet_id": "blog_001_s2",
    "text": "제목: ... 본문 중간 부분 (250자)",
    "label": "AI",
    "position": "middle"
  }
  // ... 5,000개 (1,500 × 3-5)
]
```

**라벨 상속 원칙**:
- 전체 글이 AI → 모든 스니펫 `label="AI"`
- 전체 글이 Human → 모든 스니펫 `label="HUMAN"`
- **이유**: 한 글 내 일부만 AI는 비현실적

---

### 4단계: 모델 학습

**입력**: `training_data.json` (5,000개 스니펫)

**데이터 전처리**:
```python
# 1. 텍스트 정제
- HTML 태그 제거
- 이모지 보존 (AI 탐지 특징)
- 공백 정규화
- 특수문자 처리

# 2. 라벨 인코딩
- "AI" → 1
- "HUMAN" → 0

# 3. Train/Test 분할
- Train: 80% (4,000개)
- Test: 20% (1,000개)
- Stratified Split (클래스 비율 유지)
```

**모델 옵션**:

#### 옵션 A: TF-IDF + Logistic Regression (MVP 권장)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# TF-IDF 설정
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),  # unigram + bigram
    tokenizer=konlpy_tokenizer,  # 한국어 형태소 분석
    min_df=2,
    max_df=0.9
)

# Logistic Regression
model = LogisticRegression(
    solver='liblinear',
    C=1.0,
    max_iter=1000,
    random_state=42
)

# 학습
X_train = vectorizer.fit_transform(train_texts)
model.fit(X_train, train_labels)
```

**성능 목표**:
- 모델 크기: ~1MB
- 추론 속도: 10-30ms
- 정확도: **> 78%**
- 학습 시간: 5-10분

---

#### 옵션 B: 임베딩 + 신경망 (고급)

```python
import tensorflow as tf
from tensorflow.keras import layers

# Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=10000,
    oov_token="<UNK>"
)

# 모델 구조
model = tf.keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=256, input_length=200),
    layers.Dropout(0.3),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 학습
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32
)
```

**성능 목표**:
- 모델 크기: ~10MB
- 추론 속도: 50-80ms
- 정확도: **> 85%**
- 학습 시간: 30-60분

---

### 5단계: CoreML 변환

**옵션 A 변환 (TF-IDF)**:
```python
import coremltools as ct

# Pipeline 생성
from sklearn.pipeline import make_pipeline
pipeline = make_pipeline(vectorizer, model)

# CoreML 변환
coreml_model = ct.converters.sklearn.convert(
    pipeline,
    input_features=[("text", ct.models.datatypes.String())],
    output_feature_names=["ai_probability"]
)

# 메타데이터
coreml_model.author = "Your Name"
coreml_model.short_description = "네이버 블로그 AI 글 탐지 모델"
coreml_model.version = "1.0.0"

# 저장
coreml_model.save("models/exported/BlogAIDetector.mlmodel")
```

**옵션 B 변환 (신경망)**:
```python
import coremltools as ct

# TensorFlow → CoreML
coreml_model = ct.convert(
    tf_model,
    inputs=[ct.TensorType(name="text", shape=(1, 200))],
    outputs=[ct.TensorType(name="ai_probability")]
)

# 메타데이터
coreml_model.author = "Your Name"
coreml_model.short_description = "네이버 블로그 AI 글 탐지 모델 (신경망)"

# 저장
coreml_model.save("models/exported/BlogAIDetector.mlmodel")
```

**출력**: `models/exported/BlogAIDetector.mlmodel`

---

## 모델 입출력 명세

### 입력
```python
text = f"{title} {preview}"  # 제목 + 미리보기 (100-300자)
```

**예시**:
```
"맛집 추천 서울 강남 맛집 10곳을 소개합니다. 첫 번째는..."
```

### 출력
```python
{
    "ai_probability": 0.85,      # 0.0 ~ 1.0
    "prediction": "AI",          # "AI" or "HUMAN"
    "confidence_level": "확실"   # 신뢰도
}
```

**Confidence Level 계산**:
```python
def calculate_confidence(prob: float) -> str:
    if prob < 0.2 or prob > 0.8:
        return "매우 확실"
    elif prob < 0.35 or prob > 0.65:
        return "확실"
    elif prob < 0.45 or prob > 0.55:
        return "약간 의심"
    else:
        return "불확실"  # 0.45-0.55
```

---

## 품질 목표 및 검증

### 모델 성능 요구사항

| 항목 | 목표 | 이유 |
|------|------|------|
| 모델 크기 | < 10MB | Safari 확장 빠른 로딩 |
| 추론 속도 | < 100ms | 검색 결과 10-20개 실시간 처리 |
| 정확도 | > 80% | 실용적 탐지 성능 |
| 입력 길이 | 100-300자 | 제목 + 미리보기 |

### 평가 지표

```python
from sklearn.metrics import classification_report, confusion_matrix

# 테스트셋 평가
y_pred = model.predict(X_test)

# 지표 계산
print(classification_report(y_test, y_pred))
# - Accuracy: 전체 정확도
# - Precision: AI 예측 중 실제 AI 비율
# - Recall: 실제 AI 중 탐지 비율
# - F1-score: Precision과 Recall 조화평균

# 혼동 행렬
print(confusion_matrix(y_test, y_pred))
```

**최소 통과 기준**:
- Accuracy > 78% (옵션 A) 또는 > 85% (옵션 B)
- Precision > 75%
- Recall > 75%
- F1-score > 75%

---

## 개발 일정 및 리소스

### 소요 시간

| 단계 | 작업 | 예상 시간 |
|------|------|-----------|
| Phase 1.1 | 스크래핑 (1,500개) | 2-3일 |
| Phase 1.2 | LLM 라벨링 | 1일 |
| Phase 2.1 | 스니펫 증강 | 1시간 |
| Phase 2.2 | 모델 학습 | 1일 |
| Phase 2.3 | CoreML 변환 | 2시간 |
| **총계** | | **4-5일** |

### 비용 예상

| 항목 | 수량 | 단가 | 비용 |
|------|------|------|------|
| LLM API (Claude Haiku) | 1,500개 × 2,000자 | $0.25/1M | **$0.20** |
| 또는 GPT-4o-mini | 1,500개 × 2,000자 | $0.15/1M | **$0.12** |

→ **총 예상 비용: $0.15-0.25** (매우 저렴!)

---

## 파일 구조

```
Naver-Blog-AI-Detector-ML/
├── data/
│   ├── raw/
│   │   └── blogs.json                # 1,500개 전체 글
│   ├── labeled/
│   │   └── labeled_blogs.json        # 1,500개 라벨링 완료
│   └── processed/
│       ├── training_data.json        # 5,000개 스니펫
│       ├── train.json                # 4,000개 (80%)
│       └── test.json                 # 1,000개 (20%)
│
├── scripts/
│   ├── scraper.py                    # BeautifulSoup 크롤러
│   ├── labeler.py                    # Claude/GPT API 라벨링
│   ├── preprocess.py                 # 스니펫 추출 + Train/Test 분할
│   ├── train.py                      # 모델 학습
│   ├── evaluate.py                   # 모델 평가
│   └── export_coreml.py              # CoreML 변환
│
├── models/
│   ├── checkpoints/                  # 학습 중 체크포인트
│   └── exported/
│       └── BlogAIDetector.mlmodel    # 최종 CoreML 모델
│
├── docs/
│   └── project-plan.md               # 현재 파일
│
├── requirements.txt
├── README.md
└── claude.md
```

---

## 구현 체크리스트

### Phase 1: 데이터 수집 및 라벨링
- [ ] `scraper.py`: 네이버 검색 API + BeautifulSoup 크롤러
- [ ] `data/raw/blogs.json`: 1,500개 전체 글 수집
- [ ] `labeler.py`: Claude/GPT API 라벨링
- [ ] `data/labeled/labeled_blogs.json`: 1,500개 라벨링 완료
- [ ] 품질 검증: confidence > 60, AI/Human 균형

### Phase 2: 모델 학습 및 변환
- [ ] `preprocess.py`: 스니펫 추출 (3-5개/글)
- [ ] `data/processed/training_data.json`: 5,000개 스니펫
- [ ] Train/Test 분할: 80/20
- [ ] `train.py`: TF-IDF + LR (옵션 A) 학습
- [ ] `evaluate.py`: 정확도 > 78% 검증
- [ ] `export_coreml.py`: CoreML 변환
- [ ] `models/exported/BlogAIDetector.mlmodel` 생성

### Phase 2+ (선택): 고급 모델
- [ ] 옵션 B (신경망) 학습
- [ ] 정확도 > 85% 검증
- [ ] 모델 비교 분석
- [ ] 최종 모델 선택

---

## 코딩 규칙

### Python
- **PEP 8** 준수
- **snake_case** 명명
- **타입 힌트** 사용
- **Docstring** (Google 스타일)
- **random seed 고정** (재현성)

### 예시
```python
import random
import numpy as np

# 재현성 확보
random.seed(42)
np.random.seed(42)

def extract_snippets(blog: dict, num_snippets: int = 3) -> list[dict]:
    """
    블로그 글에서 여러 스니펫 추출

    Args:
        blog: 전체 블로그 글 데이터
        num_snippets: 추출할 스니펫 개수

    Returns:
        스니펫 리스트
    """
    pass
```

---

## 다음 단계: Safari 확장 통합

CoreML 모델 생성 완료 후:

1. **Swift 프로젝트에 모델 추가**
   - `BlogAIDetector.mlmodel` → Xcode 프로젝트
   - 자동 생성된 Swift 클래스 사용

2. **JavaScript ↔ Swift 통신**
   - Content Script: 제목 + 미리보기 추출
   - Message Handler: CoreML 예측 요청
   - 예측 결과 → UI 배지 표시

3. **테스트**
   - 네이버 검색 결과 페이지
   - 배지 표시 및 자동 숨김 기능
   - Safari 14+, iOS 15+ 호환성

---

## CoreML 배포 전략

### 운영 비용 분석

**CoreML (로컬 추론)**:
```
초기 개발: $0.20 (LLM 라벨링)
월 운영 비용: $0
연간 운영 비용: $0
사용자 10,000명일 때: $0

→ 총 비용: $0.20 (1회 개발 비용만)
```

**FastAPI 대안 (참고)**:
```
초기 개발: $0.20 (LLM 라벨링) + $100 (서버 설정)
월 운영 비용: $20-100 (서버, 트래픽)
연간 운영 비용: $240-1,200
사용자 10,000명일 때: $500+/월

→ 총 비용: 1년차 $340-1,300
```

**결론**: CoreML은 **영구 무료** 운영 가능 ✅

### 성능 비교

**응답 시간 (검색 결과 10개 처리)**:

| 항목 | CoreML | FastAPI |
|------|--------|---------|
| 단일 추론 | 10-30ms | 100-500ms |
| 10개 배치 | 100-300ms | 1,000-5,000ms |
| 사용자 체감 | ⚡ 즉시 | ⏳ 답답함 |

**CoreML 성능 이점**:
- 네트워크 왕복 시간 0ms
- 서버 대기 큐 없음
- 디바이스 병렬 처리 가능

### 개인정보 보호

**CoreML 방식**:
```
사용자 검색 내용 → 디바이스 내 CoreML → 예측 결과
                    ↑
                서버 전송 없음 ✅
```

**장점**:
- GDPR, 개인정보보호법 준수 용이
- App Store 심사 통과 유리
- 사용자 신뢰 확보

**FastAPI 대안 문제점**:
- 모든 검색 키워드/블로그 내용 서버 전송
- 개인정보 처리방침 작성 필수
- 사용자 동의 절차 복잡
- 데이터 보관/삭제 정책 필요

### 확장성

**CoreML**:
```
사용자 수 증가 → 각 디바이스에서 독립 실행
               → 성능/비용 동일 유지 ✅

사용자 1명: 10-30ms, $0/월
사용자 10,000명: 10-30ms, $0/월
사용자 100만명: 10-30ms, $0/월
```

**FastAPI 대안**:
```
사용자 수 증가 → 서버 부하 증가
               → 스케일링 필요 (비용 급증)

사용자 1명: 200ms, $20/월
사용자 10,000명: 500ms+, $200/월
사용자 100만명: 서버 병목, $5,000+/월
```

### 모델 업데이트 전략

**문제**: CoreML은 앱 업데이트로 모델 배포

**해결 방법**:
```swift
// 하이브리드 전략: 번들 모델 + 원격 업데이트
class AIDetectorModel {
    private var bundledModel: BlogAIDetector  // 앱 번들
    private var remoteModel: BlogAIDetector?  // 원격 다운로드

    var activeModel: BlogAIDetector {
        return remoteModel ?? bundledModel
    }

    func checkForModelUpdate() async {
        // Firebase Remote Config 또는 GitHub Release
        if let newModelURL = await fetchLatestModelURL() {
            remoteModel = try? await downloadAndLoadModel(newModelURL)
        }
    }
}
```

**이점**:
- 앱 업데이트 없이 모델 개선 가능
- 점진적 롤아웃 가능 (A/B 테스트)
- 문제 시 즉시 롤백

### App Store 배포

**CoreML 장점**:
- ✅ 서버 의존성 없음 → 심사 우호적
- ✅ 개인정보 처리 단순함
- ✅ 오프라인 작동 → 기능 완결성
- ✅ 빠른 심사 통과 예상

**필요 준비사항**:
1. 개인정보 처리방침 (간단)
2. 모델 정확도 검증 문서
3. 오작동 시 면책 조항

---

**목표**: MVP 빠른 출시 → 사용자 피드백 → 점진적 개선

**핵심 전략**: CoreML 온디바이스 추론으로 비용 $0, 최고 성능, 개인정보 보호 달성
