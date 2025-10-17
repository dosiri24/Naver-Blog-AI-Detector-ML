# 전처리 파이프라인 계획

> **동적 데이터 처리**: 모든 데이터셋 크기는 런타임에 계산되며, 소규모/대규모 데이터 모두 지원

## 🎯 목표

**Phase 1**: LLM으로 전체 블로그 글의 AI/HUMAN 판정
**Phase 2**: 각 글을 3~5개 스니펫(100-300자)으로 분할 → 학습 데이터 생성

## 📊 데이터 흐름

```
blogs.json (N개)
   ↓ Gemini-2.5-flash-latest
labeled_blogs.json (N개 + AI/HUMAN 라벨)
   ↓ 스니펫 분할 (평균 3.3x)
training_data.json (~3.3N개 snippet)
```

**데이터 크기 계산 방식**:
- 입력 크기: `N = len(blogs.json)`
- 라벨링 출력: `N개`
- 스니펫 출력: `~3.3N개` (글 길이에 따라 3~5개 분할)

## 📁 디렉토리 구조

```
Data-Collection/
├── data/
│   ├── blogs.json                    # 원본 스크랩 데이터
│   ├── labeled/
│   │   └── labeled_blogs.json        # LLM 라벨링 완료
│   └── processed/
│       └── training_data.json        # 스니펫 분할 완료
│
├── scripts/
│   ├── scraper.py                    # (기존) 블로그 스크래핑
│   ├── labeler.py                    # (신규) LLM 자동 라벨링
│   └── preprocess.py                 # (신규) 스니펫 분할
│
├── prompts/
│   └── labeling_prompt.md            # AI 판단 기준 프롬프트
│
└── docs/
    └── preprocessing-plan.md         # 현재 파일
```

## 🔧 Phase 1: LLM 라벨링 (`labeler.py`)

### 입력
- `data/blogs.json` (동적 크기 N)
- `prompts/labeling_prompt.md` (동적 로드)

### 처리
- Gemini-2.5-flash-latest API 호출
- 입력: `title` + `full_text` (전체 본문)
- 판단 기준: 프롬프트 파일 내용
- **배치 처리**: 10개씩 API 호출 (rate limit 고려)

### 출력 스키마 (`data/labeled/labeled_blogs.json`)
```json
[
  {
    "blog_id": "blog_0059",
    "url": "https://...",
    "title": "알로에 키우기 종류...",
    "full_text": "전체 본문...",
    "keyword": "알로에 키우기",
    "scraped_at": "2025-10-16T18:01:41.295688",
    "duplicate_count": 0,
    "last_seen_at": "2025-10-16T18:01:41.295688",

    "label": "AI",
    "confidence": 0.85,
    "reasoning": "반복적인 키워드 나열과 부자연스러운 문장 구조...",
    "labeled_at": "2025-10-16T19:00:00.123456"
  }
]
```

**필드 설명**:
- `label`: `"AI"` | `"HUMAN"`
- `confidence`: `0.0 ~ 1.0` (LLM의 확신도)
- `reasoning`: 판단 근거 (100자 이내)
- `labeled_at`: 라벨링 시각 (ISO 8601)

### 프롬프트 설계 (`prompts/labeling_prompt.md`)

```markdown
# AI 블로그 글 판단 기준

당신은 네이버 블로그 글이 AI로 생성되었는지 판단하는 전문가입니다.
다음 기준으로 판단하세요:

## AI 생성 글의 특징
1. **키워드 반복**: 동일 키워드가 부자연스럽게 반복
2. **형식적 구조**: 서론-본론-결론이 기계적으로 구분
3. **이모지 과다**: 각 문단마다 이모지 사용
4. **나열식 문장**: "~하고요", "~있어요" 등 단순 나열
5. **광고성 표현**: "꼭 확인하세요", "추천드려요" 등
6. **불필요한 강조**: 특정 키워드에 **굵은 글씨** 과다 사용

## 인간 작성 글의 특징
1. **자연스러운 흐름**: 개인 경험과 감정 표현
2. **구어체 사용**: "그치만", "근데", "솔직히" 등
3. **맥락적 일관성**: 주제가 자연스럽게 전개
4. **독특한 표현**: 개인만의 어투와 표현 방식
5. **일상적 디테일**: 구체적인 개인 경험 묘사

## 출력 형식 (JSON)
반드시 다음 형식의 JSON만 출력하세요:
{
  "label": "AI" or "HUMAN",
  "confidence": 0.85,
  "reasoning": "판단 근거 (100자 이내)"
}
```

### API 함수 시그니처

```python
def label_with_gemini(
    blog: Dict,
    prompt_template: str,
    api_key: str
) -> Dict:
    """Gemini API로 단일 블로그 글 라벨링

    Args:
        blog: 원본 블로그 데이터 (title, full_text 포함)
        prompt_template: prompts/labeling_prompt.md 내용
        api_key: Gemini API 키

    Returns:
        원본 데이터 + 라벨링 결과 (label, confidence, reasoning, labeled_at)
    """
    pass


def label_all_blogs(
    input_file: Path,
    output_file: Path,
    prompt_file: Path,
    api_key: str,
    batch_size: int = 10,
    limit: int = None  # 테스트용: 처음 N개만 처리
) -> Dict[str, int]:
    """전체 블로그 데이터 라벨링

    Args:
        input_file: data/blogs.json
        output_file: data/labeled/labeled_blogs.json
        prompt_file: prompts/labeling_prompt.md
        api_key: Gemini API 키
        batch_size: API 호출 배치 크기
        limit: 테스트용 데이터 제한 (None이면 전체 처리)

    Returns:
        통계 정보 (total, ai_count, human_count, avg_confidence)
    """
    pass
```

## 🔧 Phase 2: 스니펫 분할 (`preprocess.py`)

### 입력
- `data/labeled/labeled_blogs.json` (N개 라벨링 완료)

### 처리 로직

**1. 글 길이 기반 분할 개수 결정**:
```python
def get_num_snippets(text_length: int) -> int:
    """글 길이에 따른 스니펫 개수 결정"""
    if text_length < 500:
        return 3
    elif text_length < 1500:
        return 4
    else:
        return 5
```

**2. 위치별 스니펫 추출**:
- 시작 (0~20%)
- 중간 (40~60%)
- 끝 (80~100%)
- 추가 위치 동적 계산

**3. 스니펫 길이 제약**:
- 최소: 100자
- 최대: 300자
- 네이버 미리보기 환경 시뮬레이션

**4. 라벨 상속**: 원본 글의 `label` (AI/HUMAN) 그대로 사용

### 출력 스키마 (`data/processed/training_data.json`)

```json
[
  {
    "snippet_id": "blog_0059_01",
    "original_blog_id": "blog_0059",
    "title": "알로에 키우기 종류...",
    "snippet_text": "다육식물 4개에 만원 하는데 그 중에서골라온 것이 바로 알로에 베라 화분이에요...",
    "position": "start",
    "snippet_length": 150,
    "label": "AI",
    "confidence": 0.85,
    "keyword": "알로에 키우기",
    "created_at": "2025-10-16T20:00:00.123456"
  }
]
```

**필드 설명**:
- `snippet_id`: `{blog_id}_{순번:02d}` 형식
- `original_blog_id`: 원본 블로그 추적용
- `title`: 원본 글 제목 (스니펫에 항상 포함)
- `snippet_text`: 추출된 본문 일부 (100~300자)
- `position`: `"start"` | `"early_middle"` | `"middle"` | `"late_middle"` | `"end"`
- `snippet_length`: 실제 스니펫 길이
- `label`: 원본 글의 라벨 상속
- `confidence`: 원본 글의 confidence 상속
- `keyword`: 원본 글의 검색 키워드
- `created_at`: 스니펫 생성 시각

### 함수 시그니처

```python
def split_into_snippets(
    full_text: str,
    min_length: int = 100,
    max_length: int = 300
) -> List[Dict]:
    """전체 글을 3~5개 스니펫으로 분할

    Args:
        full_text: 원본 글 전체 본문
        min_length: 최소 스니펫 길이
        max_length: 최대 스니펫 길이

    Returns:
        스니펫 리스트 (position, text, length 포함)
    """
    text_len = len(full_text)
    num_snippets = get_num_snippets(text_len)

    positions = ["start", "early_middle", "middle", "late_middle", "end"][:num_snippets]
    snippets = []

    for i, pos in enumerate(positions):
        start_ratio = i / num_snippets
        end_ratio = (i + 1) / num_snippets

        start_idx = int(text_len * start_ratio)
        end_idx = int(text_len * end_ratio)

        # 100~300자 범위 조정
        snippet = full_text[start_idx:start_idx + max_length]
        if len(snippet) < min_length and start_idx + min_length <= text_len:
            snippet = full_text[start_idx:start_idx + min_length]

        snippets.append({
            "position": pos,
            "text": snippet.strip(),
            "length": len(snippet.strip())
        })

    return snippets


def preprocess_all_blogs(
    input_file: Path,
    output_file: Path,
    min_length: int = 100,
    max_length: int = 300,
    limit: int = None  # 테스트용: 처음 N개만 처리
) -> Dict[str, int]:
    """전체 라벨링 데이터 → 스니펫 분할

    Args:
        input_file: data/labeled/labeled_blogs.json
        output_file: data/processed/training_data.json
        min_length: 최소 스니펫 길이
        max_length: 최대 스니펫 길이
        limit: 테스트용 데이터 제한 (None이면 전체 처리)

    Returns:
        통계 정보 (total_blogs, total_snippets, avg_snippets_per_blog)
    """
    pass
```

## 📈 예상 결과 (동적 계산)

| 단계 | 입력 | 출력 | 증강 비율 |
|------|------|------|----------|
| Phase 1 | N개 블로그 | N개 라벨링 | 1.0x |
| Phase 2 | N개 라벨링 | ~3.3N개 스니펫 | 3.3x |

**실제 스니펫 수 계산**:
```python
total_snippets = sum(get_num_snippets(len(blog['full_text'])) for blog in blogs)
```

## 🧪 테스트 모드

### 소규모 데이터 테스트

```bash
# 1. 처음 10개만 라벨링
python scripts/labeler.py --limit 10

# 2. 라벨링된 10개만 스니펫 분할
python scripts/preprocess.py --limit 10

# 3. 결과 확인
python -c "
import json
with open('data/labeled/labeled_blogs.json') as f:
    labeled = json.load(f)
with open('data/processed/training_data.json') as f:
    snippets = json.load(f)
print(f'라벨링: {len(labeled)}개')
print(f'스니펫: {len(snippets)}개')
print(f'증강 비율: {len(snippets)/len(labeled):.1f}x')
"
```

### 전체 데이터 처리

```bash
# 1. 전체 라벨링 (limit 없음)
python scripts/labeler.py --api-key YOUR_GEMINI_KEY

# 2. 전체 스니펫 분할
python scripts/preprocess.py

# 3. 통계 확인
python scripts/preprocess.py --stats
```

## ⚙️ CLI 인터페이스 설계

### `labeler.py`

```bash
# 기본 사용 (전체 처리)
python scripts/labeler.py --api-key YOUR_KEY

# 테스트 모드 (처음 10개만)
python scripts/labeler.py --api-key YOUR_KEY --limit 10

# 배치 크기 조정 (API rate limit 고려)
python scripts/labeler.py --api-key YOUR_KEY --batch-size 5

# 프롬프트 파일 커스터마이징
python scripts/labeler.py --api-key YOUR_KEY --prompt prompts/custom.md

# 옵션 전체
python scripts/labeler.py \
  --api-key YOUR_KEY \
  --input data/blogs.json \
  --output data/labeled/labeled_blogs.json \
  --prompt prompts/labeling_prompt.md \
  --batch-size 10 \
  --limit 50 \
  --verbose
```

### `preprocess.py`

```bash
# 기본 사용 (전체 처리)
python scripts/preprocess.py

# 테스트 모드 (처음 10개만)
python scripts/preprocess.py --limit 10

# 스니펫 길이 커스터마이징
python scripts/preprocess.py --min-length 150 --max-length 250

# 통계만 출력 (처리 안 함)
python scripts/preprocess.py --stats

# 옵션 전체
python scripts/preprocess.py \
  --input data/labeled/labeled_blogs.json \
  --output data/processed/training_data.json \
  --min-length 100 \
  --max-length 300 \
  --limit 50 \
  --verbose
```

## 💰 비용 추정 (동적)

**Gemini-2.5-flash-latest 요금**: ~$0.15-0.25 / 1M tokens

**토큰 계산**:
```python
avg_chars_per_blog = 2000  # 평균 글자 수
tokens_per_blog = avg_chars_per_blog * 1.5  # 한글은 약 1.5 tokens/char
total_tokens = len(blogs) * tokens_per_blog
estimated_cost = (total_tokens / 1_000_000) * 0.20  # 평균 $0.20/1M
```

**예시**:
- 100개 블로그: ~$0.006
- 1,000개 블로그: ~$0.06
- 10,000개 블로그: ~$0.60

## 🎯 품질 게이트

### Phase 1 완료 조건
- ✅ 모든 블로그 라벨링 완료 (실패 0개)
- ✅ AI/HUMAN 비율: 40-60% (균형 확인)
- ✅ 평균 confidence > 0.7
- ✅ reasoning 필드 누락 없음
- ✅ JSON 파싱 오류 없음

**검증 코드**:
```python
def validate_phase1(labeled_file: Path) -> Dict[str, Any]:
    with open(labeled_file) as f:
        data = json.load(f)

    total = len(data)
    ai_count = sum(1 for x in data if x['label'] == 'AI')
    human_count = total - ai_count
    avg_confidence = sum(x['confidence'] for x in data) / total

    return {
        'total': total,
        'ai_ratio': ai_count / total,
        'human_ratio': human_count / total,
        'avg_confidence': avg_confidence,
        'passed': (
            0.4 <= ai_count / total <= 0.6 and
            avg_confidence > 0.7
        )
    }
```

### Phase 2 완료 조건
- ✅ 스니펫 생성 비율: 3.0~3.5x (평균 3.3x)
- ✅ 모든 snippet 길이: 100~300자
- ✅ 원본 blog_id 추적 가능
- ✅ 라벨 분포 유지 (±5% 오차)
- ✅ position 필드 누락 없음

**검증 코드**:
```python
def validate_phase2(
    labeled_file: Path,
    snippet_file: Path
) -> Dict[str, Any]:
    with open(labeled_file) as f:
        labeled = json.load(f)
    with open(snippet_file) as f:
        snippets = json.load(f)

    total_blogs = len(labeled)
    total_snippets = len(snippets)
    augmentation_ratio = total_snippets / total_blogs

    # 길이 검증
    length_valid = all(
        100 <= s['snippet_length'] <= 300
        for s in snippets
    )

    # 라벨 분포 검증
    original_ai_ratio = sum(1 for x in labeled if x['label'] == 'AI') / total_blogs
    snippet_ai_ratio = sum(1 for x in snippets if x['label'] == 'AI') / total_snippets
    label_distribution_match = abs(original_ai_ratio - snippet_ai_ratio) < 0.05

    return {
        'total_blogs': total_blogs,
        'total_snippets': total_snippets,
        'augmentation_ratio': augmentation_ratio,
        'length_valid': length_valid,
        'label_distribution_match': label_distribution_match,
        'passed': (
            3.0 <= augmentation_ratio <= 3.5 and
            length_valid and
            label_distribution_match
        )
    }
```

## 🔄 재실행 및 중복 방지

### 부분 재실행 지원

**Phase 1 (라벨링)**:
- 기존 `labeled_blogs.json` 존재 시 → 이미 라벨링된 blog_id는 스킵
- 새로운 블로그만 라벨링하여 추가

```python
def load_existing_labels(output_file: Path) -> Dict[str, Dict]:
    """기존 라벨링 데이터 로드"""
    if not output_file.exists():
        return {}

    with open(output_file) as f:
        existing = json.load(f)

    return {blog['blog_id']: blog for blog in existing}
```

**Phase 2 (스니펫 분할)**:
- 기존 `training_data.json` 존재 시 → 이미 분할된 blog_id는 스킵
- 새로운 라벨링 데이터만 스니펫 분할

```python
def load_existing_snippets(output_file: Path) -> Set[str]:
    """이미 스니펫 분할된 blog_id 추출"""
    if not output_file.exists():
        return set()

    with open(output_file) as f:
        snippets = json.load(f)

    return {s['original_blog_id'] for s in snippets}
```

## 📊 로깅 및 통계

### 로깅 형식

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)

# 예시 로그
# 2025-10-16 19:00:00 [INFO] Phase 1 시작: 500개 블로그 라벨링
# 2025-10-16 19:00:10 [INFO] 배치 1/50 완료 (10개 처리, 평균 confidence: 0.82)
# 2025-10-16 19:05:00 [INFO] Phase 1 완료: AI 210개 (42%), HUMAN 290개 (58%)
# 2025-10-16 19:05:05 [INFO] Phase 2 시작: 500개 블로그 → 스니펫 분할
# 2025-10-16 19:05:15 [INFO] Phase 2 완료: 1,650개 스니펫 생성 (3.3x)
```

### 통계 출력 형식

```bash
$ python scripts/labeler.py --stats

=== Phase 1 통계 ===
입력: 500개 블로그
출력: 500개 라벨링 완료
- AI: 210개 (42%)
- HUMAN: 290개 (58%)
- 평균 confidence: 0.82
- 실패: 0개
비용: ~$0.12
소요 시간: 5분 23초

$ python scripts/preprocess.py --stats

=== Phase 2 통계 ===
입력: 500개 라벨링 데이터
출력: 1,650개 스니펫
- 증강 비율: 3.3x
- 평균 snippet 길이: 185자
- 3개 분할: 150개 블로그
- 4개 분할: 250개 블로그
- 5개 분할: 100개 블로그
라벨 분포: AI 693개 (42%), HUMAN 957개 (58%)
소요 시간: 12초
```

## 🚀 다음 단계: ML 학습

전처리 완료 후 다음 단계:

1. **Train/Test 분할**: `training_data.json` → 80% train, 20% test
2. **모델 학습**: TF-IDF + Logistic Regression (MVP)
3. **모델 평가**: Accuracy, Precision, Recall, F1
4. **CoreML 변환**: Safari 확장 통합

**관련 문서**: (추후 작성 예정)
- `docs/ml-training-plan.md`
- `docs/coreml-conversion.md`

---

**작성일**: 2025-10-16
**버전**: 1.0
**작성자**: Claude Code
