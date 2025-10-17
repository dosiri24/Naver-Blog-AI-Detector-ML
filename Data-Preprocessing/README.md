# Data-Preprocessing

> 네이버 블로그 AI 탐지 모델을 위한 데이터 전처리 파이프라인

## 📊 전처리 파이프라인 개요

```
Phase 1: LLM 라벨링
  blogs.json (N개)
    ↓ Gemini-2.5-flash-latest
  labeled_blogs.json (N개 + AI/HUMAN 라벨)

Phase 2: 스니펫 분할
  labeled_blogs.json (N개)
    ↓ 3~5개 스니펫 분할
  training_data.json (~3.3N개)
```

## 📁 디렉토리 구조

```
Data-Preprocessing/
├── data/
│   ├── labeled/
│   │   └── labeled_blogs.json        # Phase 1 출력
│   └── processed/
│       └── training_data.json        # Phase 2 출력
│
├── scripts/
│   ├── labeler.py                    # Phase 1: LLM 라벨링
│   └── preprocess.py                 # Phase 2: 스니펫 분할
│
├── prompts/
│   └── labeling_prompt.md            # AI 판단 기준 프롬프트
│
├── logs/                              # 실행 로그
├── docs/                              # 상세 문서
├── .env.example                       # API 키 템플릿
├── .gitignore                         # Git 제외 파일
├── requirements.txt                   # Python 패키지
└── README.md                          # 현재 파일
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Python 3.9+ 필요
python --version

# 패키지 설치
pip install -r requirements.txt
```

### 2. API 키 설정 (.env 파일)

```bash
# 1. .env 파일 생성
cp .env.example .env

# 2. .env 파일 편집
vi .env
# 또는
nano .env

# 3. API 키와 모델명 입력
# GEMINI_API_KEY=your_api_key_here
# GEMINI_MODEL=gemini-2.5-flash-latest
```

**API 키 발급**: https://makersuite.google.com/app/apikey

### 3. Phase 1: LLM 라벨링

#### 테스트 실행 (처음 10개만)

```bash
cd scripts
python labeler.py --limit 10
```

#### 전체 실행

```bash
python labeler.py
```

#### 옵션 전체

```bash
# .env 파일 사용 (권장)
python labeler.py \
  --input ../Data-Collection/data/blogs.json \
  --output ../data/labeled/labeled_blogs.json \
  --prompt ../prompts/labeling_prompt.md \
  --batch-size 10 \
  --limit 50 \
  --verbose

# 또는 CLI 인자로 직접 전달
python labeler.py \
  --api-key YOUR_KEY \
  --model gemini-2.5-flash-latest \
  --input ../Data-Collection/data/blogs.json \
  --output ../data/labeled/labeled_blogs.json \
  --prompt ../prompts/labeling_prompt.md \
  --batch-size 10 \
  --limit 50 \
  --verbose
```

### 4. Phase 2: 스니펫 분할

#### 테스트 실행 (처음 10개만)

```bash
python preprocess.py --limit 10
```

#### 전체 실행

```bash
python preprocess.py
```

#### 통계만 확인

```bash
python preprocess.py --stats
```

#### 옵션 전체

```bash
python preprocess.py \
  --input ../data/labeled/labeled_blogs.json \
  --output ../data/processed/training_data.json \
  --min-length 100 \
  --max-length 300 \
  --limit 50 \
  --verbose
```

## 📋 스크립트 상세

### `labeler.py` - LLM 자동 라벨링

**기능**:
- Gemini-2.5-flash-latest API로 블로그 글 AI/HUMAN 판정
- 동적 프롬프트 로드 (`prompts/labeling_prompt.md`)
- 배치 처리 (기본: 10개씩)
- 재실행 지원 (기존 라벨링 데이터 스킵)
- 품질 게이트 자동 검증

**출력 데이터 구조**:
```json
{
  "blog_id": "blog_0001",
  "url": "https://...",
  "title": "제목",
  "full_text": "전체 본문...",
  "keyword": "검색 키워드",
  "scraped_at": "2025-10-16T18:00:00",

  "label": "AI",
  "reasoning": "판단 근거...",
  "labeled_at": "2025-10-16T19:00:00"
}
```

**품질 게이트**:
- ✅ AI/HUMAN 비율: 40-60%
- ✅ 모든 필드 존재

**예상 비용**:
- 100개: ~$0.006
- 1,000개: ~$0.06
- 10,000개: ~$0.60

### `preprocess.py` - 스니펫 분할

**기능**:
- 라벨링된 블로그를 3~5개 스니펫으로 분할
- 글 길이에 따른 자동 분할 개수 결정
  - < 500자: 3개
  - 500~1500자: 4개
  - ≥ 1500자: 5개
- 스니펫 길이: 100~300자 (네이버 미리보기 환경)
- 라벨 자동 상속 (원본 블로그의 AI/HUMAN)
- 재실행 지원 (기존 스니펫 스킵)
- 품질 게이트 자동 검증

**출력 데이터 구조**:
```json
{
  "snippet_id": "blog_0001_01",
  "original_blog_id": "blog_0001",
  "title": "제목",
  "snippet_text": "스니펫 본문 (100-300자)...",
  "position": "start",
  "snippet_length": 150,
  "label": "AI",
  "keyword": "검색 키워드",
  "created_at": "2025-10-16T20:00:00"
}
```

**품질 게이트**:
- ✅ 증강 비율: 3.0~3.5x
- ✅ 스니펫 길이: 100~300자
- ✅ 라벨 분포 일치 (±5% 오차)

## 📊 실행 예시

### Phase 1 출력 (라벨링)

```
=== Phase 1 통계 ===
총 라벨링: 500개
- AI: 210개 (42.0%)
- HUMAN: 290개 (58.0%)
소요 시간: 323.5초 (5.4분)
예상 비용: ~$0.1200

✅ 품질 게이트 통과
```

### Phase 2 출력 (스니펫 분할)

```
=== Phase 2 통계 ===
입력: 500개 블로그
출력: 1,650개 스니펫
증강 비율: 3.3x
평균 snippet 길이: 185자

[스니펫 분할 분포]
  3개 분할: 150개 블로그
  4개 분할: 250개 블로그
  5개 분할: 100개 블로그

[라벨 분포]
  AI:
    스니펫: 693개 (42.0%)
    원본 블로그: 210개 (42.0%)
  HUMAN:
    스니펫: 957개 (58.0%)
    원본 블로그: 290개 (58.0%)

[위치 분포]
  start: 500개
  early_middle: 350개
  middle: 500개
  late_middle: 200개
  end: 100개

소요 시간: 12.3초

[품질 게이트 검증]
  augmentation_ratio: ✅ 통과
  snippet_length: ✅ 통과
  label_distribution_match: ✅ 통과

✅ 모든 품질 게이트 통과
```

## 🔧 고급 사용법

### 1. 프롬프트 커스터마이징

```bash
# 1. prompts/labeling_prompt.md 수정
vi prompts/labeling_prompt.md

# 2. 커스텀 프롬프트로 실행
python labeler.py --prompt prompts/custom_prompt.md
```

### 2. 재실행 (중단된 작업 이어서)

```bash
# Phase 1이 50%에서 중단된 경우
# → 기존 labeled_blogs.json에서 이미 처리된 blog_id는 자동 스킵
python labeler.py

# Phase 2도 동일하게 재실행 가능
python preprocess.py
```

### 3. 배치 크기 조정 (API rate limit 대응)

```bash
# Rate limit 오류 발생 시 배치 크기 줄이기
python labeler.py --batch-size 5
```

### 4. 스니펫 길이 조정

```bash
# 더 짧은 스니펫 (50~200자)
python preprocess.py --min-length 50 --max-length 200

# 더 긴 스니펫 (150~400자)
python preprocess.py --min-length 150 --max-length 400
```

## 📝 로그 파일

모든 실행 로그는 `logs/` 디렉토리에 저장됩니다:

```
logs/
├── labeling_20251016_190000.log      # Phase 1 로그
└── preprocessing_20251016_200000.log  # Phase 2 로그
```

## ❌ 문제 해결

### 1. API 키 오류

```
ERROR: Gemini API 키가 필요합니다.
```

**해결**:
```bash
# 방법 1: .env 파일 생성 (권장)
cp .env.example .env
# .env 파일에 GEMINI_API_KEY=your_api_key_here 입력

# 방법 2: CLI 인자 사용
python labeler.py --api-key your_api_key_here
```

### 2. 패키지 없음

```
ERROR: google-generativeai 패키지가 설치되지 않았습니다.
ERROR: python-dotenv 패키지가 설치되지 않았습니다.
```

**해결**:
```bash
pip install -r requirements.txt
```

### 3. Rate limit 오류

```
API 호출 실패: Rate limit exceeded
```

**해결**:
```bash
# 배치 크기 줄이기
python labeler.py --batch-size 5
```

### 4. JSON 파싱 오류

```
JSON 파싱 실패
```

**해결**:
- Gemini 응답이 JSON 형식이 아닐 수 있음
- `prompts/labeling_prompt.md`에서 JSON 출력 형식 강조
- 재시도 (최대 3회 자동 재시도)

## 🎯 다음 단계

전처리 완료 후:

1. **Train/Test 분할**: `training_data.json` → 80% train, 20% test
2. **모델 학습**: TF-IDF + Logistic Regression (MVP)
3. **모델 평가**: Accuracy, Precision, Recall, F1
4. **CoreML 변환**: Safari 확장 통합

**관련 문서**:
- [docs/preprocessing-plan.md](docs/preprocessing-plan.md) - 전처리 상세 계획
- `../docs/ml-training-plan.md` (추후 작성) - ML 학습 계획
- `../docs/coreml-conversion.md` (추후 작성) - CoreML 변환 가이드

## 📚 추가 정보

**데이터 흐름**:
```
../Data-Collection/data/blogs.json (스크랩 데이터)
  ↓ labeler.py
data/labeled/labeled_blogs.json (라벨링 완료)
  ↓ preprocess.py
data/processed/training_data.json (학습 데이터)
  ↓ train.py (추후 구현)
models/exported/BlogAIDetector.mlmodel (CoreML 모델)
```

**프로젝트 홈**:
- [GitHub Repository](https://github.com/your-repo/Naver-Blog-AI-Detector)
- [프로젝트 계획서](../CLAUDE.md)

---

**작성일**: 2025-10-16
**버전**: 1.0
**작성자**: Claude Code
