# Naver Blog AI Detector - ML Module

> Python ML 모듈: 네이버 블로그 AI 글 탐지 모델 학습 및 CoreML 변환

## 프로젝트 구조

```
Naver-Blog-AI-Detector/
├── Naver-Blog-AI-Detector-ML/    # Python ML 모듈 (현재 위치)
│   ├── data/                      # 데이터셋
│   ├── scripts/                   # Python 스크립트
│   ├── models/                    # 학습된 모델
│   └── docs/                      # 문서
└── safari-extensions/             # Swift Safari 확장
```

## 핵심 파이프라인

```
1. 스크래핑 (1,500개 전체 블로그 글)
   ↓
2. LLM 라벨링 (1,500개 → AI/HUMAN 판정)
   ↓
3. 스니펫 증강 (1,500 × 3-5 = 5,000개)
   ↓
4. ML 학습 (5,000개 스니펫)
   ↓
5. CoreML 변환 (온디바이스 추론)
```

## 왜 CoreML 방식인가?

**배포 최적화**:
- ✅ **운영 비용 $0** (서버 불필요)
- ✅ **응답 10-30ms** (로컬 추론)
- ✅ **개인정보 보호** (디바이스 내 처리)
- ✅ **오프라인 작동** (네트워크 불필요)

**데이터 전략**:
- **전체 글 수집** → LLM이 전체 맥락 파악, 정확한 라벨링
- **스니펫으로 학습** → 실제 Safari 확장 사용 환경과 동일한 데이터 분포
- **데이터 증강** → 1,500개 → 5,000개 (1개 글 = 3-5개 샘플)

실제 사용: Safari 확장은 **제목 + 미리보기 (100-300자)만** 볼 수 있음

## 기술 스택

- **Python**: 3.9+
- **ML**: scikit-learn (TF-IDF + Logistic Regression) 또는 TensorFlow (신경망)
- **변환**: coremltools 7.0+
- **크롤링**: requests, BeautifulSoup4 (권장) 또는 Selenium (백업)
- **LLM**: anthropic (Claude), openai (GPT)
- **한국어**: KoNLPy

## 디렉토리 구조

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
│   └── project-plan.md               # 상세 계획서
│
├── requirements.txt
├── README.md
└── claude.md                         # 현재 파일
```

## 개발 단계

### Phase 1: 데이터 수집 및 라벨링

**1단계: 블로그 전체 글 스크래핑 (1,500개)**
- 도구: Requests + BeautifulSoup (권장)
- 수집: 네이버 검색 API → 전체 본문 (1,000-5,000자)
- 출력: `data/raw/blogs.json`

**2단계: LLM API 자동 라벨링 (1,500개)**
- LLM: Claude 3.5 Haiku 또는 GPT-4o-mini
- 비용: ~$0.15-0.25 (1,500개)
- 라벨: AI/HUMAN + confidence + reasoning
- 출력: `data/labeled/labeled_blogs.json`

### Phase 2: 스니펫 증강 및 모델 학습

**3단계: 스니펫 추출 (5,000개)**
- 각 블로그 글에서 3-5개 스니펫 추출 (100-300자)
- 위치: 시작, 중간, 끝
- 라벨 상속: 전체 글의 AI/HUMAN 라벨 그대로 사용
- 출력: `data/processed/training_data.json`

**4단계: 모델 학습**
- 입력: 5,000개 스니펫
- Train/Test: 80/20 분할
- 옵션 A (MVP): TF-IDF + Logistic Regression (~1MB, 80% 정확도)
- 옵션 B (고급): 임베딩 + 신경망 (~10MB, 85%+ 정확도)

**5단계: CoreML 변환**
- coremltools로 .mlmodel 생성
- 출력: `models/exported/BlogAIDetector.mlmodel`

## 모델 입출력

**입력**: `text = f"{title} {preview}"` (100-300자)

**출력**:
```python
{
    "ai_probability": 0.85,      # 0.0 ~ 1.0
    "prediction": "AI",          # "AI" or "HUMAN"
    "confidence_level": "확실"   # "매우 확실", "확실", "약간 의심", "불확실"
}
```

## 빠른 시작

```bash
# 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 실행 순서
python scripts/scraper.py
python scripts/labeler.py --provider openai --api-key YOUR_KEY
python scripts/preprocess.py
python scripts/train.py
python scripts/evaluate.py
python scripts/export_coreml.py
```

## 성능 목표

| 항목 | 목표 | 이유 |
|------|------|------|
| 모델 크기 | < 10MB | Safari 확장 빠른 로딩 |
| 추론 속도 | < 100ms | 검색 결과 10-20개 실시간 처리 |
| 정확도 | > 80% | 실용적 탐지 성능 |
| 입력 길이 | 100-300자 | 제목 + 미리보기 |

## Claude Code 가이드

### 핵심 전략
- **MVP 우선**: 옵션 A (TF-IDF) → 빠른 프로토타입
- **데이터 품질**: 전체 글로 라벨링 → 높은 정확도
- **분포 일치**: 스니펫으로 학습 → Train-Test 분포 일치
- **경량화**: 모델 크기 < 10MB, 추론 < 100ms

### 개발 우선순위
1. **데이터 수집**: 1,500개 전체 글 (높은 라벨링 품질)
2. **LLM 라벨링**: Claude/GPT로 정확한 AI/HUMAN 판정
3. **스니펫 증강**: 5,000개 학습 데이터 구축 (데이터 효율성)
4. **모델 학습**: TF-IDF + LR로 빠른 MVP
5. **CoreML 변환**: Safari 확장 통합

### 코딩 규칙
- **PEP 8** 준수
- **snake_case** 명명
- **타입 힌트** 사용
- **Docstring** (Google 스타일)
- **random seed 고정** (재현성)

### 품질 게이트
- **Phase 1**: 1,500개 수집, confidence > 60, AI/Human 균형
- **Phase 2**: 5,000개 스니펫, Train/Test 분할 80/20
- **모델**: 정확도 > 78% (옵션 A) 또는 > 85% (옵션 B)
- **CoreML**: 추론 < 100ms, Safari 호환성 확인

## 예상 일정 및 비용

### 소요 시간
- 스크래핑 (1,500개): 2-3일
- LLM 라벨링: 1일
- 스니펫 증강: 1시간
- 모델 학습: 1일
- CoreML 변환: 2시간
- **총계: 4-5일**

### 비용
- LLM API: ~$0.15-0.25 (1,500개)

## CoreML 배포 장점

### 비용 & 성능
- **운영 비용**: $0/월 (영구 무료)
- **응답 속도**: 10-30ms (서버 대비 10배 빠름)
- **확장성**: 사용자 증가해도 비용/성능 동일

### 개인정보 & 보안
- **데이터 처리**: 100% 디바이스 내
- **서버 전송**: 없음
- **GDPR/개인정보법**: 준수 용이
- **App Store 심사**: 우호적

### 모델 업데이트
- **기본**: 앱 번들 모델
- **선택**: 원격 모델 다운로드 (앱 업데이트 불필요)
- **전략**: 하이브리드 (번들 + 원격)

## 다음 단계: Safari 확장 통합

CoreML 모델 완료 후:
1. Swift 프로젝트에 `BlogAIDetector.mlmodel` 추가
2. JavaScript(Content Script) → Swift(Message Handler) → CoreML 예측
3. 예측 결과 → UI 배지 표시

**상세 문서**: [docs/project-plan.md](docs/project-plan.md)

---

**핵심 전략**: CoreML 온디바이스 추론으로 **비용 $0, 최고 성능, 개인정보 보호** 달성
