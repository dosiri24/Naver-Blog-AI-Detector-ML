# 네이버 블로그 데이터 수집

네이버 통합검색 블로그 탭에서 검색 결과를 자동으로 수집하는 스크래퍼입니다.

## 🎯 목표

- 키워드 기반 블로그 전체 글 수집
- 최소 500자 이상의 본문 확보
- AI/HUMAN 라벨링을 위한 전체 맥락 제공

## 📊 테스트 결과

**키워드**: "도쿄돔호텔 후기"

```
총 블로그 수: 30개
평균 길이: 4,324자
최소 길이: 2,153자
최대 길이: 8,786자

길이별 분포:
  2,000-3,000자:  4개 ████
  3,000-4,000자: 12개 ████████████
  4,000-5,000자:  7개 ███████
  5,000-10,000자:  7개 ███████
```

**소요 시간**: 약 3분 19초 (30개 블로그)

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 키워드 설정

`data/keywords.json` 파일을 편집:

```json
{
  "test_keywords": [
    "도쿄돔호텔 후기"
  ],
  "ai_keywords": [
    "파이썬 완벽 가이드",
    "서울 맛집 총정리"
  ],
  "human_keywords": [
    "제주도 여행 솔직 후기",
    "맛집 내돈내산"
  ]
}
```

### 3. 스크래핑 실행

```bash
# 테스트 실행 (test_keywords 사용)
python scripts/scraper.py

# 데이터 분석
python scripts/analyze_data.py
```

## 📁 디렉토리 구조

```
Data-Collection/
├── data/
│   ├── raw/
│   │   └── blogs.json          # 수집된 원본 데이터
│   ├── labeled/                 # LLM 라벨링 결과 (향후)
│   ├── processed/               # 전처리된 데이터 (향후)
│   └── keywords.json            # 검색 키워드 설정
├── logs/
│   └── scraper_YYYYMMDD_HHMMSS.log  # 실행 로그
├── scripts/
│   ├── scraper.py              # 메인 스크래퍼
│   ├── analyze_data.py         # 데이터 분석
│   └── debug_scraper.py        # 디버깅 도구
├── requirements.txt             # Python 의존성
└── README.md                    # 본 파일
```

## 💾 데이터 형식

`data/raw/blogs.json`:

```json
[
  {
    "blog_id": "blog_0000",
    "url": "https://blog.naver.com/username/1234567890",
    "title": "블로그 제목",
    "full_text": "전체 본문 내용...",
    "keyword": "검색 키워드",
    "scraped_at": "2025-10-16T16:02:00.000000"
  }
]
```

## 🛡️ 봇 감지 회피 전략

스크래퍼는 다음 전략을 사용하여 안전하게 작동합니다:

- ✅ User-Agent 설정 (실제 브라우저 흉내)
- ✅ `webdriver` 속성 제거
- ✅ 랜덤 지연 (2-4초 페이지 로딩, 1-3초 블로그 간)
- ✅ 증분 저장 (10개마다 자동 저장)
- ✅ 에러 처리 및 재시도 (최대 3회)

## 📈 성능

| 항목 | 수치 |
|------|------|
| 블로그당 평균 시간 | 6-7초 |
| 30개 수집 시간 | ~3분 20초 |
| 1,500개 예상 시간 | ~2.5-3시간 |
| 성공률 | 100% (30/30) |

## 🔧 주요 기능

### `scraper.py`

**함수**:
- `setup_driver()`: Selenium WebDriver 설정 (봇 감지 회피)
- `get_search_url()`: 검색 URL 생성
- `extract_blog_urls()`: 검색 결과에서 블로그 URL 추출
- `extract_blog_content()`: 개별 블로그 본문 크롤링
- `should_exclude()`: 필터링 로직 (길이, 중복, 빈 본문)
- `save_incremental()`: 증분 저장
- `scrape_all_blogs()`: 전체 워크플로우 실행

**특징**:
- 정규표현식 기반 URL 패턴 매칭
- iframe 자동 처리 (구형/신형 에디터 지원)
- 중복 URL 제거
- 실시간 진행 상황 표시 (tqdm)

### `analyze_data.py`

수집된 데이터의 통계 분석:
- 총 개수, 평균/최소/최대 길이
- 길이별 분포 시각화
- 샘플 데이터 표시

## 🐛 문제 해결

### CSS 선택자 변경

네이버 검색 결과 페이지 구조가 변경되면 URL 추출이 실패할 수 있습니다.

**해결 방법**:
1. `debug_scraper.py` 실행하여 HTML 구조 확인:
   ```bash
   python scripts/debug_scraper.py
   ```
2. `logs/search_page.html` 파일에서 실제 구조 분석
3. `scraper.py`의 `extract_blog_urls()` 함수 수정

### 봇 감지 발생

캡차가 나타나거나 접근이 차단되면:

**해결 방법**:
1. 지연 시간 증가 (`random.uniform(3, 5)`)
2. User-Agent 변경
3. 하루 수집량 제한 (500개 이하)
4. VPN 사용 (최후의 수단)

### iframe 전환 실패

일부 블로그에서 본문 추출이 안 되면:

**해결 방법**:
- `extract_blog_content()` 함수가 이미 자동 처리
- fallback으로 iframe 없는 페이지도 지원

## 🔜 다음 단계

1. **LLM 라벨링** (`scripts/labeler.py`)
   - Claude 3.5 Haiku 또는 GPT-4o-mini
   - AI/HUMAN + confidence + reasoning

2. **스니펫 증강** (`scripts/preprocess.py`)
   - 각 블로그 → 3-5개 스니펫 (100-300자)
   - Train/Test 분할 (80/20)

3. **모델 학습** (`scripts/train.py`)
   - TF-IDF + Logistic Regression 또는
   - 임베딩 + 신경망

## 📝 라이센스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 🙏 크레딧

- **웹 스크래핑**: Selenium + BeautifulSoup
- **진행 표시**: tqdm
- **ChromeDriver 관리**: webdriver-manager

---

**작성일**: 2025-10-16
**버전**: 1.0.0
**상태**: ✅ 테스트 완료
