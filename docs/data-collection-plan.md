# 블로그 스크래핑 데이터셋 구축 계획서

> 네이버 통합검색 블로그 탭 스크래핑을 통한 1,500개 블로그 글 수집

---

## 목표

**수집 목표**: 1,500개 네이버 블로그 **전체 글 본문**
- 키워드 50개 × 30개/키워드 = 1,500개
- AI 작성 글: 750개 (50%)
- 인간 작성 글: 750개 (50%)
- 최소 길이: 500자 이상

**수집 방식**: 네이버 통합검색 블로그 탭 웹 스크래핑
- API 불필요
- 봇 감지 회피 전략 적용
- 사용자 제공 키워드 리스트 사용

---

## 1. 기술 스택

### 최종 선택: Selenium + BeautifulSoup

**선택 이유**:
1. ✅ 네이버 검색 결과는 JavaScript 렌더링 필요
2. ✅ Selenium으로 실제 브라우저처럼 동작 (봇 감지 회피)
3. ✅ BeautifulSoup으로 HTML 파싱
4. ✅ API 키 불필요

**라이브러리**:
```python
selenium          # 브라우저 자동화
beautifulsoup4    # HTML 파싱
webdriver-manager # ChromeDriver 자동 관리
```

---

## 2. 네이버 통합검색 구조 분석

### 2.1 검색 URL 형식

**기본 형식**:
```
https://search.naver.com/search.naver?ssc=tab.blog.all&query={키워드}
```

**예시**:
```
https://search.naver.com/search.naver?ssc=tab.blog.all&query=도쿄돔호텔+후기
https://search.naver.com/search.naver?ssc=tab.blog.all&query=파이썬+입문
```

**URL 파라미터**:
- `ssc=tab.blog.all`: 블로그 탭
- `query={키워드}`: 검색어 (URL 인코딩)
- `start={번호}`: 페이지네이션 (옵션, 1부터 시작)

### 2.2 검색 결과 HTML 구조

**블로그 카드 구조** (2024년 기준):
```html
<div class="detail_box">
    <!-- 제목 -->
    <a class="title_link" href="https://blog.naver.com/...">
        블로그 제목
    </a>

    <!-- 미리보기 -->
    <a class="dsc_link" href="https://blog.naver.com/...">
        본문 미리보기...
    </a>

    <!-- 블로그명 -->
    <a class="name" href="...">블로그명</a>

    <!-- 날짜 -->
    <span class="date">2025.01.16.</span>
</div>
```

**CSS 선택자**:
- 블로그 카드: `.detail_box` 또는 `.view_wrap`
- 제목: `.title_link`
- 미리보기: `.dsc_link`
- 블로그 URL: `.title_link['href']`

---

## 3. 키워드 전략

### 3.1 키워드 입력 방식

**사용자가 키워드 리스트 제공** (`data/keywords.json`):
```json
{
  "ai_keywords": [
    "파이썬 완벽 가이드",
    "서울 맛집 총정리",
    "영어 공부 방법 10가지",
    "다이어트 팁 모음",
    "노트북 추천 BEST 5"
  ],
  "human_keywords": [
    "제주도 여행 솔직 후기",
    "맛집 내돈내산",
    "육아 일기",
    "오늘의 일상",
    "진짜 맛있는 떡볶이"
  ]
}
```

**또는 텍스트 파일** (`data/keywords.txt`):
```
# AI 키워드
파이썬 완벽 가이드
서울 맛집 총정리

# Human 키워드
제주도 여행 솔직 후기
맛집 내돈내산
```

### 3.2 키워드 개수 계산

**목표**: 1,500개 블로그 글
```
50개 키워드 × 30개/키워드 = 1,500개

AI 키워드: 25개 × 30 = 750개
Human 키워드: 25개 × 30 = 750개
```

---

## 4. 크롤링 워크플로우

### 4.1 3단계 크롤링 전략

**1단계: 키워드별 검색 결과 페이지 접속**
```python
def get_search_url(keyword: str) -> str:
    """네이버 블로그 검색 URL 생성"""
    from urllib.parse import quote
    encoded_keyword = quote(keyword)
    return f"https://search.naver.com/search.naver?ssc=tab.blog.all&query={encoded_keyword}"
```

**2단계: 검색 결과에서 상위 30개 블로그 URL 추출**
```python
def extract_blog_urls(driver, keyword: str, max_count: int = 30) -> list:
    """
    검색 결과에서 블로그 URL 추출

    Args:
        driver: Selenium WebDriver
        keyword: 검색 키워드
        max_count: 수집할 최대 개수 (기본 30)

    Returns:
        블로그 URL 리스트
    """
    search_url = get_search_url(keyword)
    driver.get(search_url)

    # 페이지 로딩 대기
    time.sleep(random.uniform(2, 4))

    # BeautifulSoup으로 파싱
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # 블로그 링크 추출
    blog_cards = soup.select('.detail_box, .view_wrap')
    urls = []

    for card in blog_cards[:max_count]:
        title_link = card.select_one('.title_link')
        if title_link and title_link.get('href'):
            urls.append(title_link['href'])

    return urls
```

**3단계: 개별 블로그 페이지에서 본문 추출**
```python
def extract_blog_content(driver, url: str) -> dict:
    """
    블로그 본문 추출

    Args:
        url: 블로그 URL

    Returns:
        {'title': str, 'full_text': str}
    """
    driver.get(url)
    time.sleep(random.uniform(1, 3))

    # iframe 처리 (네이버 블로그)
    driver.switch_to.frame('mainFrame')

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # 제목 추출
    title = ""
    title_elem = soup.select_one('.se-title-text, .pcol1')
    if title_elem:
        title = title_elem.get_text(strip=True)

    # 본문 추출 (신형 에디터)
    content = ""
    content_elems = soup.select('.se-main-container .se-text')
    if content_elems:
        content = '\n'.join([elem.get_text(strip=True) for elem in content_elems])
    else:
        # 구형 에디터
        old_content = soup.select_one('#postViewArea')
        if old_content:
            content = old_content.get_text(strip=True)

    driver.switch_to.default_content()

    return {
        'title': title,
        'full_text': content
    }
```

### 4.2 전체 워크플로우

```python
def scrape_all_blogs(keywords: list, blogs_per_keyword: int = 30):
    """
    전체 블로그 스크래핑

    Args:
        keywords: 키워드 리스트
        blogs_per_keyword: 키워드당 수집할 블로그 수
    """
    driver = setup_driver()
    all_blogs = []

    for keyword in keywords:
        logging.info(f"키워드 '{keyword}' 검색 시작")

        # 1. 검색 결과에서 URL 추출
        blog_urls = extract_blog_urls(driver, keyword, blogs_per_keyword)
        logging.info(f"URL {len(blog_urls)}개 수집")

        # 2. 각 블로그 본문 추출
        for url in blog_urls:
            try:
                content = extract_blog_content(driver, url)

                # 필터링
                if len(content['full_text']) < 500:
                    continue

                blog_data = {
                    'blog_id': f"blog_{len(all_blogs):04d}",
                    'url': url,
                    'title': content['title'],
                    'full_text': content['full_text'],
                    'keyword': keyword,
                    'scraped_at': datetime.now().isoformat()
                }

                all_blogs.append(blog_data)

                # 봇 감지 회피 대기
                time.sleep(random.uniform(1, 3))

            except Exception as e:
                logging.error(f"에러: {url} - {e}")
                continue

        # 키워드 간 대기
        time.sleep(random.uniform(5, 10))

    driver.quit()
    return all_blogs
```

---

## 5. 봇 감지 회피 전략

### 5.1 Selenium 설정

**Headless 모드 + 봇 감지 회피**:
```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def setup_driver():
    """봇 감지 회피 설정이 적용된 Selenium 드라이버"""
    options = Options()

    # Headless 모드 (선택)
    # options.add_argument('--headless')

    # 봇 감지 회피 옵션
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    # User-Agent 설정
    options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    # 창 크기
    options.add_argument('--window-size=1920,1080')

    # 기타 옵션
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    # 드라이버 생성
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # JavaScript로 webdriver 속성 제거
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        '''
    })

    return driver
```

### 5.2 랜덤 지연 전략

**지연 시간 설정**:
```python
import random
import time

# 페이지 로딩 후 대기
time.sleep(random.uniform(2, 4))  # 2-4초 랜덤

# 블로그 본문 추출 후 대기
time.sleep(random.uniform(1, 3))  # 1-3초 랜덤

# 키워드 간 대기
time.sleep(random.uniform(5, 10))  # 5-10초 랜덤
```

**권장 전략**:
- **검색 페이지 로딩**: 2-4초 대기
- **블로그 본문 로딩**: 1-3초 대기
- **블로그 간 이동**: 1-3초 대기
- **키워드 간 이동**: 5-10초 대기

**예상 총 시간**:
```
50개 키워드 × (30개 블로그 × 2초 + 10초) = 약 1시간 40분
```

### 5.3 추가 회피 전략

**1. User-Agent 로테이션**:
```python
user_agents = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...',
    'Mozilla/5.0 (X11; Linux x86_64)...'
]

# 키워드마다 User-Agent 변경
options.add_argument(f'user-agent={random.choice(user_agents)}')
```

**2. 요청 헤더 추가**:
```python
# Selenium은 자동으로 일반 브라우저 헤더 전송
# 추가 헤더는 불필요하지만, 필요 시:
driver.execute_cdp_cmd('Network.setExtraHTTPHeaders', {
    'headers': {
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Accept-Encoding': 'gzip, deflate, br'
    }
})
```

**3. 마우스 이동 시뮬레이션** (고급):
```python
from selenium.webdriver.common.action_chains import ActionChains

# 페이지 스크롤
driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
time.sleep(random.uniform(0.5, 1.5))
```

---

## 6. 데이터 품질 관리

### 6.1 필터링 기준

**제외 대상**:
```python
def should_exclude(blog_data: dict) -> bool:
    """블로그 글 제외 여부 판단"""
    # 1. 길이 제한
    if len(blog_data['full_text']) < 500:
        return True  # 너무 짧음

    # 2. 중복 URL
    if blog_data['url'] in seen_urls:
        return True

    # 3. 빈 본문
    if not blog_data['full_text'].strip():
        return True

    # 4. 광고/스팸 (키워드 필터)
    spam_keywords = ['광고', '협찬', '제휴', '리뷰이벤트', '체험단']
    title_lower = blog_data['title'].lower()
    if any(keyword in title_lower for keyword in spam_keywords):
        return True

    return False
```

### 6.2 중복 제거 전략

**URL 기준 중복 제거**:
```python
seen_urls = set()

def is_duplicate(url: str) -> bool:
    """중복 URL 확인"""
    if url in seen_urls:
        return True
    seen_urls.add(url)
    return False
```

**왜 중복이 발생하나?**:
- 같은 블로그 글이 여러 키워드 검색에서 상위 노출
- 해결: URL 기준 중복 제거

---

## 7. 에러 처리

### 7.1 예상 에러 및 대응

```python
def scrape_blog_safely(driver, url: str) -> dict | None:
    """안전한 블로그 크롤링"""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            driver.get(url)
            time.sleep(random.uniform(2, 4))

            # iframe 전환
            driver.switch_to.frame('mainFrame')

            # 본문 추출
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            content = extract_content(soup)

            driver.switch_to.default_content()

            if not content['full_text']:
                logging.warning(f"빈 본문: {url}")
                return None

            return content

        except NoSuchFrameException:
            # iframe 없음 (일반 페이지)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            content = extract_content(soup)
            return content

        except TimeoutException:
            logging.error(f"타임아웃 (시도 {attempt+1}/{max_retries}): {url}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None

        except Exception as e:
            logging.error(f"에러: {url} - {e}")
            return None

    return None
```

### 7.2 에러 로깅

```python
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.log'),
        logging.StreamHandler()
    ]
)

# 사용
logging.info(f"키워드 '{keyword}' 검색 시작")
logging.warning(f"본문 추출 실패: {url}")
logging.error(f"크리티컬 에러: {error}")
```

---

## 8. 데이터 저장

### 8.1 JSON 스키마

**`data/raw/blogs.json`**:
```json
[
  {
    "blog_id": "blog_0001",
    "url": "https://blog.naver.com/example/123456",
    "title": "제주도 여행 후기",
    "full_text": "지난 주말에 제주도를 다녀왔습니다...",
    "keyword": "제주도 여행 후기",
    "scraped_at": "2025-01-16T10:30:00"
  }
]
```

### 8.2 증분 저장

**진행 상황 보존**:
```python
def save_incremental(blogs: list, filename: str = 'data/raw/blogs.json'):
    """증분 저장 (덮어쓰기)"""
    import json

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(blogs, f, ensure_ascii=False, indent=2)

    logging.info(f"진행 상황 저장: {len(blogs)}개")

# 10개마다 저장
if len(all_blogs) % 10 == 0:
    save_incremental(all_blogs)
```

---

## 9. 진행 상황 추적

### 9.1 tqdm 진행 바

```python
from tqdm import tqdm

# 키워드별 진행
for keyword in tqdm(keywords, desc="키워드 처리"):
    blog_urls = extract_blog_urls(driver, keyword, 30)

    # 블로그별 진행
    for url in tqdm(blog_urls, desc=f"'{keyword}' 블로그", leave=False):
        content = scrape_blog_safely(driver, url)
        # ...
```

**출력 예시**:
```
키워드 처리: 100%|██████████| 50/50 [1:40:00<00:00,  2.00min/keyword]
  '제주도 여행': 100%|██████████| 30/30 [01:00<00:00,  2.00s/blog]
```

### 9.2 실시간 통계

```python
stats = {
    'total_keywords': 0,
    'urls_collected': 0,
    'blogs_scraped': 0,
    'blogs_filtered': 0,
    'errors': 0
}

def print_stats():
    """통계 출력"""
    print("\n=== 수집 통계 ===")
    print(f"처리 키워드: {stats['total_keywords']}/{len(keywords)}")
    print(f"URL 수집: {stats['urls_collected']}개")
    print(f"크롤링 완료: {stats['blogs_scraped']}개")
    print(f"필터링됨: {stats['blogs_filtered']}개")
    print(f"에러: {stats['errors']}개")
```

---

## 10. 구현 체크리스트

### Phase 1: 환경 설정
- [ ] Python 가상환경 생성
- [ ] `requirements.txt` 작성 및 설치
- [ ] ChromeDriver 설치 (webdriver-manager 자동 처리)
- [ ] 디렉토리 구조 생성 (`data/raw`, `logs`)

### Phase 2: 키워드 준비
- [ ] AI 키워드 25개 작성
- [ ] Human 키워드 25개 작성
- [ ] `data/keywords.json` 저장

### Phase 3: 크롤러 구현
- [ ] `scraper.py` 기본 구조
- [ ] Selenium 드라이버 설정 (봇 감지 회피)
- [ ] 검색 결과 URL 추출 함수
- [ ] 블로그 본문 추출 함수
- [ ] 랜덤 지연 적용
- [ ] 에러 처리 로직

### Phase 4: 데이터 품질 관리
- [ ] 필터링 로직 (500자 미만, 중복, 스팸)
- [ ] 증분 저장 기능
- [ ] 로깅 시스템

### Phase 5: 테스트 및 실행
- [ ] 테스트 실행 (키워드 2개, 10개/키워드)
- [ ] 전체 실행 (키워드 50개, 30개/키워드)
- [ ] 데이터 검증 (샘플 100개)
- [ ] AI/Human 비율 확인

---

## 11. 실행 순서

### 11.1 환경 설정

```bash
# 1. 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 디렉토리 생성
mkdir -p data/raw data/labeled data/processed logs
```

### 11.2 키워드 준비

**`data/keywords.json` 생성**:
```json
{
  "ai_keywords": [
    "파이썬 완벽 가이드",
    "서울 맛집 총정리",
    ...
  ],
  "human_keywords": [
    "제주도 여행 솔직 후기",
    "맛집 내돈내산",
    ...
  ]
}
```

### 11.3 실행

```bash
# 테스트 실행 (2개 키워드)
python scripts/scraper.py --test --keywords 2

# 전체 실행 (50개 키워드 × 30개)
python scripts/scraper.py --keywords-file data/keywords.json --per-keyword 30

# 검증
python scripts/validate_data.py
```

---

## 12. 예상 소요 시간

### 시간 계산

**검색 결과 페이지 로딩**:
```
50개 키워드 × 3초/키워드 = 150초 (2.5분)
```

**블로그 본문 크롤링**:
```
1,500개 URL × 2초/URL = 3,000초 (50분)
```

**키워드 간 대기**:
```
50개 키워드 × 7.5초/키워드 = 375초 (6분)
```

**필터링 및 저장**:
```
약 3분
```

**총 예상 시간**: **약 1시간 ~ 1시간 40분**

**개발 시간**:
- 크롤러 구현: 4-6시간
- 테스트 및 디버깅: 2-3시간
- 키워드 준비: 1시간
- **총 개발 시간**: **1일**

---

## 13. 키워드 예시 (50개)

### AI 의심 키워드 (25개)

**정보 정리형**:
1. 파이썬 완벽 가이드
2. 서울 맛집 총정리
3. 영어 공부 방법 10가지
4. 다이어트 팁 모음
5. 블로그 만들기 가이드

**리스트형**:
6. 부산 여행지 TOP 10
7. 노트북 추천 BEST 5
8. 영화 추천 리스트
9. 카페 추천 BEST
10. 책 추천 리스트

**구조화형**:
11. 초보자를 위한 주식 투자
12. 운동 루틴 완벽 정리
13. 요리 레시피 모음
14. 여행 계획 세우는 법
15. 재테크 방법 정리

**기타 AI 의심**:
16. 효율적인 시간 관리
17. 건강한 식단 구성
18. 스마트폰 활용 팁
19. 인테리어 아이디어 모음
20. SNS 마케팅 전략
21. 블로그 수익화 방법
22. 창업 아이템 추천
23. 자기계발 도서 추천
24. 생산성 향상 앱
25. 온라인 강의 추천

### Human 의심 키워드 (25개)

**개인 후기**:
1. 제주도 여행 솔직 후기
2. 맛집 내돈내산 리뷰
3. 노트북 실제 사용 후기
4. 호텔 숙박 후기
5. 영화 관람 후기

**일상**:
6. 육아 일기
7. 오늘의 일상
8. 주말 나들이
9. 일상 브이로그
10. 하루 루틴

**감정 표현**:
11. 진짜 맛있는 떡볶이
12. 너무 좋았던 카페
13. 최악이었던 영화
14. 감동적이었던 책
15. 힐링되는 장소

**개인 경험**:
16. 첫 해외여행
17. 이사 후기
18. 취업 준비 일기
19. 다이어트 도전기
20. 운동 시작 후기

**기타 Human 의심**:
21. 우리 강아지 일상
22. 요리 실패담
23. 집들이 후기
24. 데이트 코스 추천
25. 동네 맛집 발견

---

## 14. 문제 해결 가이드

### Q1: Selenium이 너무 느림
**해결**:
- Headless 모드 활성화
- 이미지 로딩 비활성화
- CSS/JavaScript 로딩 최소화

```python
options.add_argument('--blink-settings=imagesEnabled=false')
```

### Q2: 봇 감지됨 (캡차 발생)
**해결**:
- 지연 시간 증가 (3-5초)
- User-Agent 변경
- VPN 사용 (최후의 수단)
- 하루 수집량 제한 (500개 이하)

### Q3: iframe 전환 실패
**해결**:
```python
try:
    driver.switch_to.frame('mainFrame')
except NoSuchFrameException:
    # iframe 없는 페이지
    pass
```

### Q4: 본문 추출 실패 (빈 텍스트)
**해결**:
- CSS 선택자 확인 (개발자 도구)
- 여러 선택자 시도 (fallback)
- 페이지 로딩 대기 시간 증가

### Q5: 중복 URL 많음
**해결**:
- `seen_urls` set으로 중복 제거
- 키워드 다양화
- 검색 결과 페이지 확장 (2페이지까지)

---

## 15. 최종 산출물

**`data/raw/blogs.json`**:
```json
{
  "metadata": {
    "total_count": 1500,
    "ai_keywords_count": 25,
    "human_keywords_count": 25,
    "average_length": 2340,
    "collection_date": "2025-01-16",
    "collection_duration_minutes": 100
  },
  "blogs": [
    {
      "blog_id": "blog_0001",
      "url": "https://blog.naver.com/...",
      "title": "제목",
      "full_text": "본문",
      "keyword": "검색 키워드",
      "scraped_at": "2025-01-16T10:30:00"
    }
    // ... 1,500개
  ]
}
```

---

## 16. requirements.txt

```txt
selenium>=4.15.0
beautifulsoup4>=4.12.0
webdriver-manager>=4.0.0
lxml>=4.9.0
tqdm>=4.66.0
```

---

**다음 단계**: LLM API 라벨링 (`scripts/labeler.py`)

**핵심 변경사항**:
- ✅ 네이버 검색 API → 웹 스크래핑
- ✅ 사용자 제공 키워드 리스트
- ✅ 키워드당 상위 30개 수집
- ✅ 봇 감지 회피 전략 적용
- ✅ 예상 수집 시간: 1-1.5시간
