"""
네이버 블로그 스크래퍼
네이버 통합검색 블로그 탭에서 검색 결과를 수집하고 전체 본문을 추출합니다.
"""

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import (
    NoSuchFrameException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager


# 로깅 설정
def setup_logging(log_dir: Path) -> None:
    """로깅 설정

    Args:
        log_dir: 로그 파일 디렉토리
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def setup_driver() -> webdriver.Chrome:
    """봇 감지 회피 설정이 적용된 Selenium 드라이버 생성

    Returns:
        설정된 Chrome WebDriver
    """
    options = Options()

    # 봇 감지 회피 옵션
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)

    # User-Agent 설정
    options.add_argument(
        'user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )

    # 창 크기
    options.add_argument('--window-size=1920,1080')

    # 기타 옵션
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    # 헤드리스 모드 (테스트 시에는 주석 처리 권장)
    # options.add_argument('--headless')

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

    logging.info("Selenium 드라이버 설정 완료")
    return driver


def get_search_url(keyword: str) -> str:
    """네이버 블로그 검색 URL 생성

    Args:
        keyword: 검색 키워드

    Returns:
        네이버 블로그 검색 URL
    """
    encoded_keyword = quote(keyword)
    return f"https://search.naver.com/search.naver?ssc=tab.blog.all&query={encoded_keyword}"


def extract_blog_urls(driver: webdriver.Chrome, keyword: str, max_count: int = 30) -> List[str]:
    """검색 결과에서 블로그 URL 추출

    Args:
        driver: Selenium WebDriver
        keyword: 검색 키워드
        max_count: 수집할 최대 개수

    Returns:
        블로그 URL 리스트
    """
    search_url = get_search_url(keyword)
    logging.info(f"검색 URL 접속: {search_url}")

    driver.get(search_url)

    # 페이지 로딩 대기
    wait_time = random.uniform(2, 4)
    logging.info(f"페이지 로딩 대기: {wait_time:.2f}초")
    time.sleep(wait_time)

    # BeautifulSoup으로 파싱
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # 블로그 링크 추출 (blog.naver.com을 포함한 모든 링크)
    all_links = soup.select('a[href*="blog.naver.com"]')

    urls = []
    seen_urls = set()

    for link in all_links:
        url = link.get('href', '')

        # 포스트 URL 패턴 확인 (username/post_id 형식)
        # 예: https://blog.naver.com/username/1234567890
        import re
        if re.match(r'https://blog\.naver\.com/[\w-]+/\d+', url):
            # 중복 제거
            if url not in seen_urls:
                urls.append(url)
                seen_urls.add(url)

                # 원하는 개수만큼 수집하면 종료
                if len(urls) >= max_count:
                    break

    logging.info(f"블로그 URL {len(urls)}개 추출")
    return urls


def extract_blog_content(driver: webdriver.Chrome, url: str) -> Optional[Dict[str, str]]:
    """블로그 본문 추출

    Args:
        driver: Selenium WebDriver
        url: 블로그 URL

    Returns:
        {'title': str, 'full_text': str} 또는 None (실패 시)
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            driver.get(url)

            # 페이지 로딩 대기
            wait_time = random.uniform(2, 4)
            time.sleep(wait_time)

            # iframe 처리 시도
            try:
                driver.switch_to.frame('mainFrame')
                logging.debug("iframe 전환 성공")
            except NoSuchFrameException:
                logging.debug("iframe 없음 - 일반 페이지")
                pass

            soup = BeautifulSoup(driver.page_source, 'html.parser')

            # 제목 추출 (여러 선택자 시도)
            title = ""
            for selector in ['.se-title-text', '.pcol1', '.se_title', '.tit_h3']:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break

            # 본문 추출
            content = ""

            # 신형 에디터 (스마트에디터 ONE)
            content_elems = soup.select('.se-main-container .se-text, .se-main-container .se-text-paragraph')
            if content_elems:
                content = '\n'.join([elem.get_text(strip=True) for elem in content_elems])
                logging.debug(f"신형 에디터 본문 추출: {len(content)}자")
            else:
                # 구형 에디터
                old_content = soup.select_one('#postViewArea, .post-view')
                if old_content:
                    content = old_content.get_text(strip=True)
                    logging.debug(f"구형 에디터 본문 추출: {len(content)}자")

            # iframe에서 빠져나오기
            try:
                driver.switch_to.default_content()
            except:
                pass

            # 본문이 비어있으면 None 반환
            if not content.strip():
                logging.warning(f"빈 본문: {url}")
                return None

            return {
                'title': title,
                'full_text': content
            }

        except TimeoutException:
            logging.error(f"타임아웃 (시도 {attempt + 1}/{max_retries}): {url}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None

        except Exception as e:
            logging.error(f"에러 발생 (시도 {attempt + 1}/{max_retries}): {url} - {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            return None

    return None


def should_exclude(blog_data: Dict, seen_urls: set, min_length: int = 500) -> bool:
    """블로그 글 제외 여부 판단

    Args:
        blog_data: 블로그 데이터
        seen_urls: 이미 수집한 URL 집합
        min_length: 최소 본문 길이

    Returns:
        True if should exclude, False otherwise
    """
    # 길이 제한
    if len(blog_data['full_text']) < min_length:
        logging.debug(f"제외: 너무 짧음 ({len(blog_data['full_text'])}자)")
        return True

    # 중복 URL
    if blog_data['url'] in seen_urls:
        logging.debug(f"제외: 중복 URL")
        return True

    # 빈 본문
    if not blog_data['full_text'].strip():
        logging.debug(f"제외: 빈 본문")
        return True

    return False


def load_existing_data(output_file: Path) -> tuple[List[Dict], Dict[str, Dict], Dict[str, Dict], int]:
    """기존 데이터 로드

    Args:
        output_file: 데이터 파일 경로

    Returns:
        (블로그 데이터 리스트, title → 블로그 데이터 딕셔너리, url → 블로그 데이터 딕셔너리, 마지막 blog_id 번호)
    """
    if not output_file.exists():
        return [], {}, {}, 0

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_blogs = json.load(f)

        # title과 url을 키로 하는 딕셔너리 생성
        title_map = {blog['title']: blog for blog in existing_blogs if 'title' in blog}
        url_map = {blog['url']: blog for blog in existing_blogs if 'url' in blog}

        # 마지막 blog_id 추출 (blog_XXXX 형식에서 숫자 추출)
        last_id = 0
        for blog in existing_blogs:
            if 'blog_id' in blog:
                try:
                    # "blog_0042" -> 42
                    id_num = int(blog['blog_id'].replace('blog_', ''))
                    last_id = max(last_id, id_num)
                except ValueError:
                    continue

        logging.info(f"기존 데이터 로드: {len(existing_blogs)}개 블로그 (마지막 ID: blog_{last_id:04d})")
        return existing_blogs, title_map, url_map, last_id

    except Exception as e:
        logging.error(f"기존 데이터 로드 실패: {e}")
        return [], {}, {}, 0


def save_with_deduplication(
    new_blogs: List[Dict],
    output_file: Path,
    existing_blogs: List[Dict] = None,
    title_map: Dict[str, Dict] = None,
    url_map: Dict[str, Dict] = None
) -> tuple[List[Dict], Dict[str, Dict], Dict[str, Dict], Dict[str, int]]:
    """중복 제거 및 저장

    title을 기준으로 중복을 체크하고, 중복 시 duplicate_count를 증가시킵니다.

    Args:
        new_blogs: 새로 수집한 블로그 데이터 리스트
        output_file: 출력 파일 경로
        existing_blogs: 기존 블로그 데이터 (None이면 파일에서 로드)
        title_map: title → 블로그 데이터 매핑 (None이면 생성)
        url_map: url → 블로그 데이터 매핑 (None이면 생성)

    Returns:
        (전체 블로그 리스트, title_map, url_map, 통계 딕셔너리)
    """
    # 기존 데이터 로드
    if existing_blogs is None or title_map is None or url_map is None:
        existing_blogs, title_map, url_map, _ = load_existing_data(output_file)

    stats = {
        'existing': len(existing_blogs),
        'new': 0,
        'duplicates': 0
    }

    # 새 데이터 처리
    for blog in new_blogs:
        title = blog.get('title', '')
        url = blog.get('url', '')

        if not title:
            continue

        if title in title_map:
            # 중복: duplicate_count 증가
            existing_blog = title_map[title]
            if 'duplicate_count' not in existing_blog:
                existing_blog['duplicate_count'] = 0
            existing_blog['duplicate_count'] += 1
            existing_blog['last_seen_at'] = datetime.now().isoformat()
            stats['duplicates'] += 1

            logging.debug(f"중복 발견: '{title[:30]}...' (총 {existing_blog['duplicate_count']}회)")
        else:
            # 새 데이터: 추가
            blog['duplicate_count'] = 0
            blog['last_seen_at'] = blog.get('scraped_at', datetime.now().isoformat())
            existing_blogs.append(blog)
            title_map[title] = blog
            if url:
                url_map[url] = blog
            stats['new'] += 1

    # 파일 저장
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_blogs, f, ensure_ascii=False, indent=2)

    logging.info(
        f"저장 완료: 총 {len(existing_blogs)}개 "
        f"(기존: {stats['existing']}, 신규: {stats['new']}, 중복: {stats['duplicates']}) "
        f"→ {output_file}"
    )

    return existing_blogs, title_map, url_map, stats


def scrape_all_blogs(
    keywords: List[str],
    blogs_per_keyword: int = 30,
    output_file: Path = None,
    min_length: int = 500,
    existing_blogs: List[Dict] = None,
    title_map: Dict[str, Dict] = None,
    url_map: Dict[str, Dict] = None,
    last_blog_id: int = 0
) -> List[Dict]:
    """전체 블로그 스크래핑

    Args:
        keywords: 키워드 리스트
        blogs_per_keyword: 키워드당 수집할 블로그 수
        output_file: 출력 파일 경로
        min_length: 최소 본문 길이
        existing_blogs: 기존 블로그 데이터 (중복 방지용)
        title_map: title → 블로그 데이터 매핑 (중복 방지용)
        url_map: url → 블로그 데이터 매핑 (중복 방지용)
        last_blog_id: 기존 데이터의 마지막 blog_id 번호

    Returns:
        수집된 블로그 데이터 리스트
    """
    driver = setup_driver()
    all_blogs = []

    # url_map이 없으면 빈 딕셔너리로 초기화
    if url_map is None:
        url_map = {}

    # 다음 blog_id 초기화 (기존 데이터의 마지막 ID + 1)
    next_blog_id = last_blog_id + 1

    # 통계에 URL 중복 카운트 추가
    logging.info(f"기존 URL {len(url_map)}개 로드 (URL 중복 체크용)")
    logging.info(f"다음 blog_id: blog_{next_blog_id:04d}")

    stats = {
        'total_keywords': len(keywords),
        'urls_collected': 0,
        'url_duplicates': 0,
        'blogs_scraped': 0,
        'blogs_filtered': 0,
        'errors': 0
    }

    try:
        # 키워드별 진행
        for keyword_idx, keyword in enumerate(tqdm(keywords, desc="키워드 처리")):
            logging.info(f"\n{'='*50}")
            logging.info(f"키워드 [{keyword_idx + 1}/{len(keywords)}]: '{keyword}'")
            logging.info(f"{'='*50}")

            try:
                # 1. 검색 결과에서 URL 추출
                blog_urls = extract_blog_urls(driver, keyword, blogs_per_keyword)
                stats['urls_collected'] += len(blog_urls)

                if not blog_urls:
                    logging.warning(f"검색 결과 없음: {keyword}")
                    continue

                # 2. 각 블로그 본문 추출
                for url_idx, url in enumerate(tqdm(blog_urls, desc=f"'{keyword}' 블로그", leave=False)):
                    try:
                        # URL 중복 체크 (스크랩 전)
                        if url in url_map:
                            # 중복 URL 발견: duplicate_count 증가
                            existing_blog = url_map[url]
                            if 'duplicate_count' not in existing_blog:
                                existing_blog['duplicate_count'] = 0
                            existing_blog['duplicate_count'] += 1
                            existing_blog['last_seen_at'] = datetime.now().isoformat()
                            stats['url_duplicates'] += 1

                            logging.debug(f"⊗ URL 중복 스킵: {url} (총 {existing_blog['duplicate_count']}회)")
                            continue

                        # 본문 추출
                        content = extract_blog_content(driver, url)

                        if not content:
                            stats['errors'] += 1
                            continue

                        # 블로그 데이터 구성
                        blog_data = {
                            'blog_id': f"blog_{next_blog_id:04d}",
                            'url': url,
                            'title': content['title'],
                            'full_text': content['full_text'],
                            'keyword': keyword,
                            'scraped_at': datetime.now().isoformat()
                        }

                        # 필터링 (길이 제한만 체크)
                        if len(blog_data['full_text']) < min_length:
                            stats['blogs_filtered'] += 1
                            logging.debug(f"제외: 너무 짧음 ({len(blog_data['full_text'])}자)")
                            continue

                        if not blog_data['full_text'].strip():
                            stats['blogs_filtered'] += 1
                            logging.debug(f"제외: 빈 본문")
                            continue

                        # 수집
                        all_blogs.append(blog_data)
                        url_map[url] = blog_data  # url_map에 추가 (메모리 내)
                        next_blog_id += 1  # blog_id 증가
                        stats['blogs_scraped'] += 1

                        logging.info(
                            f"✓ [{len(all_blogs)}] {blog_data['title'][:30]}... "
                            f"({len(content['full_text'])}자)"
                        )

                        # 봇 감지 회피 대기
                        wait_time = random.uniform(1, 3)
                        time.sleep(wait_time)

                    except Exception as e:
                        logging.error(f"블로그 추출 에러: {url} - {e}")
                        stats['errors'] += 1
                        continue

                # 키워드 처리 완료 후 저장
                if output_file and all_blogs:
                    logging.info(f"\n키워드 '{keyword}' 처리 완료 - 저장 중...")
                    existing_blogs, title_map, url_map, keyword_stats = save_with_deduplication(
                        all_blogs, output_file, existing_blogs, title_map, url_map
                    )
                    logging.info(f"키워드 '{keyword}': 신규 {keyword_stats['new']}개, 중복 {keyword_stats['duplicates']}개")
                    # 저장 후 all_blogs 초기화 (메모리 효율성)
                    all_blogs = []

                # 키워드 간 대기
                if keyword_idx < len(keywords) - 1:
                    wait_time = random.uniform(5, 10)
                    logging.info(f"다음 키워드 전 대기: {wait_time:.2f}초")
                    time.sleep(wait_time)

            except Exception as e:
                logging.error(f"키워드 처리 에러: {keyword} - {e}")
                stats['errors'] += 1
                continue

    finally:
        driver.quit()
        logging.info("드라이버 종료")

    # 최종 통계 출력
    logging.info(f"\n{'='*50}")
    logging.info("=== 수집 통계 ===")
    logging.info(f"처리 키워드: {stats['total_keywords']}")
    logging.info(f"URL 수집: {stats['urls_collected']}개")
    logging.info(f"URL 중복 스킵: {stats['url_duplicates']}개")
    logging.info(f"크롤링 완료: {stats['blogs_scraped']}개")
    logging.info(f"필터링됨: {stats['blogs_filtered']}개")
    logging.info(f"에러: {stats['errors']}개")
    logging.info(f"{'='*50}\n")

    return all_blogs


def main():
    """메인 실행 함수"""
    # 경로 설정
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    log_dir = base_dir / 'logs'

    # 로깅 설정
    setup_logging(log_dir)

    logging.info("="*50)
    logging.info("네이버 블로그 스크래퍼 시작")
    logging.info("="*50)

    # 키워드 로드
    keywords_file = data_dir / 'keywords.json'

    if not keywords_file.exists():
        logging.error(f"키워드 파일 없음: {keywords_file}")
        return

    with open(keywords_file, 'r', encoding='utf-8') as f:
        keywords_data = json.load(f)

    # 테스트 키워드 사용
    keywords = keywords_data.get('test_keywords', [])

    if not keywords:
        logging.error("키워드가 없습니다.")
        return

    logging.info(f"키워드 {len(keywords)}개 로드: {keywords}")

    # 출력 파일 경로
    output_file = data_dir / 'blogs.json'

    # 기존 데이터 로드 (중복 방지용)
    logging.info("\n기존 데이터 확인 중...")
    existing_blogs, title_map, url_map, last_blog_id = load_existing_data(output_file)

    # 스크래핑 실행
    new_blogs = scrape_all_blogs(
        keywords=keywords,
        blogs_per_keyword=30,
        output_file=output_file,
        min_length=500,
        existing_blogs=existing_blogs,
        title_map=title_map,
        url_map=url_map,
        last_blog_id=last_blog_id
    )

    # 최종 저장 (남은 블로그가 있으면)
    if new_blogs:
        final_blogs, final_title_map, final_url_map, final_stats = save_with_deduplication(
            new_blogs, output_file, existing_blogs, title_map, url_map
        )

        logging.info(f"\n✅ 수집 완료")
        logging.info(f"총 블로그: {len(final_blogs)}개")
        logging.info(f"이번 세션 신규: {final_stats['new']}개")
        logging.info(f"이번 세션 중복: {final_stats['duplicates']}개")
        logging.info(f"저장 위치: {output_file}")

        # 샘플 출력 (신규 블로그가 있으면)
        if final_stats['new'] > 0:
            # 마지막에 추가된 블로그 출력
            sample = final_blogs[-1]
            logging.info("\n=== 샘플 데이터 (마지막 신규) ===")
            logging.info(f"제목: {sample['title']}")
            logging.info(f"본문 길이: {len(sample['full_text'])}자")
            logging.info(f"키워드: {sample['keyword']}")
            logging.info(f"URL: {sample['url']}")
    else:
        # 새 블로그는 없지만 기존 데이터는 있음
        if existing_blogs:
            logging.info(f"\n⚠️ 신규 블로그 없음 (기존: {len(existing_blogs)}개)")
        else:
            logging.warning("\n⚠️ 수집된 블로그가 없습니다.")


if __name__ == "__main__":
    main()
