"""
디버깅용 스크립트: 네이버 검색 결과 HTML 구조 분석
"""

import time
from pathlib import Path

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def setup_driver():
    """디버깅용 드라이버 (헤드리스 OFF)"""
    options = Options()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument(
        'user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
    options.add_argument('--window-size=1920,1080')

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        '''
    })

    return driver


def main():
    """디버깅 메인"""
    keyword = "도쿄돔호텔 후기"
    search_url = f"https://search.naver.com/search.naver?ssc=tab.blog.all&query={keyword}"

    print(f"검색 URL: {search_url}")
    print("브라우저 실행 중...")

    driver = setup_driver()

    try:
        # 검색 페이지 접속
        driver.get(search_url)
        print("페이지 로딩 대기 (5초)...")
        time.sleep(5)

        # HTML 소스 저장
        output_dir = Path(__file__).parent.parent / 'logs'
        output_dir.mkdir(parents=True, exist_ok=True)
        html_file = output_dir / 'search_page.html'

        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(driver.page_source)

        print(f"\n✓ HTML 저장: {html_file}")

        # BeautifulSoup 파싱
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 다양한 선택자 시도
        print("\n=== CSS 선택자 테스트 ===")

        selectors = [
            '.detail_box',
            '.view_wrap',
            '.total_wrap',
            '.bx',
            '.api_ani_send',
            'a[href*="blog.naver.com"]',
            '.title_link',
            '.api_txt_lines',
        ]

        for selector in selectors:
            elements = soup.select(selector)
            print(f"{selector:30} → {len(elements):3}개")

        # 블로그 링크 추출
        print("\n=== 블로그 링크 추출 시도 ===")
        blog_links = soup.select('a[href*="blog.naver.com"]')
        print(f"총 {len(blog_links)}개 블로그 링크 발견")

        if blog_links:
            print("\n첫 5개 링크:")
            for i, link in enumerate(blog_links[:5], 1):
                href = link.get('href', '')
                text = link.get_text(strip=True)[:50]
                print(f"{i}. {text}")
                print(f"   URL: {href}")

        # 클래스 분석
        print("\n=== 주요 클래스 분석 ===")
        for tag_name in ['div', 'a', 'li']:
            tags = soup.find_all(tag_name, limit=20)
            classes = set()
            for tag in tags:
                tag_classes = tag.get('class', [])
                if tag_classes:
                    classes.update(tag_classes)

            if classes:
                print(f"\n{tag_name} 태그 클래스:")
                for cls in sorted(list(classes))[:10]:
                    print(f"  - {cls}")

        print("\n디버깅 완료. 브라우저를 닫으려면 Enter를 누르세요...")
        input()

    finally:
        driver.quit()
        print("브라우저 종료")


if __name__ == "__main__":
    main()
