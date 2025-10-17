#!/usr/bin/env python3
"""
스니펫 분할 전처리 스크립트
라벨링된 블로그 글을 3~5개 스니펫(100-300자)으로 분할하여 학습 데이터 생성
"""

import json
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from collections import Counter
import sys


# 로깅 설정
def setup_logging(log_dir: Path, verbose: bool = False) -> None:
    """로깅 설정"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info(f"로그 파일: {log_file}")


def load_labeled_blogs(input_file: Path, limit: Optional[int] = None) -> List[Dict]:
    """라벨링된 블로그 데이터 로드"""
    if not input_file.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        blogs = json.load(f)

    total = len(blogs)

    if limit and limit < total:
        blogs = blogs[:limit]
        logging.info(f"라벨링 데이터 로드: {len(blogs)}개 (전체 {total}개 중 제한)")
    else:
        logging.info(f"라벨링 데이터 로드: {len(blogs)}개")

    return blogs


def load_existing_snippets(output_file: Path) -> Set[str]:
    """이미 스니펫 분할된 blog_id 추출 (재실행 지원)"""
    if not output_file.exists():
        logging.info("기존 스니펫 데이터 없음 - 새로 시작")
        return set()

    with open(output_file, 'r', encoding='utf-8') as f:
        snippets = json.load(f)

    processed_ids = {s['original_blog_id'] for s in snippets}
    logging.info(f"기존 스니펫 데이터: {len(snippets)}개 (처리된 블로그: {len(processed_ids)}개)")

    return processed_ids


def get_num_snippets(text_length: int) -> int:
    """글 길이에 따른 스니펫 개수 결정

    Args:
        text_length: 전체 글 길이

    Returns:
        스니펫 개수 (3~5개)
    """
    if text_length < 500:
        return 3
    elif text_length < 1500:
        return 4
    else:
        return 5


def extract_keyword_based_snippets(
    full_text: str,
    keywords: List[str],
    min_length: int = 100,
    max_length: int = 300
) -> List[Dict]:
    """키워드 기반 스니펫 추출 (네이버 검색 환경 시뮬레이션)

    Args:
        full_text: 원본 글 전체 본문
        keywords: 검색 키워드 리스트
        min_length: 최소 스니펫 길이
        max_length: 최대 스니펫 길이

    Returns:
        스니펫 리스트 (keyword, text, length 포함)
    """
    import re

    snippets = []
    used_positions = set()  # 중복 방지

    for keyword in keywords:
        # 키워드가 등장하는 모든 위치 찾기 (대소문자 무시)
        pattern = re.escape(keyword)
        matches = list(re.finditer(pattern, full_text, re.IGNORECASE))

        if not matches:
            continue

        # 첫 번째 등장 위치 사용
        match = matches[0]
        keyword_start = match.start()
        keyword_end = match.end()

        # 키워드 중심으로 전후 맥락 추출
        context_before = (max_length - len(keyword)) // 2
        context_after = max_length - len(keyword) - context_before

        snippet_start = max(0, keyword_start - context_before)
        snippet_end = min(len(full_text), keyword_end + context_after)

        # 실제 스니펫 추출
        snippet_text = full_text[snippet_start:snippet_end].strip()

        # 길이 검증 및 조정
        if len(snippet_text) < min_length:
            # 더 길게 확장
            snippet_start = max(0, keyword_start - min_length // 2)
            snippet_end = min(len(full_text), snippet_start + max_length)
            snippet_text = full_text[snippet_start:snippet_end].strip()

            # 재검증: 여전히 짧으면 스킵
            if len(snippet_text) < min_length:
                continue

        # max_length 초과하면 자르기
        if len(snippet_text) > max_length:
            snippet_text = snippet_text[:max_length].strip()

        # 중복 방지 (비슷한 위치면 스킵)
        position_key = (snippet_start // 100, snippet_end // 100)
        if position_key in used_positions:
            continue

        used_positions.add(position_key)

        snippets.append({
            "keyword": keyword,
            "text": snippet_text,
            "length": len(snippet_text)
        })

        # 최대 5개까지만
        if len(snippets) >= 5:
            break

    return snippets


def split_into_snippets_fallback(
    full_text: str,
    min_length: int = 100,
    max_length: int = 300,
    num_needed: int = 3
) -> List[Dict]:
    """위치 기반 폴백 스니펫 추출 (키워드 실패 시)

    Args:
        full_text: 원본 글 전체 본문
        min_length: 최소 스니펫 길이
        max_length: 최대 스니펫 길이
        num_needed: 필요한 스니펫 개수

    Returns:
        스니펫 리스트 (position, text, length 포함)
    """
    text_len = len(full_text)

    if text_len == 0:
        return []

    position_labels = ["start", "middle", "end", "early_middle", "late_middle"]
    positions = position_labels[:num_needed]

    snippets = []

    for i, pos in enumerate(positions):
        # 각 스니펫의 시작 비율 계산
        start_ratio = i / num_needed
        start_idx = int(text_len * start_ratio)

        # 스니펫 추출 (max_length까지)
        end_idx = min(start_idx + max_length, text_len)
        snippet_text = full_text[start_idx:end_idx]

        # 최소 길이 보장
        if len(snippet_text) < min_length and start_idx + min_length <= text_len:
            snippet_text = full_text[start_idx:start_idx + min_length]

        if not snippet_text.strip():
            continue

        snippets.append({
            "position": pos,
            "text": snippet_text.strip(),
            "length": len(snippet_text.strip())
        })

    return snippets


def split_into_snippets(
    full_text: str,
    keywords: Optional[List[str]] = None,
    min_length: int = 100,
    max_length: int = 300
) -> List[Dict]:
    """스니펫 분할 (키워드 기반 우선, 하이브리드 보충)

    Args:
        full_text: 원본 글 전체 본문
        keywords: 검색 키워드 리스트 (옵션)
        min_length: 최소 스니펫 길이
        max_length: 최대 스니펫 길이

    Returns:
        스니펫 리스트
    """
    num_needed = get_num_snippets(len(full_text))

    # 1. 키워드 기반 시도
    snippets = []
    if keywords and len(keywords) > 0:
        snippets = extract_keyword_based_snippets(
            full_text, keywords, min_length, max_length
        )

    # 2. 충분하면 반환
    if len(snippets) >= num_needed:
        return snippets[:5]  # 최대 5개

    # 3. 하이브리드: 키워드 기반 + 위치 기반 보충
    if len(snippets) > 0:
        # 키워드 기반 스니펫이 일부 있음 → 부족한 만큼 위치 기반으로 보충
        num_additional = num_needed - len(snippets)
        position_snippets = split_into_snippets_fallback(
            full_text, min_length, max_length, num_additional
        )
        # 키워드 기반을 우선 배치
        return snippets + position_snippets

    # 4. 완전 폴백: 키워드 기반 실패 시 위치 기반만 사용
    return split_into_snippets_fallback(
        full_text, min_length, max_length, num_needed
    )


def create_training_snippet(
    blog: Dict,
    snippet_data: Dict,
    snippet_index: int
) -> Dict:
    """학습용 스니펫 데이터 생성

    Args:
        blog: 원본 블로그 데이터
        snippet_data: 분할된 스니펫 정보
        snippet_index: 스니펫 순번 (1부터 시작)

    Returns:
        학습 데이터 스니펫
    """
    blog_id = blog['blog_id']

    snippet = {
        'snippet_id': f"{blog_id}_{snippet_index:02d}",
        'original_blog_id': blog_id,
        'title': blog['title'],
        'snippet_text': snippet_data['text'],
        'snippet_length': snippet_data['length'],
        'label': blog['label'],
        'keywords': blog.get('keywords', []),  # 키워드 리스트
        'created_at': datetime.now().isoformat()
    }

    # 키워드 기반인 경우 keyword 필드 추가
    if 'keyword' in snippet_data:
        snippet['matched_keyword'] = snippet_data['keyword']

    # 위치 기반인 경우 position 필드 추가
    if 'position' in snippet_data:
        snippet['position'] = snippet_data['position']

    return snippet


def save_training_data(snippets: List[Dict], output_file: Path) -> None:
    """학습 데이터 저장"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(snippets, f, ensure_ascii=False, indent=2)

    logging.info(f"저장 완료: {output_file} ({len(snippets)}개)")


def calculate_statistics(
    blogs: List[Dict],
    snippets: List[Dict]
) -> Dict[str, Any]:
    """전처리 통계 계산"""
    total_blogs = len(blogs)
    total_snippets = len(snippets)

    if total_blogs == 0:
        return {
            'total_blogs': 0,
            'total_snippets': 0,
            'augmentation_ratio': 0.0,
            'avg_snippet_length': 0.0,
            'snippet_distribution': {},
            'label_distribution': {},
            'position_distribution': {}
        }

    augmentation_ratio = total_snippets / total_blogs

    # 스니펫 길이 통계
    snippet_lengths = [s['snippet_length'] for s in snippets]
    avg_snippet_length = sum(snippet_lengths) / len(snippet_lengths)

    # 스니펫 개수별 분포
    blog_snippet_counts = {}
    for blog in blogs:
        blog_id = blog['blog_id']
        count = sum(1 for s in snippets if s['original_blog_id'] == blog_id)
        blog_snippet_counts[count] = blog_snippet_counts.get(count, 0) + 1

    # 라벨 분포
    snippet_labels = Counter(s['label'] for s in snippets)
    blog_labels = Counter(b['label'] for b in blogs)

    # 위치 분포 (position 필드가 있는 스니펫만)
    position_dist = Counter(s.get('position', 'keyword-based') for s in snippets)

    # 키워드 매칭 분포
    keyword_based = sum(1 for s in snippets if 'matched_keyword' in s)
    position_based = sum(1 for s in snippets if 'position' in s)

    return {
        'total_blogs': total_blogs,
        'total_snippets': total_snippets,
        'augmentation_ratio': augmentation_ratio,
        'avg_snippet_length': avg_snippet_length,
        'snippet_distribution': blog_snippet_counts,
        'label_distribution': {
            'snippets': dict(snippet_labels),
            'blogs': dict(blog_labels)
        },
        'position_distribution': dict(position_dist),
        'snippet_method': {
            'keyword_based': keyword_based,
            'position_based': position_based
        }
    }


def print_statistics(stats: Dict[str, Any], duration: float) -> None:
    """통계 출력"""
    print("\n" + "="*50)
    print("=== Phase 2 통계 ===")
    print("="*50)
    print(f"입력: {stats['total_blogs']}개 블로그")
    print(f"출력: {stats['total_snippets']}개 스니펫")
    print(f"증강 비율: {stats['augmentation_ratio']:.1f}x")
    print(f"평균 snippet 길이: {stats['avg_snippet_length']:.0f}자")

    # 스니펫 개수별 분포
    print("\n[스니펫 분할 분포]")
    for num_snippets in sorted(stats['snippet_distribution'].keys()):
        count = stats['snippet_distribution'][num_snippets]
        print(f"  {num_snippets}개 분할: {count}개 블로그")

    # 라벨 분포
    print("\n[라벨 분포]")
    snippet_labels = stats['label_distribution']['snippets']
    blog_labels = stats['label_distribution']['blogs']

    for label in ['AI', 'HUMAN']:
        snippet_count = snippet_labels.get(label, 0)
        blog_count = blog_labels.get(label, 0)
        snippet_ratio = snippet_count / stats['total_snippets'] * 100 if stats['total_snippets'] > 0 else 0
        blog_ratio = blog_count / stats['total_blogs'] * 100 if stats['total_blogs'] > 0 else 0

        print(f"  {label}:")
        print(f"    스니펫: {snippet_count}개 ({snippet_ratio:.1f}%)")
        print(f"    원본 블로그: {blog_count}개 ({blog_ratio:.1f}%)")

    # 스니펫 생성 방식 분포
    print("\n[스니펫 생성 방식]")
    method = stats.get('snippet_method', {})
    keyword_count = method.get('keyword_based', 0)
    position_count = method.get('position_based', 0)

    if keyword_count > 0:
        keyword_ratio = keyword_count / stats['total_snippets'] * 100
        print(f"  키워드 기반: {keyword_count}개 ({keyword_ratio:.1f}%)")
    if position_count > 0:
        position_ratio = position_count / stats['total_snippets'] * 100
        print(f"  위치 기반: {position_count}개 ({position_ratio:.1f}%)")

    # 위치 분포 (위치 기반 스니펫만)
    if position_count > 0:
        print("\n[위치 분포] (위치 기반 스니펫)")
        for position in ['start', 'early_middle', 'middle', 'late_middle', 'end']:
            count = stats['position_distribution'].get(position, 0)
            if count > 0:
                print(f"  {position}: {count}개")

    print(f"\n소요 시간: {duration:.1f}초")
    print("="*50 + "\n")


def validate_phase2(
    blogs: List[Dict],
    snippets: List[Dict]
) -> Dict[str, bool]:
    """Phase 2 품질 게이트 검증"""
    stats = calculate_statistics(blogs, snippets)

    # 검증 조건
    checks = {
        'augmentation_ratio': 3.0 <= stats['augmentation_ratio'] <= 3.5,
        'snippet_length': all(
            100 <= s['snippet_length'] <= 300
            for s in snippets
        ),
        'label_distribution_match': True
    }

    # 라벨 분포 매칭 검증 (±5% 오차)
    if stats['total_blogs'] > 0 and stats['total_snippets'] > 0:
        blog_ai_ratio = stats['label_distribution']['blogs'].get('AI', 0) / stats['total_blogs']
        snippet_ai_ratio = stats['label_distribution']['snippets'].get('AI', 0) / stats['total_snippets']
        checks['label_distribution_match'] = abs(blog_ai_ratio - snippet_ai_ratio) < 0.05

    return checks


def preprocess_all_blogs(
    input_file: Path,
    output_file: Path,
    min_length: int = 100,
    max_length: int = 300,
    limit: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """전체 라벨링 데이터 → 스니펫 분할

    Args:
        input_file: 입력 라벨링 데이터
        output_file: 출력 학습 데이터
        min_length: 최소 스니펫 길이
        max_length: 최대 스니펫 길이
        limit: 테스트용 데이터 제한
        verbose: 상세 로깅

    Returns:
        통계 정보
    """
    start_time = time.time()

    # 데이터 로드
    blogs = load_labeled_blogs(input_file, limit)
    processed_ids = load_existing_snippets(output_file)

    # 기존 스니펫 로드 (재실행 시)
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_snippets = json.load(f)
    else:
        existing_snippets = []

    # 이미 처리된 블로그 필터링
    blogs_to_process = [
        blog for blog in blogs
        if blog['blog_id'] not in processed_ids
    ]

    if len(blogs_to_process) == 0:
        logging.info("모든 블로그가 이미 처리됨 - 처리할 데이터 없음")
        stats = calculate_statistics(blogs, existing_snippets)
        duration = time.time() - start_time
        print_statistics(stats, duration)
        return stats

    logging.info(f"처리 대상: {len(blogs_to_process)}개 블로그 (기존 {len(processed_ids)}개 스킵)")
    logging.info(f"스니펫 길이: {min_length}~{max_length}자")

    # 스니펫 분할 실행
    new_snippets = []
    failed_blogs = []

    for i, blog in enumerate(blogs_to_process, 1):
        blog_id = blog.get('blog_id', f'unknown_{i}')

        try:
            full_text = blog.get('full_text', '')

            if not full_text:
                logging.warning(f"[{i}/{len(blogs_to_process)}] {blog_id}: 본문 없음 - 스킵")
                continue

            # 키워드 추출
            keywords = blog.get('keywords', [])

            # 스니펫 분할 (키워드 기반)
            snippet_list = split_into_snippets(full_text, keywords, min_length, max_length)

            if len(snippet_list) == 0:
                logging.warning(f"[{i}/{len(blogs_to_process)}] {blog_id}: 스니펫 생성 실패 - 스킵")
                continue

            # 학습 데이터 생성
            for idx, snippet_data in enumerate(snippet_list, 1):
                training_snippet = create_training_snippet(blog, snippet_data, idx)
                new_snippets.append(training_snippet)

            logging.info(
                f"[{i}/{len(blogs_to_process)}] {blog_id}: {len(snippet_list)}개 스니펫 생성 "
                f"(평균 {sum(s['length'] for s in snippet_list) / len(snippet_list):.0f}자)"
            )

        except Exception as e:
            logging.error(f"스니펫 분할 실패: {blog_id} - {e}")
            failed_blogs.append(blog_id)
            continue

    # 기존 스니펫과 병합
    all_snippets = existing_snippets + new_snippets

    # 최종 저장
    save_training_data(all_snippets, output_file)

    # 통계 계산 및 출력
    stats = calculate_statistics(blogs, all_snippets)
    duration = time.time() - start_time

    print_statistics(stats, duration)

    if failed_blogs:
        logging.warning(f"실패한 블로그: {len(failed_blogs)}개")
        logging.warning(f"실패 목록: {', '.join(failed_blogs)}")

    # 품질 게이트 검증
    checks = validate_phase2(blogs, all_snippets)

    logging.info("\n[품질 게이트 검증]")
    for check_name, passed in checks.items():
        status = "✅ 통과" if passed else "❌ 실패"
        logging.info(f"  {check_name}: {status}")

    if all(checks.values()):
        logging.info("✅ 모든 품질 게이트 통과")
    else:
        logging.warning("⚠️  일부 품질 게이트 실패 - 데이터 검토 필요")

    return stats


def show_stats_only(input_file: Path, output_file: Path) -> None:
    """통계만 출력 (처리 안 함)"""
    if not input_file.exists():
        print(f"입력 파일 없음: {input_file}")
        return

    if not output_file.exists():
        print(f"출력 파일 없음: {output_file}")
        return

    blogs = load_labeled_blogs(input_file, limit=None)

    with open(output_file, 'r', encoding='utf-8') as f:
        snippets = json.load(f)

    stats = calculate_statistics(blogs, snippets)
    print_statistics(stats, 0.0)

    # 품질 게이트 검증
    checks = validate_phase2(blogs, snippets)

    print("[품질 게이트 검증]")
    for check_name, passed in checks.items():
        status = "✅ 통과" if passed else "❌ 실패"
        print(f"  {check_name}: {status}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='스니펫 분할 전처리',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용 (전체 처리)
  python preprocess.py

  # 테스트 모드 (처음 10개만)
  python preprocess.py --limit 10

  # 스니펫 길이 커스터마이징
  python preprocess.py --min-length 150 --max-length 250

  # 통계만 출력 (처리 안 함)
  python preprocess.py --stats
        """
    )

    # 선택 인자
    parser.add_argument(
        '--input',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'labeled' / 'labeled_blogs.json',
        help='입력 라벨링 데이터 경로 (기본: data/labeled/labeled_blogs.json)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'processed' / 'training_data.json',
        help='출력 학습 데이터 경로 (기본: data/processed/training_data.json)'
    )

    parser.add_argument(
        '--min-length',
        type=int,
        default=100,
        help='최소 스니펫 길이 (기본: 100)'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=300,
        help='최대 스니펫 길이 (기본: 300)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='처리할 데이터 개수 제한 (테스트용)'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='통계만 출력 (처리 안 함)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='상세 로깅'
    )

    args = parser.parse_args()

    # 통계만 출력
    if args.stats:
        show_stats_only(args.input, args.output)
        return

    # 로깅 설정
    log_dir = Path(__file__).parent.parent / 'logs'
    setup_logging(log_dir, args.verbose)

    logging.info("="*50)
    logging.info("Phase 2: 스니펫 분할 전처리 시작")
    logging.info("="*50)
    logging.info(f"입력: {args.input}")
    logging.info(f"출력: {args.output}")
    logging.info(f"스니펫 길이: {args.min_length}~{args.max_length}자")
    if args.limit:
        logging.info(f"데이터 제한: {args.limit}개")

    try:
        stats = preprocess_all_blogs(
            input_file=args.input,
            output_file=args.output,
            min_length=args.min_length,
            max_length=args.max_length,
            limit=args.limit,
            verbose=args.verbose
        )

        logging.info("Phase 2 완료!")

    except Exception as e:
        logging.error(f"실행 실패: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
