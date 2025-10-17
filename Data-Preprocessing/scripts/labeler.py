#!/usr/bin/env python3
"""
LLM 자동 라벨링 스크립트 (병렬 처리 + 이중 검증)
Gemini-2.5-flash-latest API로 블로그 글의 AI/HUMAN 여부 판단
- 10개씩 배치로 병렬 호출
- 각 블로그 글을 2번 라벨링하여 두 번 다 AI인 경우만 AI로 판정
"""

import json
import logging
import argparse
import time
import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import sys

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
except ImportError:
    print("ERROR: google-generativeai 패키지가 설치되지 않았습니다.")
    print("설치: pip install google-generativeai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("ERROR: python-dotenv 패키지가 설치되지 않았습니다.")
    print("설치: pip install python-dotenv")
    sys.exit(1)

# .env 파일 로드 (스크립트 위치 기준)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


# 로깅 설정
def setup_logging(log_dir: Path, verbose: bool = False) -> None:
    """로깅 설정"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"labeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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


def load_prompt_template(prompt_file: Path) -> str:
    """프롬프트 템플릿 파일 로드"""
    if not prompt_file.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_file}")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        template = f.read()

    logging.info(f"프롬프트 로드: {prompt_file} ({len(template)}자)")
    return template


def load_blogs(input_file: Path, limit: Optional[int] = None) -> List[Dict]:
    """블로그 데이터 로드"""
    if not input_file.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        blogs = json.load(f)

    total = len(blogs)

    if limit and limit < total:
        blogs = blogs[:limit]
        logging.info(f"블로그 로드: {len(blogs)}개 (전체 {total}개 중 제한)")
    else:
        logging.info(f"블로그 로드: {len(blogs)}개")

    return blogs


def load_existing_labels(output_file: Path) -> Dict[str, Dict]:
    """기존 라벨링 데이터 로드 (재실행 지원)"""
    if not output_file.exists():
        logging.info("기존 라벨링 데이터 없음 - 새로 시작")
        return {}

    with open(output_file, 'r', encoding='utf-8') as f:
        existing = json.load(f)

    existing_map = {blog['blog_id']: blog for blog in existing}
    logging.info(f"기존 라벨링 데이터: {len(existing_map)}개 로드")

    return existing_map


async def label_with_gemini_async(
    blog: Dict,
    prompt_template: str,
    model: genai.GenerativeModel,
    max_retries: int = 3
) -> Dict:
    """Gemini API로 단일 블로그 글 비동기 라벨링

    Args:
        blog: 원본 블로그 데이터
        prompt_template: 프롬프트 템플릿
        model: Gemini 모델 인스턴스
        max_retries: 최대 재시도 횟수

    Returns:
        라벨링 결과 (label, reasoning, keywords)
    """
    title = blog.get('title', '')
    full_text = blog.get('full_text', '')

    # 프롬프트 구성
    user_prompt = f"""
다음 네이버 블로그 글을 분석하여 AI 생성인지 인간 작성인지 판단하세요.

**제목**: {title}

**본문**:
{full_text}

---

위 기준에 따라 JSON 형식으로만 답변하세요.
"""

    # API 호출 (재시도 로직)
    for attempt in range(1, max_retries + 1):
        try:
            response = await model.generate_content_async(
                [prompt_template, user_prompt],
                generation_config={
                    'temperature': 0.3,  # 일관성을 위해 낮은 temperature
                    'max_output_tokens': 30000
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )

            # 응답 검증
            if not response.candidates:
                raise ValueError("응답에 candidates가 없습니다")

            candidate = response.candidates[0]

            # finish_reason 체크
            finish_reason = candidate.finish_reason
            if finish_reason != 1:  # 1 = STOP (정상 완료)
                finish_reason_map = {
                    0: 'UNSPECIFIED',
                    1: 'STOP',
                    2: 'MAX_TOKENS',
                    3: 'SAFETY',
                    4: 'RECITATION',
                    5: 'OTHER'
                }
                reason_name = finish_reason_map.get(finish_reason, f'UNKNOWN({finish_reason})')

                # SAFETY나 MAX_TOKENS는 재시도
                if finish_reason in [2, 3]:
                    raise ValueError(f"응답 생성 실패: finish_reason={reason_name}")
                else:
                    raise RuntimeError(f"응답 생성 실패: finish_reason={reason_name}")

            # 응답 텍스트 추출
            if not candidate.content or not candidate.content.parts:
                raise ValueError("응답에 content.parts가 없습니다")

            response_text = candidate.content.parts[0].text.strip()

            # JSON 추출 (```json ``` 제거)
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()

            result = json.loads(response_text)

            # 결과 검증
            if 'label' not in result or 'reasoning' not in result or 'keywords' not in result:
                raise ValueError("응답에 필수 필드 누락 (label, reasoning, keywords)")

            if result['label'] not in ['AI', 'HUMAN']:
                raise ValueError(f"잘못된 label 값: {result['label']}")

            # keywords 검증
            keywords = result['keywords']
            if not isinstance(keywords, list):
                raise ValueError("keywords는 리스트여야 합니다")

            if not (3 <= len(keywords) <= 5):
                raise ValueError(f"keywords는 3-5개여야 합니다 (현재: {len(keywords)}개)")

            return result

        except json.JSONDecodeError as e:
            logging.warning(f"JSON 파싱 실패 (시도 {attempt}/{max_retries}): {e}")
            logging.debug(f"응답 텍스트: {response_text[:200]}")
            if attempt == max_retries:
                raise
            await asyncio.sleep(1)

        except Exception as e:
            logging.warning(f"API 호출 실패 (시도 {attempt}/{max_retries}): {e}")
            if attempt == max_retries:
                raise
            await asyncio.sleep(2)

    raise RuntimeError(f"최대 재시도 횟수 초과: {blog.get('blog_id', 'unknown')}")


async def label_blog_twice_async(
    blog: Dict,
    prompt_template: str,
    model: genai.GenerativeModel
) -> Dict:
    """한 블로그 글을 2번 라벨링하여 두 번 다 AI인 경우만 AI로 판정

    Args:
        blog: 원본 블로그 데이터
        prompt_template: 프롬프트 템플릿
        model: Gemini 모델 인스턴스

    Returns:
        원본 데이터 + 최종 라벨링 결과
    """
    blog_id = blog.get('blog_id', 'unknown')

    # 2번 병렬 호출
    results = await asyncio.gather(
        label_with_gemini_async(blog, prompt_template, model),
        label_with_gemini_async(blog, prompt_template, model),
        return_exceptions=True
    )

    # 에러 체크
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            raise RuntimeError(f"{blog_id} 라벨링 실패 (시도 {i+1}): {result}")

    result1, result2 = results
    label1 = result1['label']
    label2 = result2['label']

    # 두 번 다 AI인 경우만 AI로 판정, 나머지는 HUMAN
    if label1 == 'AI' and label2 == 'AI':
        final_label = 'AI'
        final_reasoning = result1['reasoning'][:100]
        final_keywords = result1['keywords']
        logging.debug(f"{blog_id}: AI + AI → AI")
    else:
        final_label = 'HUMAN'
        # HUMAN 판정이 있으면 그것을 사용, 없으면 첫 번째 결과 사용
        if label1 == 'HUMAN':
            final_reasoning = result1['reasoning'][:100]
            final_keywords = result1['keywords']
        else:
            final_reasoning = result2['reasoning'][:100]
            final_keywords = result2['keywords']
        logging.debug(f"{blog_id}: {label1} + {label2} → HUMAN")

    # 원본 데이터에 라벨링 결과 추가
    labeled_blog = blog.copy()
    labeled_blog['label'] = final_label
    labeled_blog['reasoning'] = final_reasoning
    labeled_blog['keywords'] = final_keywords
    labeled_blog['labeled_at'] = datetime.now().isoformat()

    # 로깅용으로 일치 여부 반환 (JSON에는 저장 안 함)
    labeled_blog['_double_check_info'] = {
        'first_label': label1,
        'second_label': label2,
        'agreement': label1 == label2
    }

    return labeled_blog


async def process_batch_async(
    blogs: List[Dict],
    prompt_template: str,
    model: genai.GenerativeModel
) -> List[Dict]:
    """배치(10개)를 병렬로 처리

    Args:
        blogs: 블로그 데이터 리스트 (최대 10개)
        prompt_template: 프롬프트 템플릿
        model: Gemini 모델 인스턴스

    Returns:
        라벨링된 블로그 데이터 리스트
    """
    tasks = [
        label_blog_twice_async(blog, prompt_template, model)
        for blog in blogs
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    labeled_blogs = []
    failed_blogs = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            blog_id = blogs[i].get('blog_id', f'unknown_{i}')
            logging.error(f"배치 처리 실패: {blog_id} - {result}")
            failed_blogs.append(blog_id)
        else:
            labeled_blogs.append(result)

    if failed_blogs:
        logging.warning(f"배치 내 실패: {len(failed_blogs)}개 - {', '.join(failed_blogs)}")

    return labeled_blogs


def save_labeled_blogs(blogs: List[Dict], output_file: Path) -> None:
    """라벨링된 블로그 데이터 저장 (로깅 정보 제거)"""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # _double_check_info 제거하고 저장
    clean_blogs = []
    for blog in blogs:
        clean_blog = {k: v for k, v in blog.items() if not k.startswith('_')}
        clean_blogs.append(clean_blog)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_blogs, f, ensure_ascii=False, indent=2)

    logging.info(f"저장 완료: {output_file} ({len(clean_blogs)}개)")


def calculate_statistics(labeled_blogs: List[Dict]) -> Dict[str, Any]:
    """라벨링 통계 계산"""
    total = len(labeled_blogs)

    if total == 0:
        return {
            'total': 0,
            'ai_count': 0,
            'human_count': 0,
            'ai_ratio': 0.0,
            'human_ratio': 0.0,
            'agreement_count': 0,
            'agreement_ratio': 0.0
        }

    ai_count = sum(1 for b in labeled_blogs if b.get('label') == 'AI')
    human_count = total - ai_count

    # _double_check_info에서 일치율 계산 (로깅용)
    agreement_count = sum(
        1 for b in labeled_blogs
        if b.get('_double_check_info', {}).get('agreement', False)
    )

    return {
        'total': total,
        'ai_count': ai_count,
        'human_count': human_count,
        'ai_ratio': ai_count / total,
        'human_ratio': human_count / total,
        'agreement_count': agreement_count,
        'agreement_ratio': agreement_count / total if total > 0 else 0.0
    }


def print_statistics(stats: Dict[str, Any], duration: float) -> None:
    """통계 출력"""
    print("\n" + "="*50)
    print("=== Phase 1 통계 (이중 검증) ===")
    print("="*50)
    print(f"총 라벨링: {stats['total']}개")
    print(f"- AI: {stats['ai_count']}개 ({stats['ai_ratio']*100:.1f}%)")
    print(f"- HUMAN: {stats['human_count']}개 ({stats['human_ratio']*100:.1f}%)")
    print(f"- 2번 일치: {stats['agreement_count']}개 ({stats['agreement_ratio']*100:.1f}%)")
    print(f"소요 시간: {duration:.1f}초 ({duration/60:.1f}분)")

    # 비용 추정 (Gemini-2.5-flash-latest: ~$0.20/1M tokens, 2번 호출이므로 2배)
    avg_chars = 2000
    estimated_tokens = stats['total'] * avg_chars * 1.5 * 2  # 2번 호출
    estimated_cost = (estimated_tokens / 1_000_000) * 0.20
    print(f"예상 비용: ~${estimated_cost:.4f}")
    print("="*50 + "\n")


async def label_all_blogs_async(
    input_file: Path,
    output_file: Path,
    prompt_file: Path,
    api_key: str,
    model_name: str = 'gemini-2.5-flash-latest',
    batch_size: int = 10,
    limit: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """전체 블로그 데이터 라벨링 (병렬 처리 + 이중 검증)

    Args:
        input_file: 입력 블로그 데이터 (blogs.json)
        output_file: 출력 라벨링 데이터 (labeled_blogs.json)
        prompt_file: 프롬프트 파일
        api_key: Gemini API 키
        model_name: Gemini 모델명
        batch_size: 배치당 처리 개수 (기본: 10)
        limit: 테스트용 데이터 제한
        verbose: 상세 로깅 여부

    Returns:
        통계 정보
    """
    start_time = time.time()

    # Gemini API 설정
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    logging.info(f"사용 모델: {model_name}")
    logging.info(f"배치 크기: {batch_size}개 (각 블로그 2번 라벨링)")

    # 데이터 로드
    prompt_template = load_prompt_template(prompt_file)
    blogs = load_blogs(input_file, limit)
    existing_labels = load_existing_labels(output_file)

    # 이미 라벨링된 데이터 필터링
    blogs_to_label = [
        blog for blog in blogs
        if blog['blog_id'] not in existing_labels
    ]

    if len(blogs_to_label) == 0:
        logging.info("모든 블로그가 이미 라벨링됨 - 처리할 데이터 없음")
        stats = calculate_statistics(list(existing_labels.values()))
        duration = time.time() - start_time
        print_statistics(stats, duration)
        return stats

    logging.info(f"라벨링 대상: {len(blogs_to_label)}개 (기존 {len(existing_labels)}개 스킵)")

    # 배치 단위로 처리
    total_batches = (len(blogs_to_label) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(blogs_to_label))
        batch_blogs = blogs_to_label[start_idx:end_idx]

        logging.info(f"배치 {batch_idx+1}/{total_batches} 처리 중 ({len(batch_blogs)}개)...")

        try:
            # 배치 병렬 처리
            labeled_batch = await process_batch_async(batch_blogs, prompt_template, model)

            # 결과 저장
            for idx, labeled_blog in enumerate(labeled_batch, 1):
                blog_id = labeled_blog['blog_id']
                existing_labels[blog_id] = labeled_blog

                # 전체 진행 상황 계산
                global_idx = start_idx + idx

                label = labeled_blog['label']
                double_check = labeled_blog.get('_double_check_info', {})
                agreement = "일치" if double_check.get('agreement', False) else "불일치"
                first_label = double_check.get('first_label', '?')
                second_label = double_check.get('second_label', '?')
                logging.info(
                    f"[{global_idx}/{len(blogs_to_label)}] {blog_id}: "
                    f"{first_label} + {second_label} → {label} ({agreement})"
                )

            # 배치마다 저장
            all_labeled = list(existing_labels.values())
            save_labeled_blogs(all_labeled, output_file)

            batch_stats = calculate_statistics(all_labeled)
            logging.info(
                f"배치 {batch_idx+1}/{total_batches} 완료 "
                f"(AI: {batch_stats['ai_count']}, HUMAN: {batch_stats['human_count']}, "
                f"일치율: {batch_stats['agreement_ratio']*100:.1f}%)"
            )

            # Rate limit 고려
            if batch_idx < total_batches - 1:
                await asyncio.sleep(2)

        except Exception as e:
            logging.error(f"배치 {batch_idx+1} 처리 실패: {e}")
            continue

    # 최종 저장
    all_labeled = list(existing_labels.values())
    save_labeled_blogs(all_labeled, output_file)

    # 통계 계산 및 출력
    stats = calculate_statistics(all_labeled)
    duration = time.time() - start_time

    print_statistics(stats, duration)

    return stats


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='LLM 자동 라벨링 (Gemini-2.5-flash-latest) - 병렬 처리 + 이중 검증',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # .env 파일 생성 (권장)
  cp .env.example .env
  # .env 파일에 GEMINI_API_KEY, GEMINI_MODEL 입력 후
  python labeler.py

  # CLI 인자로 API 키와 모델 전달
  python labeler.py --api-key YOUR_KEY --model gemini-2.5-flash-latest

  # 테스트 모드 (처음 10개만)
  python labeler.py --limit 10

  # 배치 크기 조정 (기본: 10)
  python labeler.py --batch-size 5

  # 상세 로깅
  python labeler.py --verbose

특징:
  - 10개씩 배치로 병렬 처리 (속도 향상)
  - 각 블로그 글을 2번 라벨링하여 두 번 다 AI인 경우만 AI로 판정
  - 정확도와 보수적 판단을 위한 이중 검증
        """
    )

    # API 키 (.env 파일 또는 CLI 인자)
    parser.add_argument(
        '--api-key',
        default=None,
        help='Gemini API 키 (없으면 .env 파일의 GEMINI_API_KEY 사용)'
    )

    # 모델명 (.env 파일 또는 CLI 인자)
    parser.add_argument(
        '--model',
        default=None,
        help='Gemini 모델명 (없으면 .env 파일의 GEMINI_MODEL 사용, 기본값: gemini-2.5-flash-latest)'
    )

    # 선택 인자
    parser.add_argument(
        '--input',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'Data-Collection' / 'data' / 'blogs.json',
        help='입력 블로그 데이터 경로 (기본: ../Data-Collection/data/blogs.json)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'labeled' / 'labeled_blogs.json',
        help='출력 라벨링 데이터 경로 (기본: data/labeled/labeled_blogs.json)'
    )

    parser.add_argument(
        '--prompt',
        type=Path,
        default=Path(__file__).parent.parent / 'prompts' / 'labeling_prompt.md',
        help='프롬프트 파일 경로 (기본: prompts/labeling_prompt.md)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='배치 크기 (기본: 10)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='처리할 데이터 개수 제한 (테스트용)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='상세 로깅'
    )

    args = parser.parse_args()

    # API 키 확인 (CLI 인자 > .env 파일)
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')

    if not api_key:
        print("ERROR: Gemini API 키가 필요합니다.")
        print("방법 1: .env 파일 생성 (권장)")
        print("  1) .env.example을 .env로 복사")
        print("  2) GEMINI_API_KEY=your_api_key_here 입력")
        print("방법 2: CLI 인자 사용")
        print("  python labeler.py --api-key YOUR_KEY")
        sys.exit(1)

    # 모델명 확인 (CLI 인자 > .env 파일 > 기본값)
    model_name = args.model or os.environ.get('GEMINI_MODEL') or 'gemini-2.5-flash-latest'

    # 로깅 설정
    log_dir = Path(__file__).parent.parent / 'logs'
    setup_logging(log_dir, args.verbose)

    logging.info("="*50)
    logging.info("Phase 1: LLM 자동 라벨링 시작 (병렬 처리 + 이중 검증)")
    logging.info("="*50)
    logging.info(f"모델: {model_name}")
    logging.info(f"입력: {args.input}")
    logging.info(f"출력: {args.output}")
    logging.info(f"프롬프트: {args.prompt}")
    logging.info(f"배치 크기: {args.batch_size}")
    if args.limit:
        logging.info(f"데이터 제한: {args.limit}개")

    try:
        stats = asyncio.run(
            label_all_blogs_async(
                input_file=args.input,
                output_file=args.output,
                prompt_file=args.prompt,
                api_key=api_key,
                model_name=model_name,
                batch_size=args.batch_size,
                limit=args.limit,
                verbose=args.verbose
            )
        )

        logging.info("Phase 1 완료!")

        # 품질 게이트 체크
        if 0.4 <= stats['ai_ratio'] <= 0.6:
            logging.info("✅ 품질 게이트 통과 (AI/HUMAN 비율)")
        else:
            logging.warning("⚠️  품질 게이트 실패 - 데이터 검토 필요")
            logging.warning(f"  - AI/HUMAN 비율 불균형: {stats['ai_ratio']*100:.1f}% AI")

        # 일치율 체크
        if stats['agreement_ratio'] >= 0.7:
            logging.info(f"✅ 이중 검증 일치율 양호: {stats['agreement_ratio']*100:.1f}%")
        else:
            logging.warning(f"⚠️  이중 검증 일치율 낮음: {stats['agreement_ratio']*100:.1f}%")

    except Exception as e:
        logging.error(f"실행 실패: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
