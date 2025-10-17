"""수집된 데이터 분석"""
import json
from pathlib import Path

def main():
    # 데이터 로드
    data_file = Path(__file__).parent.parent / 'data' / 'raw' / 'blogs.json'

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 통계 계산
    lengths = [len(d['full_text']) for d in data]

    print("="*50)
    print("수집 데이터 통계")
    print("="*50)
    print(f"총 블로그 수: {len(data)}개")
    print(f"평균 길이: {sum(lengths)//len(lengths):,}자")
    print(f"최소 길이: {min(lengths):,}자")
    print(f"최대 길이: {max(lengths):,}자")

    print("\n길이별 분포:")
    bins = [0, 1000, 2000, 3000, 4000, 5000, 10000]
    for i in range(len(bins)-1):
        count = sum(1 for l in lengths if bins[i] <= l < bins[i+1])
        print(f"  {bins[i]:>5,}-{bins[i+1]:>5,}자: {count:2}개 {'█'*count}")

    # 샘플 표시
    print("\n첫 3개 블로그 제목:")
    for i, blog in enumerate(data[:3], 1):
        print(f"{i}. {blog['title'][:50]}... ({len(blog['full_text'])}자)")

if __name__ == "__main__":
    main()
