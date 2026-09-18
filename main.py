import os
import sys
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    data_path = os.path.join(os.path.dirname(__file__), "data", "train_and_test2.csv")

    if not os.path.exists(data_path):
        print(f"[!] 데이터 파일을 찾을 수 없습니다: {data_path}")
        print("[!] 먼저 'python download_data.py'를 실행해주세요.")
        return

    # 데이터 로드
    df = pd.read_csv(data_path)
    print("=== 데이터셋 정보 ===")
    print(f"형상(Shape): {df.shape} (행: {df.shape[0]}, 열: {df.shape[1]})")
    print("\n=== 상위 5개 행 ===")
    print(df.head())

    print("\n=== 컬럼 목록 ===")
    print(df.columns.tolist())

    print("\n=== 결측치 요약 ===")
    print(df.isnull().sum())

if __name__ == "__main__":
    main()
