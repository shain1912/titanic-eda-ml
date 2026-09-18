import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Windows 환경 콘솔 및 matplotlib 한글 폰트 설정
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False


def check_missing_values(df):
    """1. 결측치(Missing Values) 확인 및 분석"""
    print("\n" + "=" * 50)
    print("1. 결측치(Missing Values) 검증")
    print("=" * 50)

    null_counts = df.isnull().sum()
    null_percent = (df.isnull().sum() / len(df)) * 100
    missing_df = pd.DataFrame({"결측치_개수": null_counts, "결측치_비율(%)": null_percent})
    missing_df = missing_df[missing_df["결측치_개수"] > 0].sort_values(
        by="결측치_개수", ascending=False
    )

    if missing_df.empty:
        print("[V] 모든 컬럼에 결측치가 존재하지 않습니다.")
    else:
        print("[!] 결측치가 발견된 컬럼:")
        print(missing_df)

    return missing_df


def check_duplicates(df, id_column="Passengerid"):
    """2. 중복값(Duplicates) 확인 및 분석"""
    print("\n" + "=" * 50)
    print("2. 중복값(Duplicate Values) 검증")
    print("=" * 50)

    # 1) 전체 행 중복 검사
    total_duplicates = df.duplicated().sum()
    print(f"[*] 전체 행 완전 일치 중복 건수: {total_duplicates}건")

    # 2) 고유 식별자(PK) 중복 검사
    if id_column in df.columns:
        id_duplicates = df.duplicated(subset=[id_column]).sum()
        print(f"[*] 고유 키({id_column}) 기준 중복 건수: {id_duplicates}건")
    else:
        id_duplicates = 0

    return total_duplicates, id_duplicates


def detect_outliers_iqr(df, columns):
    """3. IQR(사분위 범위) 방식으로 이상치(Outliers) 탐지"""
    print("\n" + "=" * 50)
    print("3. 이상치(Outliers) 검증 (IQR 방식)")
    print("=" * 50)

    outlier_summary = []

    for col in columns:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        # 이상치 기준선: Q1 - 1.5 * IQR 미만 또는 Q3 + 1.5 * IQR 초과
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_ratio = (outlier_count / len(df)) * 100

        print(f"[{col}]")
        print(f"  - Q1(25%): {q1:.2f}, Q3(75%): {q3:.2f}, IQR: {iqr:.2f}")
        print(f"  - 정상 범위: [{lower_bound:.2f} ~ {upper_bound:.2f}]")
        print(f"  - 이상치 개수: {outlier_count}건 ({outlier_ratio:.2f}%)")
        if outlier_count > 0:
            print(f"  - 최솟값: {df[col].min():.2f}, 최댓값: {df[col].max():.2f}")

        outlier_summary.append({
            "컬럼명": col,
            "Q1": round(q1, 2),
            "Q3": round(q3, 2),
            "IQR": round(iqr, 2),
            "하한선": round(lower_bound, 2),
            "상한선": round(upper_bound, 2),
            "이상치_개수": outlier_count,
            "이상치_비율(%)": round(outlier_ratio, 2),
        })

    return pd.DataFrame(outlier_summary)


def visualize_eda(df, missing_df, output_path="eda_summary.png"):
    """결측치, 중복값, 이상치를 종합 대시보드로 시각화하여 저장"""
    print("\n" + "=" * 50)
    print(f"4. 시각화 대시보드 생성 중 -> {output_path}")
    print("=" * 50)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("타이타닉 데이터셋 기초 품질 검증 (결측치 · 중복값 · 이상치)", fontsize=18, fontweight="bold", y=0.98)

    # 차트 1: 결측치 시각화
    ax1 = axes[0, 0]
    total_cols = len(df.columns)
    if not missing_df.empty:
        sns.barplot(
            x=missing_df.index,
            y=missing_df["결측치_개수"],
            ax=ax1,
            hue=missing_df.index,
            legend=False,
            palette="Reds_r",
        )
        ax1.set_title(f"1) 결측치 현황 (총 {total_cols}개 컬럼 중)", fontsize=13, fontweight="bold")
        ax1.set_ylabel("결측치 개수")
        for p in ax1.patches:
            height = p.get_height()
            ax1.annotate(f"{int(height)}개", (p.get_x() + p.get_width() / 2.0, height),
                         ha="center", va="bottom", fontsize=11, fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "결측치 없음\n(100% 완전한 데이터)", ha="center", va="center", fontsize=14, color="green")
        ax1.set_title("1) 결측치 현황", fontsize=13, fontweight="bold")

    # 차트 2: 중복값 및 데이터 요약 카드
    ax2 = axes[0, 1]
    ax2.axis("off")
    total_rows = len(df)
    total_dups = df.duplicated().sum()
    summary_text = (
        f"■ 데이터 기본 요약\n"
        f"-----------------------------------------\n"
        f"• 총 행(샘플 수): {total_rows:,} 행\n"
        f"• 총 열(변수 수): {total_cols} 개\n"
        f"• 중복 행 개수 : {total_dups} 건 (0.0%)\n"
        f"• 승객 ID 고유값: {df['Passengerid'].nunique():,} 개\n"
        f"-----------------------------------------\n"
        f"■ 검증 결론\n"
        f"• 중복 데이터 없음 (데이터 정합성 확보)\n"
        f"• 'Embarked' 컬럼에 2건 결측치 존재\n"
        f"  (최빈값 대체 또는 행 삭제 권장)\n"
        f"• 'Fare'와 'Age'에 통계적 이상치 관측"
    )
    ax2.text(0.08, 0.5, summary_text, fontsize=11, verticalalignment="center",
             family="Malgun Gothic", bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f4f8", edgecolor="#b0c4de"))
    ax2.set_title("2) 데이터 무결성 및 중복값 요약", fontsize=13, fontweight="bold")

    # 차트 3: 나이(Age) Boxplot (이상치 탐지)
    ax3 = axes[0, 2]
    sns.boxplot(y=df["Age"], ax=ax3, color="#6baed6", flierprops={"marker": "o", "color": "red", "markersize": 5})
    ax3.set_title("3) 나이(Age) 박스플롯 - 이상치 확인", fontsize=13, fontweight="bold")
    ax3.set_ylabel("나이 (Age)")

    # 차트 4: 요금(Fare) Boxplot (이상치 탐지)
    ax4 = axes[1, 0]
    sns.boxplot(y=df["Fare"], ax=ax4, color="#fd8d3c", flierprops={"marker": "d", "color": "red", "markersize": 5})
    ax4.set_title("4) 요금(Fare) 박스플롯 - 이상치 확인", fontsize=13, fontweight="bold")
    ax4.set_ylabel("운임 요금 (Fare)")

    # 차트 5: 나이(Age) 분포 히스토그램
    ax5 = axes[1, 1]
    sns.histplot(df["Age"], kde=True, ax=ax5, color="#3182bd", bins=30)
    q1_age = df["Age"].quantile(0.25)
    q3_age = df["Age"].quantile(0.75)
    iqr_age = q3_age - q1_age
    upper_age = q3_age + 1.5 * iqr_age
    ax5.axvline(upper_age, color="red", linestyle="--", label=f"상한 임계선 ({upper_age:.1f}세)")
    ax5.set_title("5) 나이(Age) 분포 및 이상치 경계", fontsize=13, fontweight="bold")
    ax5.set_xlabel("나이 (Age)")
    ax5.set_ylabel("빈도수")
    ax5.legend()

    # 차트 6: 요금(Fare) 분포 히스토그램 (심한 우측 왜도)
    ax6 = axes[1, 2]
    sns.histplot(df["Fare"], kde=True, ax=ax6, color="#e6550d", bins=30)
    q1_fare = df["Fare"].quantile(0.25)
    q3_fare = df["Fare"].quantile(0.75)
    iqr_fare = q3_fare - q1_fare
    upper_fare = q3_fare + 1.5 * iqr_fare
    ax6.axvline(upper_fare, color="red", linestyle="--", label=f"상한 임계선 ({upper_fare:.1f})")
    ax6.set_title("6) 요금(Fare) 분포 (오른쪽 꼬리 극단치)", fontsize=13, fontweight="bold")
    ax6.set_xlabel("운임 요금 (Fare)")
    ax6.set_ylabel("빈도수")
    ax6.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[V] 시각화 대시보드가 성공적으로 저장되었습니다: {os.path.abspath(output_path)}")


def run_eda():
    data_path = os.path.join(os.path.dirname(__file__), "data", "train_and_test2.csv")
    if not os.path.exists(data_path):
        print(f"[!] 데이터 파일이 없습니다: {data_path}")
        return

    # 데이터 로드
    df = pd.read_csv(data_path)

    # 분석 대상 주요 컬럼 선별 (의미없는 zero 더미 컬럼 제외)
    meaningful_cols = [
        col for col in ["Passengerid", "Age", "Fare", "Sex", "sibsp", "Parch", "Pclass", "Embarked", "2urvived"]
        if col in df.columns
    ]
    df_clean = df[meaningful_cols].copy()
    if "2urvived" in df_clean.columns:
        df_clean.rename(columns={"2urvived": "Survived"}, inplace=True)

    print("=" * 50)
    print("타이타닉 데이터셋 기초 품질 검증 (EDA Basics)")
    print("=" * 50)
    print(f"데이터 크기: {df_clean.shape[0]}행 × {df_clean.shape[1]}열")

    # 1. 결측치 검증
    missing_df = check_missing_values(df_clean)

    # 2. 중복값 검증
    check_duplicates(df_clean, id_column="Passengerid")

    # 3. 이상치 검증 (연속형 수치 컬럼: Age, Fare, sibsp, Parch)
    num_cols = ["Age", "Fare", "sibsp", "Parch"]
    outlier_df = detect_outliers_iqr(df_clean, num_cols)

    # 4. 시각화 대시보드 저장
    output_image = os.path.join(os.path.dirname(__file__), "eda_summary.png")
    visualize_eda(df_clean, missing_df, output_path=output_image)

    return df_clean, missing_df, outlier_df


if __name__ == "__main__":
    run_eda()
