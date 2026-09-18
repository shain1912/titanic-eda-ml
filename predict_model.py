import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# Windows 환경 콘솔 UTF-8 및 matplotlib 한글 폰트 설정
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False


def load_and_preprocess_data(data_path):
    """1. 데이터 로드, 결측치 처리 및 특성 공학(Feature Engineering)"""
    print("\n" + "=" * 60)
    print("1. 데이터 로드 및 특성 공학(Feature Engineering)")
    print("=" * 60)

    df = pd.read_csv(data_path)

    # 1) 컬럼 선별 및 이름 표준화
    cols = ["Passengerid", "Age", "Fare", "Sex", "sibsp", "Parch", "Pclass", "Embarked", "2urvived"]
    df = df[[c for c in cols if c in df.columns]].copy()
    if "2urvived" in df.columns:
        df.rename(columns={"2urvived": "Survived"}, inplace=True)

    # 2) 결측치 대체: Embarked 최빈값으로 채우기
    embarked_mode = df["Embarked"].mode()[0]
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    print(f"[*] Embarked 결측치 2건 -> 최빈값({embarked_mode})으로 대체 완료")

    # 3) 특성 공학(Feature Engineering)
    # - FamilySize: 동승 가족 수 + 본인 1명
    df["FamilySize"] = df["sibsp"] + df["Parch"] + 1
    # - IsAlone: 혼자 탑승했는지 여부 (1: 나홀로 탑승, 0: 가족 동반)
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    # - Fare_Log: 요금(Fare)의 극단적 왜도 완화를 위한 로그 변환
    df["Fare_Log"] = np.log1p(df["Fare"])

    print(f"[*] 특성 생성 완료: FamilySize, IsAlone, Fare_Log")
    print(f"[*] 전처리 후 데이터 형상: {df.shape} (행: {df.shape[0]}, 열: {df.shape[1]})")

    return df


def visualize_eda(df, output_path="eda_survival_analysis.png"):
    """2. 생존율과 주요 특성 간의 관계 심층 시각화"""
    print("\n" + "=" * 60)
    print(f"2. EDA 시각화 생성 중 -> {output_path}")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("타이타닉 생존 요인 탐색적 데이터 분석 (EDA)", fontsize=18, fontweight="bold", y=0.98)

    # 1) 성별에 따른 생존율 (0: 남성, 1: 여성)
    ax1 = axes[0, 0]
    sns.barplot(x="Sex", y="Survived", data=df, ax=ax1, palette=["#3498db", "#e74c3c"], hue="Sex", legend=False)
    ax1.set_title("1) 성별 생존율 (0: 남성, 1: 여성)", fontsize=13, fontweight="bold")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["남성 (Male)", "여성 (Female)"])
    ax1.set_ylabel("생존율 (Survival Rate)")
    ax1.set_ylim(0, 1.0)
    for p in ax1.patches:
        height = p.get_height()
        ax1.annotate(f"{height * 100:.1f}%", (p.get_x() + p.get_width() / 2.0, height),
                     ha="center", va="bottom", fontsize=11, fontweight="bold")

    # 2) 객실 등급(Pclass)에 따른 생존율
    ax2 = axes[0, 1]
    sns.barplot(x="Pclass", y="Survived", data=df, ax=ax2, palette="Blues_r", hue="Pclass", legend=False)
    ax2.set_title("2) 객실 등급별 생존율", fontsize=13, fontweight="bold")
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["1등석 (First)", "2등석 (Second)", "3등석 (Third)"])
    ax2.set_ylabel("생존율 (Survival Rate)")
    ax2.set_ylim(0, 1.0)
    for p in ax2.patches:
        height = p.get_height()
        ax2.annotate(f"{height * 100:.1f}%", (p.get_x() + p.get_width() / 2.0, height),
                     ha="center", va="bottom", fontsize=11, fontweight="bold")

    # 3) 성별 x 객실 등급 교차 생존율
    ax3 = axes[0, 2]
    sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df, ax=ax3, palette=["#3498db", "#e74c3c"])
    ax3.set_title("3) 등급 및 성별 조합 생존율", fontsize=13, fontweight="bold")
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(["1등석", "2등석", "3등석"])
    ax3.set_ylabel("생존율")
    ax3.legend(["남성", "여성"], title="성별")
    ax3.set_ylim(0, 1.0)

    # 4) 나이(Age) 분포와 생존 여부 (KDE)
    ax4 = axes[1, 0]
    sns.kdeplot(df[df["Survived"] == 1]["Age"], ax=ax4, label="생존 (Survived)", fill=True, color="#2ecc71", alpha=0.5)
    sns.kdeplot(df[df["Survived"] == 0]["Age"], ax=ax4, label="사망 (Dead)", fill=True, color="#e74c3c", alpha=0.3)
    ax4.set_title("4) 나이 분포 vs 생존 여부", fontsize=13, fontweight="bold")
    ax4.set_xlabel("나이 (Age)")
    ax4.set_ylabel("밀도 (Density)")
    ax4.legend()

    # 5) 가족 수(FamilySize)에 따른 생존율
    ax5 = axes[1, 1]
    sns.barplot(x="FamilySize", y="Survived", data=df, ax=ax5, palette="viridis", hue="FamilySize", legend=False)
    ax5.set_title("5) 동승 가족 수(FamilySize)별 생존율", fontsize=13, fontweight="bold")
    ax5.set_xlabel("총 가족 수 (본인 포함)")
    ax5.set_ylabel("생존율")
    ax5.set_ylim(0, 1.0)
    for p in ax5.patches:
        height = p.get_height()
        ax5.annotate(f"{height * 100:.0f}%", (p.get_x() + p.get_width() / 2.0, height),
                     ha="center", va="bottom", fontsize=10)

    # 6) 수치형 특성 간 상관관계 히트맵 (Correlation Heatmap)
    ax6 = axes[1, 2]
    corr_cols = ["Survived", "Pclass", "Sex", "Age", "Fare_Log", "FamilySize", "IsAlone"]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax6, vmin=-0.6, vmax=0.6)
    ax6.set_title("6) 주요 특성 간 상관계수 히트맵", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[V] EDA 시각화 차트 저장 완료: {output_path}")


def train_and_evaluate_models(df):
    """3. 머신러닝 예측 모델 훈련 및 성능 평가"""
    print("\n" + "=" * 60)
    print("3. 머신러닝 예측 모델 학습 및 평가")
    print("=" * 60)

    # 특성(X) 및 타깃(y) 분리
    feature_cols = ["Pclass", "Sex", "Age", "Fare_Log", "Embarked", "FamilySize", "IsAlone"]
    X = df[feature_cols].copy()
    y = df["Survived"].copy()

    # 데이터 분할 (Train: 80%, Test: 20%) - 층화 추출(Stratify) 적용
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] 훈련 데이터 크기: {X_train.shape[0]}개, 테스트 데이터 크기: {X_test.shape[0]}개")

    # 스케일링 (로지스틱 회귀 등 거리 기반 모델을 위한 정규화)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 비교할 모델 정의
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        # 로지스틱 회귀는 스케일링된 데이터 사용, 트리 기반 모델은 원본 데이터 사용
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
        trained_models[name] = model

        print(f"\n[{name}]")
        print(f"  • 정확도 (Accuracy) : {acc:.4f}")
        print(f"  • 정밀도 (Precision): {prec:.4f}")
        print(f"  • 재현율 (Recall)   : {rec:.4f}")
        print(f"  • F1-Score         : {f1:.4f}")
        print(f"  • ROC-AUC          : {auc:.4f}")

    return X_train, X_test, y_train, y_test, feature_cols, results, trained_models, scaler


def visualize_model_performance(results, trained_models, feature_cols, y_test, output_path="model_evaluation.png"):
    """4. 모델 성능 평가 지표 및 특성 중요도 시각화"""
    print("\n" + "=" * 60)
    print(f"4. 모델 평가 시각화 생성 중 -> {output_path}")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("머신러닝 생존 예측 모델 성능 비교 및 해석", fontsize=18, fontweight="bold", y=0.98)

    # 1) 모델별 주요 성능 지표 비교 막대 차트
    ax1 = axes[0, 0]
    metrics_df = pd.DataFrame({
        name: [m["Accuracy"], m["F1-Score"], m["ROC-AUC"]]
        for name, m in results.items()
    }, index=["정확도 (Accuracy)", "F1-Score", "ROC-AUC"]).T

    metrics_df.plot(kind="bar", ax=ax1, colormap="tab10", width=0.7)
    ax1.set_title("1) 모델별 성능 지표 비교", fontsize=13, fontweight="bold")
    ax1.set_ylabel("점수 (Score)")
    ax1.set_ylim(0.5, 1.0)
    ax1.set_xticklabels(metrics_df.index, rotation=0)
    ax1.legend(loc="lower right")

    # 2) 랜덤 포레스트 혼동 행렬(Confusion Matrix)
    ax2 = axes[0, 1]
    rf_pred = results["Random Forest"]["y_pred"]
    cm = confusion_matrix(y_test, rf_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax2,
                xticklabels=["사망 예측(0)", "생존 예측(1)"],
                yticklabels=["실제 사망(0)", "실제 생존(1)"])
    ax2.set_title("2) Random Forest 혼동 행렬 (Confusion Matrix)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("실제값 (Actual)")
    ax2.set_xlabel("예측값 (Predicted)")

    # 3) ROC Curve 비교
    ax3 = axes[1, 0]
    for name, m in results.items():
        fpr, tpr, _ = roc_curve(y_test, m["y_proba"])
        ax3.plot(fpr, tpr, label=f"{name} (AUC = {m['ROC-AUC']:.3f})", linewidth=2)
    ax3.plot([0, 1], [0, 1], "k--", label="무작위 예측선 (AUC = 0.5)")
    ax3.set_title("3) ROC 곡선 (ROC Curves)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("위양성률 (False Positive Rate)")
    ax3.set_ylabel("진양성률 (True Positive Rate)")
    ax3.legend(loc="lower right")

    # 4) 랜덤 포레스트 특성 중요도 (Feature Importance)
    ax4 = axes[1, 1]
    rf_model = trained_models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    importances.plot(kind="barh", color="#1abc9c", ax=ax4)
    ax4.set_title("4) Random Forest 특성 중요도 (Feature Importance)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("상대적 중요도")
    for i, v in enumerate(importances):
        ax4.text(v + 0.005, i, f"{v * 100:.1f}%", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[V] 모델 평가 시각화 차트 저장 완료: {output_path}")


def predict_passenger(model, feature_cols, pclass, sex, age, fare, embarked, sibsp, parch):
    """5. 새로운 가상 승객 정보로 생존 확률 예측"""
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    fare_log = np.log1p(fare)

    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "Fare_Log": fare_log,
        "Embarked": embarked,
        "FamilySize": family_size,
        "IsAlone": is_alone,
    }])[feature_cols]

    proba = model.predict_proba(input_df)[0][1]
    prediction = int(proba >= 0.5)
    return prediction, proba


def run_all():
    data_path = os.path.join(os.path.dirname(__file__), "data", "train_and_test2.csv")
    if not os.path.exists(data_path):
        print(f"[!] 데이터 파일을 찾을 수 없습니다: {data_path}")
        return

    # 1. 데이터 로드 및 전처리
    df = load_and_preprocess_data(data_path)

    # 2. EDA 심층 시각화
    eda_img = os.path.join(os.path.dirname(__file__), "eda_survival_analysis.png")
    visualize_eda(df, output_path=eda_img)

    # 3. 모델 훈련 및 평가
    X_train, X_test, y_train, y_test, feature_cols, results, trained_models, scaler = train_and_evaluate_models(df)

    # 4. 모델 성능 시각화
    model_img = os.path.join(os.path.dirname(__file__), "model_evaluation.png")
    visualize_model_performance(results, trained_models, feature_cols, y_test, output_path=model_img)

    # 5. 가상 승객 생존 예측 테스트
    print("\n" + "=" * 60)
    print("5. 가상 승객 생존 시뮬레이션 예측 테스트")
    print("=" * 60)

    best_model = trained_models["Random Forest"]
    
    # 승객 A: 1등석 28세 여성 (혼자 탑승, 요금 $150)
    pred_a, proba_a = predict_passenger(best_model, feature_cols, pclass=1, sex=1, age=28, fare=150.0, embarked=0, sibsp=0, parch=0)
    print(f"[승객 A] 1등석 28세 여성: {'생존 예측(O)' if pred_a == 1 else '사망 예측(X)'} (생존 확률: {proba_a * 100:.1f}%)")

    # 승객 B: 3등석 22세 남성 (혼자 탑승, 요금 $7.5)
    pred_b, proba_b = predict_passenger(best_model, feature_cols, pclass=3, sex=0, age=22, fare=7.5, embarked=2, sibsp=0, parch=0)
    print(f"[승객 B] 3등석 22세 남성: {'생존 예측(O)' if pred_b == 1 else '사망 예측(X)'} (생존 확률: {proba_b * 100:.1f}%)")

    # 승객 C: 2등석 5세 여아 (부모와 동승, 요금 $30)
    pred_c, proba_c = predict_passenger(best_model, feature_cols, pclass=2, sex=1, age=5, fare=30.0, embarked=2, sibsp=1, parch=2)
    print(f"[승객 C] 2등석 5세 여아(가족동반): {'생존 예측(O)' if pred_c == 1 else '사망 예측(X)'} (생존 확률: {proba_c * 100:.1f}%)")


if __name__ == "__main__":
    run_all()
