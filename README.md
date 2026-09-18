# Titanic Dataset Project (Week 3)

Kaggle `heptapod/titanic` 데이터셋을 활용한 분석 프로젝트 구조입니다.

## 📁 프로젝트 구조

```text
week3/
├── data/
│   └── train_and_test2.csv   # 다운로드된 타이타닉 데이터셋
├── titanic_full_analysis_and_modeling.ipynb  # [신규] EDA + 머신러닝 모델링 통합 주피터 노트북
├── titanic_eda.ipynb         # 데이터 품질 기초 검증 주피터 노트북
├── predict_model.py          # [신규] EDA 심층 시각화 및 머신러닝 예측 모델링 실행 스크립트
├── eda_basics.py             # 결측치, 중복값, 이상치 검증 및 시각화 종합 스크립트
├── eda_survival_analysis.png # [신규] 생존 요인 EDA 6종 종합 시각화 대시보드
├── model_evaluation.png      # [신규] 머신러닝 모델 평가 및 특성 중요도 차트
├── eda_summary.png           # 데이터 무결성 검증 대시보드
├── download_data.py          # kagglehub를 이용해 데이터셋을 받아 data/ 폴더로 복사하는 스크립트
├── main.py                   # pandas로 데이터를 불러와 기본 정보를 확인하는 예제 스크립트
├── requirements.txt          # 필요 라이브러리 목록 (scikit-learn 포함)
└── README.md
```

## 🚀 사용 방법

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 머신러닝 예측 모델 및 시각화 한 번에 실행
```bash
python predict_model.py
```

### 3. 대화형 주피터 노트북 실행
```bash
# 통합 분석 및 모델링 노트북 실행
jupyter notebook titanic_full_analysis_and_modeling.ipynb
```
