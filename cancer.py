import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import os

# --- 한글 폰트 설정 시작 ---
# 나눔 폰트 경로 (Streamlit Cloud의 기본 설치 경로)
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

if os.path.exists(font_path):
    # 경로에 폰트가 있으면 설정 적용
    font_prop = font_manager.FontProperties(fname=font_path)
    rc('font', family=font_prop.get_name())
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
else:
    # 로컬(윈도우) 환경 등을 위한 예외 처리
    plt.rc('font', family='Malgun Gothic')
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# 모델과 스케일러 불러오기
model = joblib.load("lung_model.pkl")
scaler = joblib.load("lung_scaler.pkl")

# 데이터 불러오기
df = pd.read_csv("lung_cancer_examples.csv")

st.title("흡연/음주 클러스터 예측")

# 사용자 입력
glucose = st.number_input("나이 입력", min_value=0.0)
bmi = st.number_input("흡연량 입력", min_value=0.0)
age = st.number_input("음주량 입력", min_value=0.0)

# 예측 버튼
if st.button("클러스터 예측"):

    # 입력 데이터 생성
    new_patient = pd.DataFrame(
        [[glucose, bmi, age]],
        columns=['나이', '흡연량', '음주량']
    )

    # 스케일링
    new_patient_scaled = scaler.transform(new_patient)

    # 예측
    pred_cluster = model.predict(new_patient_scaled)

    # 결과 출력
    st.success(f"예측된 클러스터: {pred_cluster[0]}")

    # 시각화
    plt.figure(figsize=(8,6))

    # 기존 데이터
    plt.scatter(
        df['흡연량'],
        df['음주량'],
        c=df['cluster'],
        alpha=0.5
    )

    # 새 데이터 표시
    plt.scatter(
        bmi,
        age,
        c='black',
        s=300,
        marker='X'
    )

    plt.xlabel('흡연량')
    plt.ylabel('음주량')
    plt.title('클러스터 시각화')

    # Streamlit에 그래프 출력
    st.pyplot(plt)
