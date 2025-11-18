# app_micro.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from core_micro import (
    fetch_1min_intraday,
    generate_prediction,
    backtest_prediction,
    run_full_pipeline,
)

#############################################
# Streamlit 기본 설정
#############################################
st.set_page_config(page_title="초단기 예측 툴 (micro)", layout="wide")
st.title("⚡ 초단기 미시 예측 시스템 v1.0")

#############################################
# 캐싱 (새로고침해도 전체 재계산 방지)
#############################################
@st.cache_data(show_spinner=False)
def cached_fetch(ticker, days):
    return fetch_1min_intraday(ticker, days)

@st.cache_data(show_spinner=False)
def cached_prediction(df):
    return generate_prediction(df)

#############################################
# 사이드바 입력
#############################################
st.sidebar.header("📌 입력값")
ticker = st.sidebar.text_input("티커", value="SPY")
days = st.sidebar.number_input("수집 일수(1~10)", 1, 10, 3)


#############################################
# 탭 구성
#############################################
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1) 데이터", "2) 예측(5분)", "3) 예측(15분)", "4) 전체 일괄 실행", "5) 예측 정확도 검증"]
)

#############################################
# 1번탭: Raw 데이터
#############################################
with tab1:
    st.subheader("📌 1분봉 데이터")
    df = cached_fetch(ticker, days)
    if df is None or df.empty:
        st.warning("데이터 없음")
    else:
        st.dataframe(df.tail(200))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["Close"])
        ax.set_title(f"{ticker} 1분봉 Close")
        st.pyplot(fig)

#############################################
# 2번탭: 5분 예측
#############################################
with tab2:
    st.subheader("📌 5분 예측 결과")

    if df is None or df.empty:
        st.warning("데이터 없음")
    else:
        pred = generate_prediction(df, [5])
        df5 = pred[5]

        st.dataframe(df5.tail(200))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df5["Close"], label="Close")
        ax.plot(df5["pred_5m"], label="Pred 5m")
        ax.legend()
        st.pyplot(fig)

#############################################
# 3번탭: 15분 예측
#############################################
with tab3:
    st.subheader("📌 15분 예측 결과")

    if df is None or df.empty:
        st.warning("데이터 없음")
    else:
        pred = generate_prediction(df, [15])
        df15 = pred[15]

        st.dataframe(df15.tail(200))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df15["Close"], label="Close")
        ax.plot(df15["pred_15m"], label="Pred 15m")
        ax.legend()
        st.pyplot(fig)

#############################################
# 4번탭: 전체 일괄 실행
#############################################
with tab4:
    st.subheader("📌 전체 파이프라인 실행 (1→2→3 자동)")

    if st.button("🚀 전체 실행"):
        result = run_full_pipeline(ticker, days)
        if result is None:
            st.error("실행 실패")
        else:
            st.success("전체 파이프라인 실행 완료!")

            st.write("### Raw")
            st.dataframe(result["raw"].tail(100))

            st.write("### Predictions")
            for h, df_h in result["predictions"].items():
                st.write(f"#### Horizon = {h}분")
                st.dataframe(df_h.tail(50))

#############################################
# 5번탭: 과거 예측 얼마나 맞았나?
#############################################
with tab5:
    st.subheader("📌 예측 정확도 검증")

    horizon = st.number_input("Horizon (분)", 1, 60, 5)

    if df is None or df.empty:
        st.warning("데이터 없음")
    else:
        acc, df_test = backtest_prediction(df, horizon=horizon)

        st.write(f"### 👍 예측 방향 정확도: **{acc*100:.2f}%**")

        st.dataframe(df_test.tail(200))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_test["Close"], label="Close")
        ax.plot(df_test["pred"], label="Pred")
        ax.legend()
        st.pyplot(fig)
