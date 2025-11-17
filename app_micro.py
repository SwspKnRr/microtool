# app_micro.py
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import streamlit as st

from core_micro import (
    fetch_1min_data,
    build_feature_frame,
    build_targets,
    get_feature_target_matrices,
    train_models,
    predict_latest,
)

# ---------- 한글 폰트 설정 (Windows 기준: 굴림) ---------- #
matplotlib.rcParams["font.family"] = "Gulim"
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 페이지 기본 설정 ---------- #
st.set_page_config(
    page_title="초단기 1분봉 방향성 예측 툴",
    layout="wide",
)

st.title("⚡ 1달 1분봉 기반 초단기 방향성 예측 웹앱")
st.caption("최근 1개월 1분봉 데이터를 기반으로 5/10/30분 + 사용자 정의 X분 후 상승 확률 예측")


# ---------- 세션 상태 초기화 ---------- #
def init_state():
    defaults = {
        "raw_df": None,
        "feat_df": None,
        "model_df": None,
        "horizons": None,
        "X": None,
        "y_dict": None,
        "feature_cols": None,
        "models": None,
        "metrics": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ---------- 사이드바 설정 ---------- #
with st.sidebar:
    st.header("⚙ 설정")

    ticker = st.text_input("티커 (예: SPY, QQQ, AAPL 등)", value="QQQ")

    days = st.slider("최근 N일 (1~30일)", min_value=1, max_value=30, value=20, step=1)

    st.markdown("---")
    st.subheader("⏱ 예측 타임프레임")

    base_horizons = st.multiselect(
        "기본 예측 분 단위 (여러 개 선택 가능)",
        options=[5, 10, 30],
        default=[5, 10, 30],
    )

    custom_h = st.number_input(
        "사용자 정의 X분 (1~60분)",
        min_value=1,
        max_value=60,
        value=15,
        step=1,
    )

    st.markdown("---")
    random_state = st.number_input(
        "Random Seed (재현성용)", min_value=0, max_value=9999, value=42, step=1
    )

    st.markdown("---")
    st.caption("① 데이터 다운로드 → ② 피처/타깃 생성 → ③ 모델 학습 → ④ 실시간 시그널 확인")


# ---------- 메인 탭 구성 ---------- #
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1️⃣ 데이터 다운로드",
        "2️⃣ 피처 & 타깃 생성",
        "3️⃣ 모델 학습",
        "4️⃣ 실시간 시그널",
    ]
)


# ==================== 1) 데이터 탭 ==================== #
with tab1:
    st.subheader("1️⃣ 1분봉 데이터 다운로드")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("📥 1분봉 데이터 불러오기"):
            with st.spinner("데이터 다운로드 중..."):
                try:
                    df_raw = fetch_1min_data(ticker, days=days)
                except Exception as e:
                    st.error(f"데이터 다운로드 중 오류 발생: {e}")
                else:
                    if df_raw is None or df_raw.empty:
                        st.warning("데이터를 가져오지 못했습니다. 티커/기간을 다시 확인하세요.")
                    else:
                        st.session_state["raw_df"] = df_raw
                        st.success(f"{ticker} 최근 {days}일 1분봉 데이터 다운로드 완료!")

    with col2:
        df_raw = st.session_state["raw_df"]
        if df_raw is not None:
            st.write("🔹 데이터 샘플 (최근 10개)")
            st.dataframe(df_raw.tail(10))
        else:
            st.info("좌측에서 데이터를 먼저 다운로드하세요.")

    if st.session_state["raw_df"] is not None:
        df_raw = st.session_state["raw_df"]
        st.markdown("---")
        st.write("📊 종가 간단 차트")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_raw["Close"].tail(500))  # 최근 500개 정도만
        ax.set_title(f"{ticker} 1분봉 종가 (최근 500개)")
        ax.set_xlabel("시간")
        ax.set_ylabel("가격")
        st.pyplot(fig)


# ==================== 2) 피처 & 타깃 생성 탭 ==================== #
with tab2:
    st.subheader("2️⃣ 피처 & 타깃 생성")

    df_raw = st.session_state["raw_df"]
    if df_raw is None:
        st.warning("먼저 1분봉 데이터를 다운로드 해주세요. (탭 1)")
    else:
        st.write(f"티커: **{ticker}**, 최근 **{days}일** 1분봉 기준으로 피처/타깃을 생성합니다.")

        if st.button("🧮 피처 & 타깃 만들기"):
            with st.spinner("피처/타깃 생성 중..."):
                try:
                    feat_df = build_feature_frame(df_raw)
                    model_df, horizons = build_targets(
                        feat_df,
                        base_horizons=base_horizons,
                        custom_horizon=int(custom_h) if custom_h else None,
                        threshold=0.0,
                    )
                    X, y_dict, feature_cols = get_feature_target_matrices(
                        model_df, horizons
                    )
                except Exception as e:
                    st.error(f"피처/타깃 생성 중 오류 발생: {e}")
                else:
                    st.session_state["feat_df"] = feat_df
                    st.session_state["model_df"] = model_df
                    st.session_state["horizons"] = horizons
                    st.session_state["X"] = X
                    st.session_state["y_dict"] = y_dict
                    st.session_state["feature_cols"] = feature_cols

                    st.success(
                        f"피처/타깃 생성 완료! 사용되는 샘플 수: {model_df.shape[0]:,}개, "
                        f"예측 타임프레임: {horizons}분"
                    )

        model_df = st.session_state["model_df"]
        horizons = st.session_state["horizons"]

        if model_df is not None and horizons is not None:
            st.markdown("### 🔍 피처/타깃 데이터 샘플")
            st.dataframe(model_df.tail(10))

            st.markdown("### 📈 타깃 분포 (상승/하락 비율)")
            rows = []
            for h in horizons:
                y = model_df[f"y_{h}"]
                up_ratio = (y == 1).mean()
                rows.append(
                    {
                        "horizon_min": h,
                        "samples": int(len(y)),
                        "up_ratio": up_ratio,
                        "down_ratio": 1 - up_ratio,
                    }
                )
            dist_df = pd.DataFrame(rows).set_index("horizon_min")
            st.dataframe(dist_df.style.format({"up_ratio": "{:.2%}", "down_ratio": "{:.2%}"}))


# ==================== 3) 모델 학습 탭 ==================== #
with tab3:
    st.subheader("3️⃣ 모델 학습")

    X = st.session_state["X"]
    y_dict = st.session_state["y_dict"]
    horizons = st.session_state["horizons"]

    if X is None or y_dict is None or horizons is None:
        st.warning("먼저 피처/타깃을 생성해 주세요. (탭 2)")
    else:
        st.write(
            f"총 샘플 수: **{X.shape[0]:,}** 개, "
            f"예측 타임프레임: **{horizons} 분**"
        )

        if st.button("🤖 RandomForest로 모델 학습하기"):
            with st.spinner("모델 학습 중..."):
                try:
                    models, metrics_df = train_models(
                        X, y_dict, random_state=random_state
                    )
                except Exception as e:
                    st.error(f"모델 학습 중 오류 발생: {e}")
                else:
                    st.session_state["models"] = models
                    st.session_state["metrics"] = metrics_df
                    st.success("모델 학습 완료!")

        metrics_df = st.session_state["metrics"]
        if metrics_df is not None:
            st.markdown("### 📊 성능 지표 (테스트 구간)")
            st.dataframe(
                metrics_df.style.format(
                    {
                        "accuracy": "{:.3f}",
                        "precision": "{:.3f}",
                        "recall": "{:.3f}",
                    }
                )
            )


# ==================== 4) 실시간 시그널 탭 ==================== #
with tab4:
    st.subheader("4️⃣ 실시간 시그널 (가장 최근 시점 기준)")

    models = st.session_state["models"]
    model_df = st.session_state["model_df"]
    feature_cols = st.session_state["feature_cols"]
    horizons = st.session_state["horizons"]

    if models is None or model_df is None or feature_cols is None or horizons is None:
        st.warning("먼저 모델을 학습해 주세요. (탭 3)")
    else:
        latest_row = model_df.iloc[-1]
        probs = predict_latest(models, latest_row, feature_cols)

        st.markdown("### 🔮 현재 시점 기준 예측 결과")

        # 표로 출력
        rows = []
        for h in sorted(probs.keys()):
            rows.append(
                {
                    "horizon_min": h,
                    "up_prob": probs[h],
                }
            )
        prob_df = pd.DataFrame(rows).set_index("horizon_min")
        st.dataframe(prob_df.style.format({"up_prob": "{:.2%}"}))

        # 막대그래프
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(prob_df.index.astype(str), prob_df["up_prob"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("상승 확률")
        ax.set_xlabel("예측 타임프레임 (분)")
        ax.set_title("현재 캔들 기준 각 타임프레임 상승 확률")
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("### 🕒 가장 최근 캔들 정보")

        raw_df = st.session_state["raw_df"]
        if raw_df is not None:
            st.write(raw_df.tail(5))
        else:
            st.info("원시 데이터(raw_df)가 세션에 없습니다. 탭 1에서 다시 다운로드해 주세요.")
