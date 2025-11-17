# app_micro.py
import time
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from core_micro import (
    fetch_2min_data,
    fetch_1min_intraday,
    build_feature_frame,
    build_targets,
    get_feature_target_matrices,
    train_models,
    predict_latest,
)

# ---------- 한글 폰트 설정 (Windows 기준: 맑은 고딕) ---------- #
matplotlib.rcParams["font.family"] = "Gulim"
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 페이지 기본 설정 ---------- #
st.set_page_config(
    page_title="초단기 2분봉/1분봉 방향성 예측 툴",
    layout="wide",
)

st.title("⚡ 최근 60일 2분봉 + 실시간 1분봉 초단기 예측 웹앱")
st.caption("2분봉 60일로 학습하고, 1분봉 실시간 차트에서 시그널 + 예상 가격 확인")


# ---------- 세션 상태 초기화 ---------- #
def init_state():
    defaults = {
        "raw_df": None,          # 2분봉 데이터
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

    days = st.slider("최근 N일 (1~60일, 2분봉 학습용)", min_value=1, max_value=60, value=40, step=1)

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
    st.caption("① 2분봉 데이터 다운로드 → ② 피처/타깃 생성 → ③ 모델 학습 → ④ 1분봉 실시간 시그널")


# ---------- 메인 탭 구성 ---------- #
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1️⃣ 데이터 다운로드 (2분봉)",
        "2️⃣ 피처 & 타깃 생성",
        "3️⃣ 모델 학습",
        "4️⃣ 실시간 시그널 (1분봉)",
    ]
)


# ==================== 1) 데이터 탭 ==================== #
with tab1:
    st.subheader("1️⃣ 2분봉 데이터 다운로드 (프리/데이/애프터 포함, 주말 제외)")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("📥 2분봉 데이터 불러오기"):
            with st.spinner("데이터 다운로드 중..."):
                try:
                    df_raw = fetch_2min_data(ticker, days=days)
                except Exception as e:
                    st.error(f"데이터 다운로드 중 오류 발생: {e}")
                else:
                    if df_raw is None or df_raw.empty:
                        st.warning("데이터를 가져오지 못했습니다. 티커/기간을 다시 확인하세요.")
                    else:
                        st.session_state["raw_df"] = df_raw
                        st.success(
                            f"{ticker} 최근 {days}일 2분봉 데이터 다운로드 완료! "
                            "(프리/애프터 포함, 주말 제외)"
                        )

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
        st.write("📊 종가 간단 라인 차트 (최근 500캔들)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_raw["Close"].tail(500))
        ax.set_title(f"{ticker} 2분봉 종가 (최근 500캔들)")
        ax.set_xlabel("시간")
        ax.set_ylabel("가격")
        st.pyplot(fig)


# ==================== 2) 피처 & 타깃 생성 탭 ==================== #
with tab2:
    st.subheader("2️⃣ 피처 & 타깃 생성 (2분봉 기반)")

    df_raw = st.session_state["raw_df"]
    if df_raw is None:
        st.warning("먼저 2분봉 데이터를 다운로드 해주세요. (탭 1)")
    else:
        st.write(
            f"티커: **{ticker}**, 최근 **{days}일** 2분봉 기준으로 "
            f"피처/타깃을 생성합니다. (주말 제외)"
        )

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
            st.dataframe(
                dist_df.style.format(
                    {
                        "up_ratio": "{:.2%}",
                        "down_ratio": "{:.2%}",
                    }
                )
            )


# ==================== 3) 모델 학습 탭 ==================== #
with tab3:
    st.subheader("3️⃣ 모델 학습 (2분봉 피처 기반)")

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


# ==================== 4) 실시간 시그널 탭 (1분봉) ==================== #
with tab4:
    st.subheader("4️⃣ 실시간 시그널 (1분봉 / 현재가 / 캔들차트 + 예상 가격)")

    models = st.session_state["models"]
    model_df = st.session_state["model_df"]
    feature_cols = st.session_state["feature_cols"]
    horizons = st.session_state["horizons"]

    if models is None or model_df is None or feature_cols is None or horizons is None:
        st.warning("먼저 2분봉 기반 모델을 학습해 주세요. (탭 3)")
    else:
        # ----- 4-1. 예측 결과 (2분봉 최신 샘플 기준) ----- #
        latest_row = model_df.iloc[-1]
        probs = predict_latest(models, latest_row, feature_cols)

        st.markdown("### 🔮 현재(가장 최근 2분봉) 기준 예측 결과")

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

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(prob_df.index.astype(str), prob_df["up_prob"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("상승 확률")
        ax.set_xlabel("예측 타임프레임 (분)")
        ax.set_title("현재 2분봉 기준 각 타임프레임 상승 확률")
        st.pyplot(fig)

        st.markdown("---")

        # ----- 4-2. 실시간 1분봉 차트 & 현재가 + 예상가 ----- #
        st.markdown("### 🕯 1분봉 실시간 캔들 차트 + 현재가 + 예상 가격")

        # 상단: 새로고침 옵션
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
        with col_ctrl1:
            auto_refresh = st.checkbox("자동 새로고침 (5초)", value=False)
        with col_ctrl2:
            refresh_now = st.button("🔄 지금 새로고침")
        with col_ctrl3:
            n_candles = st.slider(
                "표시할 캔들 수 (1분봉)",
                min_value=50,
                max_value=500,
                value=150,
                step=10,
            )

        # 수동 새로고침 버튼 → 즉시 rerun
        if refresh_now:
            st.rerun()

        # 1분봉 데이터 불러오기
        with st.spinner("1분봉 데이터 불러오는 중..."):
            try:
                intraday_df = fetch_1min_intraday(ticker, days=3)
            except Exception as e:
                st.error(f"1분봉 데이터 다운로드 중 오류 발생: {e}")
                intraday_df = None

        if intraday_df is not None and not intraday_df.empty:
            df_plot = intraday_df.tail(n_candles).copy()

            last_price = df_plot["Close"].iloc[-1]
            last_time = df_plot.index[-1]

            # ===== 1분 뒤 / 3분 뒤 예상 가격 (단순 추세 기반) ===== #
            # 최근 N개로 선형회귀해서 기울기 추정 → 분당 변화량 * 1, 3
            reg_window = min(50, len(df_plot))
            y = df_plot["Close"].tail(reg_window).values
            x = np.arange(reg_window)

            if reg_window >= 2:
                slope, intercept = np.polyfit(x, y, 1)
                # x= reg_window-1 이 현재 캔들에 대응 → 여기서 +1, +3
                pred_1m = last_price + slope * 1
                pred_3m = last_price + slope * 3
            else:
                pred_1m = float("nan")
                pred_3m = float("nan")

            # 메인 레이아웃: 차트(좌) + 정보(우)
            chart_col, info_col = st.columns([4, 1])

            with chart_col:
                fig_c = go.Figure(
                    data=[
                        go.Candlestick(
                            x=df_plot.index,
                            open=df_plot["Open"],
                            high=df_plot["High"],
                            low=df_plot["Low"],
                            close=df_plot["Close"],
                            name="1분봉",
                        )
                    ]
                )

                # 예측 가격을 수평선으로 표시
                shapes = []
                if np.isfinite(pred_1m):
                    shapes.append(
                        dict(
                            type="line",
                            xref="paper",
                            x0=0,
                            x1=1,
                            yref="y",
                            y0=pred_1m,
                            y1=pred_1m,
                            line=dict(color="blue", width=1, dash="dot"),
                        )
                    )
                if np.isfinite(pred_3m):
                    shapes.append(
                        dict(
                            type="line",
                            xref="paper",
                            x0=0,
                            x1=1,
                            yref="y",
                            y0=pred_3m,
                            y1=pred_3m,
                            line=dict(color="orange", width=1, dash="dot"),
                        )
                    )

                fig_c.update_layout(
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=10, r=10, t=40, b=40),
                    height=450,
                    title=f"{ticker} 1분봉 캔들 (최근 {n_candles}개)",
                    shapes=shapes,
                )

                # 예측 레벨에 텍스트 힌트 (범례 대신)
                annotations = []
                if np.isfinite(pred_1m):
                    annotations.append(
                        dict(
                            xref="paper",
                            x=0.01,
                            y=pred_1m,
                            xanchor="left",
                            yanchor="bottom",
                            text="+1분 예상",
                            showarrow=False,
                            font=dict(size=10, color="blue"),
                        )
                    )
                if np.isfinite(pred_3m):
                    annotations.append(
                        dict(
                            xref="paper",
                            x=0.01,
                            y=pred_3m,
                            xanchor="left",
                            yanchor="bottom",
                            text="+3분 예상",
                            showarrow=False,
                            font=dict(size=10, color="orange"),
                        )
                    )
                if annotations:
                    fig_c.update_layout(annotations=annotations)

                st.plotly_chart(fig_c, use_container_width=True)

            with info_col:
                st.markdown("#### 💰 현재가")
                st.metric(label="Price", value=f"{last_price:,.2f}")

                st.markdown("#### 🔮 예상가")
                if np.isfinite(pred_1m):
                    st.metric(label="+1분 예상", value=f"{pred_1m:,.2f}")
                else:
                    st.write("+1분 예상: 계산 불가")

                if np.isfinite(pred_3m):
                    st.metric(label="+3분 예상", value=f"{pred_3m:,.2f}")
                else:
                    st.write("+3분 예상: 계산 불가")

                st.markdown("#### 🕒 시각")
                st.write(last_time.strftime("%Y-%m-%d %H:%M:%S"))

                # 장 상태 대략 표시 (시간대 기준)
                h = last_time.hour
                if 4 <= h < 9:
                    st.caption("프리장(Pre-market) 추정")
                elif 9 <= h < 16:
                    st.caption("정규장(Regular) 추정")
                else:
                    st.caption("애프터장(After-hours) 추정")

                st.markdown("---")
                st.caption("※ 1/3분 예상가는 최근 추세(선형회귀) 기반 단순 추정값입니다.")

            st.markdown("#### 🔎 최근 1분봉 원시 데이터 (마지막 5개 캔들)")
            st.dataframe(intraday_df.tail(5))
        else:
            st.info("1분봉 데이터를 가져오지 못했습니다. 티커/시간대를 다시 확인해 주세요.")

        # 자동 새로고침 로직 (간단한 5초 주기)
        if auto_refresh:
            time.sleep(5)
            st.rerun()
