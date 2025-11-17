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

# ---------- 페이지 기본 설정 ---------- #단타로 과자 먹자
st.set_page_config(
    page_title="최근 60일 2분봉 학습 / 실시간 1분봉 예측 웹앱",
    layout="wide",
)

st.title("⚡ 단타로 과자 먹자")
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
    st.subheader("4️⃣ 실시간 시그널 (1분봉 / 현재가 / 모델 보정 예상가 + 과거 예측 검증)")

    models = st.session_state["models"]
    model_df = st.session_state["model_df"]
    feature_cols = st.session_state["feature_cols"]
    horizons = st.session_state["horizons"]

    if models is None or model_df is None or feature_cols is None or horizons is None:
        st.warning("먼저 2분봉 기반 모델을 학습해 주세요. (탭 3)")
    else:
        # ----- 4-1. 예측 결과 (2분봉 최신 샘플 기준 / 테이블만) ----- #
        latest_row = model_df.iloc[-1]
        probs = predict_latest(models, latest_row, feature_cols)  # {model_horizon: p_up}

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

        st.markdown("---")

        # ----- 4-2. 실시간 1분봉 차트 & 현재가 + 모델 보정 예상가 ----- #
        st.markdown("### 🕯 1분봉 실시간 캔들 차트 + 현재가 + 모델 보정 예상가")

        # 상단: 새로고침 옵션 + 캔들 수
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.2, 1.2, 2.6])
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

        # 예상 시간 체크박스들
        st.markdown("#### ⏱ 예상 가격 표시 옵션")
        col_pred1, col_pred2, col_pred3, col_pred4, col_pred5 = st.columns(5)
        with col_pred1:
            show_1 = st.checkbox("+1분", value=True)
        with col_pred2:
            show_3 = st.checkbox("+3분", value=True)
        with col_pred3:
            show_10 = st.checkbox("+10분", value=False)
        with col_pred4:
            show_30 = st.checkbox("+30분", value=False)
        with col_pred5:
            show_60 = st.checkbox("+60분", value=False)

        col_pred6, col_pred7, col_pred8 = st.columns(3)
        with col_pred6:
            show_120 = st.checkbox("+2시간", value=False)   # 120분
        with col_pred7:
            show_300 = st.checkbox("+5시간", value=False)   # 300분
        with col_pred8:
            show_close = st.checkbox("종가", value=False)

        horizon_flags = {
            1: show_1,
            3: show_3,
            10: show_10,
            30: show_30,
            60: show_60,
            120: show_120,
            300: show_300,
        }

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

            # ===== 최근 추세 기반 여러 시간대 예상 가격 + 모델 확률 보정 ===== #
            reg_window = min(50, len(df_plot))
            y = df_plot["Close"].tail(reg_window).values
            x = np.arange(reg_window)

            preds: dict[int, float] = {}  # {horizon_min: adjusted_price}
            pred_close = None  # 종가 예상 (있으면 float)

            # 모델 horizon 리스트와 probs dict에서 쓸 키 준비
            model_horizons = list(probs.keys())

            def get_nearest_model_prob(target_min: int) -> float | None:
                """사용자 horizon(분)을 가장 가까운 모델 horizon과 매칭해서 p_up 가져오기."""
                if not model_horizons:
                    return None
                nearest_h = min(model_horizons, key=lambda H: abs(H - target_min))
                return probs.get(nearest_h, None)

            if reg_window >= 2:
                slope, intercept = np.polyfit(x, y, 1)

                # 1) 일반 horizon들 (1/3/10/30/60/120/300분)
                for h_min, flag in horizon_flags.items():
                    if not flag:
                        continue

                    # 단순 추세 기반 예상가
                    p_trend = last_price + slope * h_min

                    # 모델 확률
                    p_up = get_nearest_model_prob(h_min)

                    if p_up is None:
                        preds[h_min] = p_trend
                    else:
                        # 확률 기반 가중치 (조금 더 공격적으로)
                        base = 0.3  # 최소 추세 비중
                        confidence = 2 * abs(p_up - 0.5)  # 0~1
                        w = base + (1 - base) * confidence
                        w = float(np.clip(w, 0.0, 1.0))

                        p_adj = (1 - w) * last_price + w * p_trend
                        preds[h_min] = p_adj

                # 2) 종가 예상
                if show_close:
                    # 미국장 기준: 정규장 9~16시 (실제로는 9:30~지만 여기선 단순화)
                    hour = last_time.hour
                    minute = last_time.minute
                    if 9 <= hour < 16:
                        # 오늘 16:00 기준으로 남은 분 수
                        close_dt = last_time.replace(hour=16, minute=0, second=0, microsecond=0)
                        minutes_to_close = int((close_dt - last_time).total_seconds() // 60)
                        if minutes_to_close > 0:
                            p_trend_close = last_price + slope * minutes_to_close
                            p_up_close = get_nearest_model_prob(minutes_to_close)

                            if p_up_close is None:
                                pred_close = p_trend_close
                            else:
                                base = 0.3
                                confidence = 2 * abs(p_up_close - 0.5)
                                w = base + (1 - base) * confidence
                                w = float(np.clip(w, 0.0, 1.0))
                                pred_close = (1 - w) * last_price + w * p_trend_close
                        else:
                            pred_close = None
                    else:
                        pred_close = None  # 정규장 아닐 때는 종가 예상 안 함

            # ===== 30분 전에 예상했던 현재가 (과거 예측 검증) ===== #
            back_result = None  # dict 형태로 저장 예정

            try:
                t_now = intraday_df.index[-1]
                t_back = t_now - pd.Timedelta(minutes=30)

                # 1분봉 기준 30분 전까지의 구간에서 다시 추세선 추정
                intraday_back = intraday_df[intraday_df.index <= t_back]
                if len(intraday_back) >= 10:
                    back_window = min(50, len(intraday_back))
                    y_back = intraday_back["Close"].tail(back_window).values
                    x_back = np.arange(back_window)
                    slope_back, intercept_back = np.polyfit(x_back, y_back, 1)

                    price_back = intraday_back["Close"].iloc[-1]

                    # 그 시점에서 "30분 뒤" (지금) 가격에 대한 추세 기반 예상
                    p_trend_back_30 = price_back + slope_back * 30

                    # 같은 시점의 2분봉 모델 확률 복원
                    df2 = model_df
                    # 2분봉 인덱스에서 t_back 이전/같은 시점 중 가장 최근 것
                    idx_candidates = df2.index[df2.index <= t_back]
                    if len(idx_candidates) > 0:
                        idx_back = idx_candidates[-1]
                        past_row = df2.loc[idx_back]
                        past_probs = predict_latest(models, past_row, feature_cols)

                        # 30분에 가장 가까운 horizon 사용
                        model_hs_back = list(past_probs.keys())
                        nearest_h_back = min(model_hs_back, key=lambda H: abs(H - 30))
                        p_up_back = past_probs[nearest_h_back]

                        # 가중치 계산 (현재와 동일 로직)
                        base = 0.3
                        confidence = 2 * abs(p_up_back - 0.5)
                        w_back = base + (1 - base) * confidence
                        w_back = float(np.clip(w_back, 0.0, 1.0))

                        p_adj_back_30 = (1 - w_back) * price_back + w_back * p_trend_back_30

                        error = last_price - p_adj_back_30
                        error_pct = error / last_price if last_price != 0 else np.nan

                        back_result = {
                            "pred": p_adj_back_30,
                            "actual": last_price,
                            "error": error,
                            "error_pct": error_pct,
                            "time_back": intraday_back.index[-1],
                        }
            except Exception:
                back_result = None  # 에러 나면 걍 안 보여줌

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
                            increasing=dict(
                               line=dict(color="#FF4949"),   # 파스텔 레드
                               fillcolor="#FF4949",
                            ),
                             decreasing=dict(
                               line=dict(color="#3C87FF"),   # 파스텔 블루
                               fillcolor="#3C87FF",
                            ),
                            name="1분봉",
                        )
                    ]
                )

                # 👇 이 아래에 추가
                fig_c.update_layout(
                    dragmode=False,
                     xaxis=dict(fixedrange=True),
                     yaxis=dict(fixedrange=True),
                     modebar_remove=[
                          "zoom",
                     ]
                    )

                # 예측 가격 수평선 + annotation
                shapes = []
                annotations = []

                # annotation의 x 위치를 가로로 분산
                x_positions = {
                    1: 0.05,
                    3: 0.20,
                    10: 0.35,
                    30: 0.50,
                    60: 0.65,
                    120: 0.80,
                    300: 0.95,
                }
                colors = {
                    1: "blue",
                    3: "orange",
                    10: "green",
                    30: "purple",
                    60: "red",
                    120: "brown",
                    300: "darkcyan",
                }

                for h_min, price in preds.items():
                    if not np.isfinite(price):
                        continue

                    shapes.append(
                        dict(
                            type="line",
                            xref="paper",
                            x0=0,
                            x1=1,
                            yref="y",
                            y0=price,
                            y1=price,
                            line=dict(color=colors.get(h_min, "gray"), width=1, dash="dot"),
                        )
                    )

                    x_anno = x_positions.get(h_min, 0.5)
                    annotations.append(
                        dict(
                            xref="paper",
                            x=x_anno,
                            y=price,
                            xanchor="center",
                            yanchor="bottom",
                            text=f"+{h_min}분",
                            showarrow=False,
                            font=dict(size=10, color=colors.get(h_min, "gray")),
                        )
                    )

                # 종가 예상도 차트에 표시
                if pred_close is not None and np.isfinite(pred_close):
                    shapes.append(
                        dict(
                            type="line",
                            xref="paper",
                            x0=0,
                            x1=1,
                            yref="y",
                            y0=pred_close,
                            y1=pred_close,
                            line=dict(color="black", width=1, dash="dash"),
                        )
                    )
                    annotations.append(
                        dict(
                            xref="paper",
                            x=0.5,
                            y=pred_close,
                            xanchor="center",
                            yanchor="bottom",
                            text="종가",
                            showarrow=False,
                            font=dict(size=10, color="black"),
                        )
                    )

                fig_c.update_layout(
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=10, r=10, t=40, b=40),
                    height=450,
                    title=f"{ticker} 1분봉 캔들 (최근 {n_candles}개)",
                    shapes=shapes,
                    annotations=annotations,
                )

                st.plotly_chart(fig_c, use_container_width=True)

            with info_col:
                st.markdown("#### 💰 현재가")
                st.metric(label="Price", value=f"{last_price:,.2f}")

                st.markdown("#### 🔮 모델 보정 예상가")
                if preds:
                    for h_min in sorted(preds.keys()):
                        price = preds[h_min]
                        st.metric(label=f"+{h_min}분", value=f"{price:,.2f}")
                else:
                    st.write("예상가: 계산 불가 (데이터 또는 모델 확률 부족)")

                if pred_close is not None and np.isfinite(pred_close):
                    st.metric(label="종가 예상", value=f"{pred_close:,.2f}")

                st.markdown("#### ⏪ 30분 전 예측 vs 현재")
                if back_result is not None:
                    st.write(
                        f"30분 전 시점: {back_result['time_back'].strftime('%H:%M')}"
                    )
                    st.write(f"그때 30분 뒤 예상가: {back_result['pred']:.2f}")
                    st.write(f"현재 실제가: {back_result['actual']:.2f}")
                    st.write(
                        f"오차: {back_result['error']:+.2f} ({back_result['error_pct']*100:+.2f}%)"
                    )
                else:
                    st.write("30분 전 예측값을 계산할 수 있는 데이터가 부족합니다.")

                st.markdown("#### 🕒 시각")
                st.write(last_time.strftime("%Y-%m-%d %H:%M:%S"))

                # 장 상태 대략 표시 (시간대 기준, 미국장 가정)
                h = last_time.hour
                if 4 <= h < 9:
                    st.caption("프리장(Pre-market) 추정")
                elif 9 <= h < 16:
                    st.caption("정규장(Regular) 추정")
                else:
                    st.caption("애프터장(After-hours) 추정")

                st.markdown("---")
                st.caption(
                    "※ 예상 가격은 최근 1분봉 추세 + 2분봉 모델 상승 확률을 함께 반영한 단순 보정값입니다.\n"
                    "※ 30분 전 예측 비교는 '그때의 추세 + 그때의 모델 확률'로 복원한 값과 현재가의 차이를 보여줍니다."
                )

            st.markdown("#### 🔎 최근 1분봉 원시 데이터 (마지막 5개 캔들)")
            st.dataframe(intraday_df.tail(5))
        else:
            st.info("1분봉 데이터를 가져오지 못했습니다. 티커/시간대를 다시 확인해 주세요.")

        # 자동 새로고침 로직 (간단한 5초 주기)
        if auto_refresh:
            time.sleep(5)
            st.rerun()
