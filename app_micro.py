# app_micro.py (V2)

import time
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib

from zoneinfo import ZoneInfo

from core_micro import (
    fetch_2min_data,
    fetch_1min_intraday,
    build_feature_frame,
    build_targets,
    get_feature_target_matrices,
    train_models,
    predict_latest,
    compute_scaling_A,
    compute_scaling_B,
)

matplotlib.rcParams["font.family"] = "Gulim"
matplotlib.rcParams["axes.unicode_minus"] = False


# ===============================
# KST 변환
# ===============================
def to_kst(df: pd.DataFrame):
    if df is None or df.empty:
        return df
    idx = df.index
    if idx.tz is None:
        df.index = idx.tz_localize("UTC").tz_convert("Asia/Seoul")
    else:
        df.index = idx.tz_convert("Asia/Seoul")
    return df


# ===============================
# 미국 세션
# ===============================
def get_kst_session_times(use_dst):
    if use_dst:
        return dt.time(22, 30), dt.time(5, 0)
    else:
        return dt.time(23, 30), dt.time(6, 0)
        
        
# ===============================
# 세션 판별
# ===============================
def is_regular_kst(ts, open_kst, close_kst):
    t = ts.time()
    return (t >= open_kst) or (t < close_kst)


def minutes_to_close_kst(ts, open_kst, close_kst):
    if ts.tz is None:
        ts = ts.tz_localize("Asia/Seoul")

    t = ts.time()
    if not is_regular_kst(ts, open_kst, close_kst):
        return None

    if t >= open_kst:
        close_dt = ts.replace(
            hour=close_kst.hour,
            minute=close_kst.minute,
            second=0,
            microsecond=0
        ) + dt.timedelta(days=1)
    else:
        close_dt = ts.replace(
            hour=close_kst.hour,
            minute=close_kst.minute,
            second=0,
            microsecond=0
        )

    return max(0, int((close_dt - ts).total_seconds() // 60))


def session_mask_kst(times, open_kst, close_kst):
    out = []
    for ts in times:
        t = ts.time()
        if is_regular_kst(ts, open_kst, close_kst):
            out.append("regular")
        elif t < open_kst:
            out.append("premarket")
        else:
            out.append("after")
    return out



# ===============================
# Streamlit UI 설정
# ===============================
st.set_page_config(
    page_title="단타 예측 웹앱 V2",
    layout="wide"
)

st.title("⚡ 단타 예측 웹앱 V2 (Scaling + Regression Version)")


# 세션 변수 초기화
def init_state():
    keys = [
        "raw_df", "feat_df", "model_df", "horizons", "X",
        "y_dict", "feature_cols", "models", "metrics",
        "scaling_A", "scaling_B"
    ]
    for k in keys:
        if k not in st.session_state:
            st.session_state[k] = None


init_state()


# ===============================
# 사이드바
# ===============================
with st.sidebar:
    st.header("⚙ 기본 설정")

    ticker = st.text_input("티커", value="QQQ")
    days = st.slider("최근 N일 (2분봉)", 5, 60, 40)
    use_dst = st.checkbox("DST 적용", True)

    base_horizons = st.multiselect("예측 horizon(분)", [5, 10, 30], default=[5, 10, 30])
    custom_h = st.number_input("추가 horizon", 1, 60, 15)
    random_state = st.number_input("Random Seed", 0, 9999, 42)

    st.markdown("---")
    st.subheader("📐 스케일링 옵션")

    scaling_mode = st.selectbox(
        "스케일링 모드",
        ["NONE (기본 선형)", "A (통계 기반)", "B (ML 커브)"]
    )

    apply_scaling_live = st.checkbox("실시간 예측에도 스케일링 적용", value=False)
    apply_scaling_backtest = st.checkbox("힌드캐스트에도 스케일링 적용", value=True)


# ===============================
# 탭 구성
# ===============================
tab_live, tab_back = st.tabs(["1️⃣ 실시간 예측", "2️⃣ 하루 힌드캐스트"])


# ============================================================
# 1️⃣ 실시간 예측 탭
# ============================================================
with tab_live:

    st.subheader("🚀 원클릭 엔진 준비")

    if st.button("📡 2분봉 다운로드 + 피처/타깃 + 모델학습"):
        with st.spinner("2분봉 다운로드 중..."):
            raw = fetch_2min_data(ticker, days)
            raw = to_kst(raw)

        with st.spinner("엔진 학습 중..."):
            feat = build_feature_frame(raw)
            model_df, horizons = build_targets(feat, base_horizons, custom_h)
            X, y_dict, feat_cols = get_feature_target_matrices(model_df, horizons)

            models, metrics = train_models(X, y_dict, random_state)

        st.session_state["raw_df"] = raw
        st.session_state["feat_df"] = feat
        st.session_state["model_df"] = model_df
        st.session_state["horizons"] = horizons
        st.session_state["X"] = X
        st.session_state["y_dict"] = y_dict
        st.session_state["feature_cols"] = feat_cols
        st.session_state["models"] = models
        st.session_state["metrics"] = metrics

        st.success("엔진 준비 완료!")

    metrics = st.session_state["metrics"]
    if metrics is not None:
        st.markdown("### 📊 엔진 성능 요약")
        st.dataframe(metrics, use_container_width=True)

    st.markdown("---")
    st.subheader("🔮 실시간 1분봉 예측")

    models = st.session_state["models"]
    feat_cols = st.session_state["feature_cols"]
    model_df = st.session_state["model_df"]

    if models is None:
        st.warning("먼저 엔진을 준비하세요.")
        st.stop()

    # 1분봉 데이터 로드
    with st.spinner("1분봉 로딩 중..."):
        intraday = fetch_1min_intraday(ticker, days=3)
        intraday = to_kst(intraday)

    show_n = st.slider("표시 캔들 수", 50, 500, 150)

    df_plot = intraday.tail(show_n)
    last_price = df_plot["Close"].iloc[-1]
    last_ts = df_plot.index[-1]

    # 엔진 최신 row 예측
    latest_row = model_df.iloc[-1]
    raw_rets = predict_latest(models, latest_row, feat_cols)

    # horizon 리스트
    horizon_list = sorted(raw_rets.keys())

    # ================================
    # 스케일링 적용
    # ================================
    def apply_scaling(h, ret):
        if scaling_mode == "NONE (기본 선형)":
            return ret

        # 힌드캐스트 결과 기반 스케일링 값 필요
        if scaling_mode == "A (통계 기반)" and st.session_state["scaling_A"]:
            scales = st.session_state["scaling_A"]
            nearest = min(scales.keys(), key=lambda x: abs(x - h))
            return ret * scales.get(nearest, 1.0)

        if scaling_mode == "B (ML 커브)" and st.session_state["scaling_B"]:
            reg = st.session_state["scaling_B"]
            scale_val = float(reg.predict(np.array([[h]]))[0])
            return ret * scale_val

        return ret

    # horizon별 예측가 계산
    pred_prices = {}
    for h in horizon_list:
        r = raw_rets[h]

        # 실시간에 스케일링 적용 여부
        if apply_scaling_live:
            r = apply_scaling(h, r)

        pred_prices[h] = last_price * (1 + r)

    # ================================
    # 차트 그리기
    # ================================
    fig = go.Figure()

    # 캔들
    fig.add_trace(
        go.Candlestick(
            x=df_plot.index,
            open=df_plot["Open"],
            high=df_plot["High"],
            low=df_plot["Low"],
            close=df_plot["Close"],
            name="1분봉"
        )
    )

    # 예측선
    colors = ["blue", "orange", "green", "purple", "red", "darkcyan"]
    for i, h in enumerate(horizon_list):
        fig.add_trace(
            go.Scatter(
                x=[df_plot.index[0], df_plot.index[-1]],
                y=[pred_prices[h], pred_prices[h]],
                mode="lines",
                line=dict(color=colors[i % len(colors)], dash="dot"),
                name=f"+{h}분"
            )
        )

    fig.update_layout(height=500, title=f"{ticker} 1분봉 실시간 예측")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 현재가 / 예측가")
    for h in horizon_list:
        st.write(f"**+{h}분:** {pred_prices[h]:.2f}")



# ============================================================
# 2️⃣ 하루 힌드캐스트 탭
# ============================================================
with tab_back:

    st.subheader("📅 하루 힌드캐스트 (과거 하루 전체 예측)")

    df_raw = st.session_state["raw_df"]
    if df_raw is None:
        st.warning("먼저 실시간 탭에서 엔진 준비를 하세요.")
        st.stop()

    offset = st.slider("며칠 전 미국장 평가?", 1, 6, 3)

    now_utc = dt.datetime.now(dt.timezone.utc)
    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))
    eval_date = now_et.date() - dt.timedelta(days=offset)
    train_end = eval_date - dt.timedelta(days=1)

    idx_et = df_raw.index.tz_convert("America/New_York")
    mask_train = idx_et.date <= train_end
    mask_eval = idx_et.date == eval_date

    train_df = df_raw[mask_train]
    eval_df = df_raw[mask_eval]

    # 다시 학습
    st.write("🔧 과거 데이터로 다시 학습 중...")
    feat = build_feature_frame(train_df)
    model_df, horizons_bt = build_targets(feat, base_horizons, custom_h)
    X, y_dict, feat_cols = get_feature_target_matrices(model_df, horizons_bt)
    models_bt, metrics_bt = train_models(X, y_dict, random_state)

    st.dataframe(metrics_bt, use_container_width=True)

    # 평가일 예측 전개
    feat_eval = build_feature_frame(eval_df).dropna()
    close_raw = eval_df["Close"].astype(float)

    idx_positions = {ts: i for i, ts in enumerate(eval_df.index)}
    results = []

    def minutes_to_steps(h):
        return max(1, int(round(h / 2)))

    for ts in feat_eval.index:
        if ts not in idx_positions:
            continue
        pos = idx_positions[ts]
        cur = float(close_raw.iloc[pos])

        row = feat_eval.loc[ts, feat_cols]

        for h in horizons_bt:
            ret_pred = float(models_bt[h].predict(row.values.reshape(1, -1))[0])

            # 힌드캐스트에서도 스케일링 적용 여부
            if apply_scaling_backtest:
                ret_pred = apply_scaling(h, ret_pred)

            pred_price = cur * (1 + ret_pred)

            target_idx = pos + minutes_to_steps(h)
            if target_idx < len(close_raw):
                actual = float(close_raw.iloc[target_idx])
            else:
                actual = None

            results.append({
                "time": ts,
                "horizon": h,
                "pred_price": pred_price,
                "actual_price": actual
            })

    res_df = pd.DataFrame(results)

    st.success("예측 완료!")

    # -----------------------------
    # 스케일링 파라미터 저장
    # -----------------------------
    if scaling_mode == "A (통계 기반)":
        st.session_state["scaling_A"] = compute_scaling_A(res_df)
        st.write("📐 스케일링 A:", st.session_state["scaling_A"])

    if scaling_mode == "B (ML 커브)":
        st.session_state["scaling_B"] = compute_scaling_B(res_df)
        st.write("📐 스케일링 B 모델 학습 완료")


    # -----------------------------
    # 시각화 (예측 vs 실제)
    # -----------------------------
    st.markdown("### 📈 예측 vs 실제")

    h_sel = st.selectbox("horizon 선택", sorted(horizons_bt))
    view = res_df[res_df["horizon"] == h_sel].dropna()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=view["time"],
        y=view["pred_price"],
        name="pred",
        line=dict(color="blue")
    ))
    fig2.add_trace(go.Scatter(
        x=view["time"],
        y=view["actual_price"],
        name="actual",
        line=dict(color="red")
    ))

    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)