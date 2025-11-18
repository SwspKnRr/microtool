import time
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier

from core_micro import (
    fetch_2min_data,
    fetch_1min_intraday,   # ← 이게 꼭 있어야 함!!!
    build_feature_frame,
    build_targets,
    get_feature_target_matrices,
    train_models,
    predict_latest,
)



# ================================
# ⚙ 공통 설정
# ================================

# 모든 탭(4번 및 5번)에서 동일하게 사용할 Horizon 구성
HORIZONS = [1, 3, 5, 10, 15, 30, 60, 120, 300]   # 분 단위
CLOSE_TAG = "close"


# ================================
# ⏱ KST 변환 관련 유틸
# ================================

def to_kst(df: pd.DataFrame):
    """yfinance 데이터(UTC or naive)를 KST(Asia/Seoul)로 변환."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df.tz_convert("Asia/Seoul")


def get_kst_session_times(use_dst: bool):
    """한국시간 기준 미국 정규장 open/close 시간 반환."""
    if use_dst:
        # DST 적용 (미국 09:30~16:00 → KST 22:30~05:00)
        return dt.time(22, 30), dt.time(5, 0)
    else:
        # DST 미적용 (미국 09:30~16:00 → KST 23:30~06:00)
        return dt.time(23, 30), dt.time(6, 0)


def minutes_to_close_kst(ts: pd.Timestamp, open_kst: dt.time, close_kst: dt.time):
    """현재(KST)로부터 미국 정규장 폐장까지 남은 분."""
    if ts.tz is None:
        ts = ts.tz_localize("Asia/Seoul")

    t = ts.time()

    # 정규장 여부 판단
    if t >= open_kst or t < close_kst:
        # 정규장 중
        if t >= open_kst:
            # 밤 시간대 → 다음날 close
            close_dt = ts.replace(hour=close_kst.hour, minute=close_kst.minute,
                                  second=0, microsecond=0) + dt.timedelta(days=1)
        else:
            # 이미 자정 넘어 새벽구간
            close_dt = ts.replace(hour=close_kst.hour, minute=close_kst.minute,
                                  second=0, microsecond=0)
        return int((close_dt - ts).total_seconds() // 60)

    # 정규장 아님
    return None


# ================================
# 📐 2분봉 피처 생성
# ================================

def build_features_2m(df_2m: pd.DataFrame):
    """2분봉 데이터로 기본 피처 생성."""
    df = df_2m.copy()

    df["ret1"] = df["Close"].pct_change()
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["vol"] = df["Volume"]
    df["trend"] = df["Close"].diff()

    df = df.dropna()
    return df


# ================================
# 🎯 Horizon 타깃 생성
# ================================

def build_targets_2m(df_feat: pd.DataFrame, horizons: list[int]):
    """
    각 horizon 분 뒤 상승 여부(0/1) 타깃 생성.
    """
    df = df_feat.copy()
    df_tg = df.copy()

    for h in horizons:
        df_tg[f"y_{h}"] = (df["Close"].shift(-h) > df["Close"]).astype(int)

    df_tg = df_tg.dropna()
    return df_tg


# ================================
# 🤖 RandomForest 학습
# ================================

def train_models_2m(df_tg: pd.DataFrame, horizons: list[int], random_state=42):
    """
    각 horizon 분 뒤 상승확률을 예측하는 RandomForest 모델 세트 학습.
    """
    features = ["ret1", "ma5", "ma20", "vol", "trend"]

    X = df_tg[features]
    models = {}
    metrics = []

    for h in horizons:
        y = df_tg[f"y_{h}"]
        # Regular RandomForest
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            random_state=random_state
        )
        clf.fit(X, y)
        models[h] = clf

    return models, features


# ================================
# ⭐ 핵심 엔진 ⭐  (4번·5번탭 공통 사용)
# 예측 로직 100% 동일
# ================================

def engine_predict(
    df_1m: pd.DataFrame,       # 1분봉 (KST)
    df_2m: pd.DataFrame,       # 2분봉 (KST)
    models: dict,              # horizon→model
    feature_cols: list[str],   # ["ret1","ma5","ma20","vol","trend"]
    horizons: list[int],       # [1,3,5, ... 300]
    trend_window=40,
    use_dst=True
):
    """
    4번탭(실시간)과 5번탭(과거 힌드캐스트)이 **동일한 엔진을 사용**하도록 설계.
    """

    results_prob = {}      # {h: p_up}
    results_price = {}     # {h: adjusted_price}
    pred_close = None

    # ======================
    # 1) 최신 2분봉에서 모델 확률 계산
    # ======================
    latest_2m = df_2m.iloc[-1]
    X_input = latest_2m[feature_cols].values.reshape(1, -1)

    for h, model in models.items():
        prob_up = model.predict_proba(X_input)[0, 1]
        results_prob[h] = prob_up

    # ======================
    # 2) 최근 1분봉을 이용한 선형추세 계산
    # ======================
    if len(df_1m) < trend_window:
        tw = len(df_1m)
    else:
        tw = trend_window

    y_arr = df_1m["Close"].tail(tw).values
    x_arr = np.arange(tw)

    if len(y_arr) >= 2:
        slope, intercept = np.polyfit(x_arr, y_arr, 1)
    else:
        slope = 0.0
        intercept = y_arr[-1]

    last_price = df_1m["Close"].iloc[-1]
    last_time = df_1m.index[-1]

    # ======================
    # 3) horizon별 예상 가격 계산
    # ======================
    def nearest_model_prob(mins):
        if len(results_prob) == 0:
            return None
        nearest_h = min(results_prob.keys(), key=lambda h: abs(h - mins))
        return results_prob.get(nearest_h, None)

    for h in horizons:
        # 단순 추세 기반
        price_trend = last_price + slope * h

        p_up = nearest_model_prob(h)

        if p_up is None:
            results_price[h] = price_trend
        else:
            # confidence 기반 가중 평균 (좀 더 공격적으로)
            base_w = 0.3
            conf = 2 * abs(p_up - 0.5)       # 0~1
            w = base_w + (1 - base_w) * conf
            w = float(np.clip(w, 0, 1))

            adj_price = (1 - w) * last_price + w * price_trend
            results_price[h] = adj_price

    # ======================
    # 4) 종가 예측
    # ======================
    open_kst, close_kst = get_kst_session_times(use_dst)
    minutes_left = minutes_to_close_kst(last_time, open_kst, close_kst)

    if minutes_left is not None and minutes_left > 0:
        price_trend_close = last_price + slope * minutes_left
        p_up_close = nearest_model_prob(minutes_left)

        if p_up_close is None:
            pred_close = price_trend_close
        else:
            base_w = 0.3
            conf = 2 * abs(p_up_close - 0.5)
            w = base_w + (1 - base_w) * conf
            w = float(np.clip(w, 0, 1))
            pred_close = (1 - w) * last_price + w * price_trend_close

    return {
        "prob_up": results_prob,
        "pred_price": results_price,
        "pred_close": pred_close,
        "last_price": last_price,
        "last_time": last_time,
    }

# ==========================================
# 📌 Part 2 — 실시간 시그널 (4번 탭)
# ==========================================

tab1, tab2 = st.tabs(["📡 실시간 예측", "📅 30일 힌드캐스트"])


# ================================
# 📡 TAB1 — 실시간 예측
# ================================
with tab1:

    st.header("📡 실시간 1분봉 예측 (KST 기준)")
    st.caption("2분봉 학습 모델 + 동일 엔진(engine_predict)로 실시간 예측합니다.")

    # -----------------------------
    # 1) 사이드 기능 (새로고침, 캔들 수)
    # -----------------------------
    colA, colB, colC = st.columns([1.2, 1.2, 2])

    with colA:
        auto_refresh = st.checkbox("자동 새로고침 (5초)", value=False)

    with colB:
        manual_refresh = st.button("🔄 수동 새로고침")

    with colC:
        n_candles = st.slider(
            "표시할 1분봉 캔들 수",
            min_value=50,
            max_value=500,
            value=150,
            step=10
        )

    if manual_refresh:
        st.rerun()

    # -----------------------------
    # 2) 2분봉 모델 학습 여부 확인
    # -----------------------------
    if (
        "models" not in st.session_state
        or "features" not in st.session_state
        or "df_2m" not in st.session_state
    ):
        st.warning("먼저 2분봉 모델을 학습하세요.")
        st.stop()

    models = st.session_state["models"]
    feature_cols = st.session_state["features"]
    df_2m = st.session_state["df_2m"]
    ticker = st.session_state["ticker"]
    use_dst = st.session_state["use_dst"]

    # -----------------------------
    # 3) 1분봉 데이터 가져오기
    # -----------------------------
    with st.spinner("1분봉 다운로드 중..."):
        df_1m = fetch_1min_intraday(ticker, days=3)
        if df_1m is None or df_1m.empty:
            st.error("1분봉 데이터를 가져오지 못했습니다.")
            st.stop()
        df_1m = to_kst(df_1m)

    # 필요한 만큼만 슬라이싱
    df_plot = df_1m.tail(n_candles)

    last_price = df_plot["Close"].iloc[-1]
    last_time = df_plot.index[-1]


    # -----------------------------
    # 4) 예측 엔진 호출 (핵심)
    # -----------------------------
    engine_out = engine_predict(
        df_1m=df_1m,
        df_2m=df_2m,
        models=models,
        feature_cols=feature_cols,
        horizons=HORIZONS,
        trend_window=40,
        use_dst=use_dst
    )

    prob_up = engine_out["prob_up"]
    pred_price = engine_out["pred_price"]
    pred_close = engine_out["pred_close"]


    # -----------------------------
    # 5) 30분 전 예측 복원
    # -----------------------------
    def get_30min_back_prediction(df_1m, df_2m):
        now_ts = df_1m.index[-1]
        t_back = now_ts - dt.timedelta(minutes=30)

        # t_back 이전까지만 slice
        df1_back = df_1m[df_1m.index <= t_back]
        df2_back = df_2m[df_2m.index <= t_back]

        if len(df1_back) < 50 or len(df2_back) < 50:
            return None

        # 그 당시 엔진 호출
        back_out = engine_predict(
            df_1m=df1_back,
            df_2m=df2_back,
            models=models,
            feature_cols=feature_cols,
            horizons=[30],     # 30분만 복원
            trend_window=40,
            use_dst=use_dst
        )

        return {
            "made_at": df1_back.index[-1],
            "pred_30": list(back_out["pred_price"].values())[0],
            "actual_now": df_1m["Close"].iloc[-1]
        }


    back30 = get_30min_back_prediction(df_1m, df_2m)


    # -----------------------------
    # 6) 실시간 캔들 차트 출력
    # -----------------------------
    st.markdown("### 🕯 실시간 1분봉 캔들 차트 + 예측가")

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df_plot.index,
                open=df_plot["Open"],
                high=df_plot["High"],
                low=df_plot["Low"],
                close=df_plot["Close"],
                increasing=dict(line=dict(color="#FF8A8A"), fillcolor="#FF8A8A"),
                decreasing=dict(line=dict(color="#6EA6FF"), fillcolor="#6EA6FF"),
                name="1분봉"
            )
        ]
    )

    # annotation 위치 미리 설정해두기
    x_pos = {
        1: 0.02, 3: 0.12, 5: 0.22, 10: 0.32, 15: 0.42,
        30: 0.52, 60: 0.62, 120: 0.72, 300: 0.85
    }

    shapes = []
    annos = []

    for h, price in pred_price.items():
        if not np.isfinite(price):
            continue
        shapes.append(
            dict(
                type="line", xref="paper", x0=0, x1=1,
                yref="y", y0=price, y1=price,
                line=dict(color="purple", width=1, dash="dot")
            )
        )
        annos.append(
            dict(
                xref="paper", x=x_pos.get(h, 0.5),
                y=price,
                text=f"+{h}분",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=10, color="purple")
            )
        )

    # 종가예측
    if pred_close is not None:
        shapes.append(
            dict(
                type="line", xref="paper", x0=0, x1=1,
                yref="y", y0=pred_close, y1=pred_close,
                line=dict(color="black", width=1, dash="dash")
            )
        )
        annos.append(
            dict(
                xref="paper",
                x=0.5, y=pred_close,
                text="종가예측",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=10, color="black")
            )
        )

    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, t=40, b=40),
        xaxis_rangeslider_visible=False,
        shapes=shapes,
        annotations=annos,
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        modebar_remove=["zoom", "pan", "select", "lasso2d"]
    )

    st.plotly_chart(fig, use_container_width=True)


    # -----------------------------
    # 7) 사이드 정보 패널
    # -----------------------------
    colX, colY = st.columns([1.2, 1])

    with colX:
        st.subheader("💰 현재가")
        st.metric(label="Price", value=f"{last_price:,.2f}")

        st.subheader("📈 Horizon별 예상가")
        for h in HORIZONS:
            v = pred_price.get(h)
            if v is not None:
                st.metric(label=f"+{h}분", value=f"{v:,.2f}")

        if pred_close is not None:
            st.metric(label="종가예측", value=f"{pred_close:,.2f}")

    with colY:
        st.subheader("⏳ 30분 전 예측 복구")

        if back30 is None:
            st.info("데이터 부족으로 30분 전 예측을 계산할 수 없습니다.")
        else:
            st.write(f"예측 시점: {back30['made_at'].strftime('%H:%M')}")
            st.write(f"그때의 30분 뒤 예상가: **{back30['pred_30']:.2f}**")
            st.write(f"현재 실제가: **{back30['actual_now']:.2f}**")

    # -----------------------------
    # 8) 자동 새로고침
    # -----------------------------
    if auto_refresh:
        time.sleep(5)
        st.rerun()

# ==========================================
# 📌 Part 3 — 30일 힌드캐스트 (탭2)
# ==========================================

with tab2:

    st.header("📅 최근 30개 거래일 힌드캐스트 (4번탭 엔진과 완벽히 동일)")

    st.markdown("""
    이 탭은 **4번 탭에서 사용하는 정확히 동일한 예측 엔진**을  
    **과거 30개 거래일에 적용해 실제와 얼마나 일치했는지** 평가합니다.  
    """)

    # --------------------------------------------
    # 1) 1분봉 다운로드 (최근 60일)
    # --------------------------------------------
    st.subheader("📥 최근 60일 1분봉 다운로드")

    with st.spinner("다운로드 중..."):
        df_1m_all = fetch_1min_intraday(st.session_state["ticker"], days=60)

    if df_1m_all is None or df_1m_all.empty:
        st.error("1분봉 데이터를 다운로드할 수 없습니다.")
        st.stop()

    df_1m_all = to_kst(df_1m_all)
    df_1m_all = df_1m_all.sort_index()

    # --------------------------------------------
    # 2) 최근 30개 거래일 확보
    # --------------------------------------------
    df_1m_all["date"] = df_1m_all.index.date
    unique_days = sorted(df_1m_all["date"].unique(), reverse=True)

    trading_days = unique_days[:30]
    trading_days = sorted(trading_days)   # 오래된 → 최근 순서

    st.write(f"📆 확보된 거래일 수: **{len(trading_days)}일**")

    # 사용자 선택
    target_date = st.selectbox("테스트할 날짜 선택:", trading_days)

    st.markdown(f"### 🔎 선택된 날짜: **{target_date}**")

    # --------------------------------------------
    # 3) 해당 날짜의 1분봉 / 이전까지의 2분봉 생성
    # --------------------------------------------
    day_df = df_1m_all[df_1m_all["date"] == target_date]
    if len(day_df) < 100:
        st.error("해당 날짜의 데이터 부족")
        st.stop()

    # 전일까지 슬라이스
    prev_df = df_1m_all[df_1m_all["date"] < target_date]
    if len(prev_df) < 300:
        st.error("전일까지의 데이터가 부족합니다.")
        st.stop()

    # 2분봉 resample
    df_2m_prev = prev_df.resample("2T").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

    df_2m_prev = df_2m_prev.dropna()

    # --------------------------------------------
    # 4) 2분봉 모델 학습
    # --------------------------------------------
    st.subheader("🤖 2분봉 모델 학습 (전일 데이터 기반)")

    df_feat = build_features_2m(df_2m_prev)
    df_tg = build_targets_2m(df_feat, HORIZONS)

    models, feature_cols = train_models_2m(df_tg, HORIZONS, random_state=42)

    st.success("모델 학습 완료!")


    # --------------------------------------------
    # 5) 종일 예측 시뮬레이션
    # --------------------------------------------
    st.subheader(f"🔮 {target_date} 하루 전체 예측 시뮬레이션")

    results = []

    # 드문드문 문제 생기는 걸 방지하기 위해 KST 전체 사용
    full_df_2m_for_day = df_1m_all.resample("2T").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()
    full_df_2m_for_day = to_kst(full_df_2m_for_day)

    # 종일 loop
    for i in range(40, len(day_df)):

        cur_slice_1m = day_df.iloc[:i]
        current_ts = cur_slice_1m.index[-1]
        cur_price = cur_slice_1m["Close"].iloc[-1]

        # 2분봉은 전체 2분봉 중 "현재시각 이전"까지만 사용
        cur_slice_2m = full_df_2m_for_day[full_df_2m_for_day.index < current_ts]

        if len(cur_slice_2m) < 50:
            continue

        out = engine_predict(
            df_1m=cur_slice_1m,
            df_2m=cur_slice_2m,
            models=models,
            feature_cols=feature_cols,
            horizons=HORIZONS,
            trend_window=40,
            use_dst=st.session_state["use_dst"]
        )

        for h in HORIZONS:
            eval_ts = current_ts + dt.timedelta(minutes=h)
            if eval_ts not in day_df.index:
                continue
            actual_price = day_df.loc[eval_ts, "Close"]

            results.append({
                "time": current_ts,
                "horizon": h,
                "pred_price": out["pred_price"][h],
                "actual_price": actual_price,
                "current_price": cur_price
            })

    if len(results) == 0:
        st.error("예측 결과가 없습니다.")
        st.stop()

    res_df = pd.DataFrame(results)

    st.success(f"{len(res_df)}개 예측 샘플 생성됨!")


    # --------------------------------------------
    # 6) Horizon별 성능 집계
    # --------------------------------------------
    st.subheader("📊 Horizon별 성능 요약")

    perf_rows = []

    for h in HORIZONS:
        sub = res_df[res_df["horizon"] == h]

        if len(sub) == 0:
            continue

        pred = sub["pred_price"].values
        act = sub["actual_price"].values
        base = sub["current_price"].values

        # 방향 정확도
        acc = ( (pred > base) == (act > base) ).mean()

        # MAE / MAPE / RMSE
        mae = np.mean(np.abs(pred - act))
        rmse = np.sqrt(np.mean((pred - act) ** 2))
        mape = np.mean(np.abs((pred - act) / act))

        perf_rows.append({
            "horizon": h,
            "samples": len(sub),
            "accuracy": acc,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
        })

    perf_df = pd.DataFrame(perf_rows)
    st.dataframe(perf_df.style.format({
        "accuracy": "{:.3f}",
        "MAE": "{:.3f}",
        "RMSE": "{:.3f}",
        "MAPE": "{:.3%}",
    }))


    # --------------------------------------------
    # 7) Horizon별 예측 vs 실제 차트
    # --------------------------------------------
    st.subheader("📉 Horizon별 상세 차트")

    chosen_h = st.selectbox("어떤 Horizon을 볼까요?", HORIZONS)

    view = res_df[res_df["horizon"] == chosen_h]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=view["time"], y=view["actual_price"],
        name="실제", line=dict(color="red")
    ))

    fig.add_trace(go.Scatter(
        x=view["time"], y=view["pred_price"],
        name="예측", line=dict(color="blue", dash="dot")
    ))

    st.plotly_chart(fig, use_container_width=True)


    # --------------------------------------------
    # 8) 간단 해석 추가
    # --------------------------------------------
    st.subheader("🧠 엔진 해석")

    sub = res_df[res_df["horizon"] == chosen_h]

    avg_err = np.mean(sub["pred_price"] - sub["actual_price"])
    bias = "상승쪽으로 쏠림" if avg_err > 0 else "하락쪽으로 쏠림"

    st.write(f"**평균 예측 오차:** {avg_err:.2f} → **{bias} 경향**")

    st.write("""
    - 예측선(파란색)이 실제(빨간색)보다 위에 많다면 상승 bias  
    - 아래에 많다면 하락 bias  
    - RMSE/MAE가 작을수록 더 정확  
    - Accuracy는 '방향 맞춘 비율'  
    """)

