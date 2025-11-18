# app_micro.py

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
    fetch_1min_intraday,
)


# ======================================
# ⚙ 전역 설정
# ======================================

HORIZONS = [1, 3, 5, 10, 15, 30, 60, 120, 300]   # 분 단위

st.set_page_config(page_title="단타로 과자 먹자", layout="wide")
st.title("⚡ 단타로 과자 먹자")
st.caption("2분봉으로 학습하고, 1분봉에서 실시간으로 예측 + 과거 힌드캐스트까지 보는 툴")


# ======================================
# ⏱ KST / 세션 관련 유틸
# ======================================

def to_kst(df: pd.DataFrame):
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    return df.tz_convert("Asia/Seoul")


def get_kst_session_times(use_dst: bool):
    if use_dst:
        return dt.time(22, 30), dt.time(5, 0)
    else:
        return dt.time(23, 30), dt.time(6, 0)


def minutes_to_close_kst(ts: pd.Timestamp, open_kst: dt.time, close_kst: dt.time):
    if ts.tz is None:
        ts = ts.tz_localize("Asia/Seoul")
    t = ts.time()
    if t >= open_kst or t < close_kst:
        if t >= open_kst:
            close_dt = ts.replace(hour=close_kst.hour, minute=close_kst.minute,
                                  second=0, microsecond=0) + dt.timedelta(days=1)
        else:
            close_dt = ts.replace(hour=close_kst.hour, minute=close_kst.minute,
                                  second=0, microsecond=0)
        return int((close_dt - ts).total_seconds() // 60)
    return None


# ======================================
# 📐 2분봉 피처 / 타깃 / 학습
# ======================================

def build_features_2m(df_2m: pd.DataFrame):
    df = df_2m.copy()
    df["ret1"] = df["Close"].pct_change()
    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["vol"] = df["Volume"]
    df["trend"] = df["Close"].diff()
    return df.dropna()


def build_targets_2m(df_feat: pd.DataFrame, horizons: list[int]):
    df = df_feat.copy()
    for h in horizons:
        df[f"y_{h}"] = (df["Close"].shift(-h) > df["Close"]).astype(int)
    return df.dropna()


def train_models_2m(df_tg: pd.DataFrame, horizons: list[int], random_state=42):
    features = ["ret1", "ma5", "ma20", "vol", "trend"]
    X = df_tg[features]
    models = {}
    for h in horizons:
        y = df_tg[f"y_{h}"]
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            random_state=random_state,
        )
        clf.fit(X, y)
        models[h] = clf
    return models, features


# ======================================
# ⭐ 공통 엔진: 4/5번 탭 모두 이거 사용
# ======================================

def engine_predict(
    df_1m: pd.DataFrame,
    df_2m: pd.DataFrame,
    models: dict,
    feature_cols: list[str],
    horizons: list[int],
    trend_window: int,
    use_dst: bool,
):
    prob_up = {}
    pred_price = {}
    pred_close = None

    # --- 최신 2분봉에서 모델 확률 ---
    latest_2m = df_2m.iloc[-1]
    X_input = latest_2m[feature_cols].values.reshape(1, -1)
    for h, model in models.items():
        p = model.predict_proba(X_input)[0, 1]
        prob_up[h] = p

    # --- 최근 1분봉 추세 ---
    if len(df_1m) < 2:
        return {
            "prob_up": prob_up,
            "pred_price": pred_price,
            "pred_close": None,
            "last_price": df_1m["Close"].iloc[-1] if len(df_1m) else None,
            "last_time": df_1m.index[-1] if len(df_1m) else None,
        }

    tw = min(trend_window, len(df_1m))
    y_arr = df_1m["Close"].tail(tw).values
    x_arr = np.arange(tw)
    slope, intercept = np.polyfit(x_arr, y_arr, 1)

    last_price = df_1m["Close"].iloc[-1]
    last_time = df_1m.index[-1]

    def nearest_model_prob(mins: int):
        if not prob_up:
            return None
        nearest_h = min(prob_up.keys(), key=lambda hh: abs(hh - mins))
        return prob_up.get(nearest_h, None)

    for h in horizons:
        trend_price = last_price + slope * h
        p_up = nearest_model_prob(h)
        if p_up is None:
            pred_price[h] = trend_price
        else:
            base_w = 0.3
            conf = 2 * abs(p_up - 0.5)
            w = base_w + (1 - base_w) * conf
            w = float(np.clip(w, 0, 1))
            pred_price[h] = (1 - w) * last_price + w * trend_price

    # --- 종가 예측 ---
    open_kst, close_kst = get_kst_session_times(use_dst)
    left_min = minutes_to_close_kst(last_time, open_kst, close_kst)
    if left_min is not None and left_min > 0:
        trend_close = last_price + slope * left_min
        p_up_close = nearest_model_prob(left_min)
        if p_up_close is None:
            pred_close = trend_close
        else:
            base_w = 0.3
            conf = 2 * abs(p_up_close - 0.5)
            w = base_w + (1 - base_w) * conf
            w = float(np.clip(w, 0, 1))
            pred_close = (1 - w) * last_price + w * trend_close

    return {
        "prob_up": prob_up,
        "pred_price": pred_price,
        "pred_close": pred_close,
        "last_price": last_price,
        "last_time": last_time,
    }


# ======================================
# 🧱 세션 기본값
# ======================================

if "ticker" not in st.session_state:
    st.session_state["ticker"] = "QQQ"
if "use_dst" not in st.session_state:
    st.session_state["use_dst"] = True
if "trend_window" not in st.session_state:
    st.session_state["trend_window"] = 100
if "train_days" not in st.session_state:
    st.session_state["train_days"] = 40
if "random_state" not in st.session_state:
    st.session_state["random_state"] = 42
if "models" not in st.session_state:
    st.session_state["models"] = None
if "features" not in st.session_state:
    st.session_state["features"] = None
if "df_2m" not in st.session_state:
    st.session_state["df_2m"] = None
if "engine_config" not in st.session_state:
    st.session_state["engine_config"] = None
if "need_retrain" not in st.session_state:
    st.session_state["need_retrain"] = False


# ======================================
# 🎛 사이드바: 엔진 설정 + 재학습 버튼
# ======================================

with st.sidebar:
    st.header("⚙ 엔진 설정")

    ticker = st.text_input("티커", value=st.session_state["ticker"])
    st.session_state["ticker"] = ticker

    train_days = st.slider("2분봉 학습 기간(일)", 10, 60, st.session_state["train_days"], 1)
    st.session_state["train_days"] = train_days

    trend_window = st.slider("추세 window (1분봉 캔들 수)", 20, 200, st.session_state["trend_window"], 10)
    st.session_state["trend_window"] = trend_window

    random_state = st.number_input("Random Seed", 0, 9999, st.session_state["random_state"], 1)
    st.session_state["random_state"] = int(random_state)

    use_dst = st.checkbox("미국 써머타임(DST) 적용", value=st.session_state["use_dst"])
    st.session_state["use_dst"] = use_dst

    st.markdown("---")
    if st.button("🧠 엔진 다시 학습하기"):
        st.session_state["need_retrain"] = True

    st.markdown("---")
    st.caption("※ 엔진은 2분봉으로 학습되고, 1분봉에서 예측에 사용됩니다.")


# ======================================
# 🔁 엔진 재학습 (버튼 눌렀을 때만)
# ======================================

if st.session_state["need_retrain"]:
    st.info("엔진 학습 중...")

    df_2m = fetch_2min_data(ticker, days=train_days)
    if df_2m is None or df_2m.empty:
        st.error("2분봉 데이터를 가져올 수 없습니다.")
    else:
        df_2m = to_kst(df_2m)
        feat_2m = build_features_2m(df_2m)
        df_tg = build_targets_2m(feat_2m, HORIZONS)
        models, feature_cols = train_models_2m(df_tg, HORIZONS, random_state=random_state)

        st.session_state["df_2m"] = df_2m
        st.session_state["models"] = models
        st.session_state["features"] = feature_cols
        st.session_state["engine_config"] = {
            "ticker": ticker,
            "train_days": train_days,
            "trend_window": trend_window,
            "random_state": random_state,
            "use_dst": use_dst,
        }
        st.success("✅ 엔진 학습 완료!")

    st.session_state["need_retrain"] = False


# ======================================
# 📑 탭 구성: 1) 실시간, 2) 힌드캐스트
# ======================================

tab1, tab2 = st.tabs(["📡 실시간 예측", "📅 하루 힌드캐스트"])


# ======================================
# 📡 TAB1 — 실시간 예측
# ======================================

with tab1:
    st.subheader("📡 실시간 1분봉 예측 (KST)")

    if st.session_state["models"] is None or st.session_state["df_2m"] is None:
        st.warning("먼저 사이드바에서 '엔진 다시 학습하기'를 눌러주세요.")
    else:
        models = st.session_state["models"]
        feature_cols = st.session_state["features"]
        df_2m = st.session_state["df_2m"]
        use_dst = st.session_state["use_dst"]
        trend_window = st.session_state["trend_window"]
        ticker = st.session_state["ticker"]

        colA, colB, colC = st.columns([1.2, 1.2, 2])
        with colA:
            auto_refresh = st.checkbox("자동 새로고침 (5초)", value=False)
        with colB:
            manual_refresh = st.button("🔄 수동 새로고침")
        with colC:
            n_candles = st.slider("표시할 1분봉 캔들 수", 50, 500, 150, 10)

        if manual_refresh:
            st.rerun()

        with st.spinner("1분봉 불러오는 중..."):
            df_1m = fetch_1min_intraday(ticker, days=3)
        if df_1m is None or df_1m.empty:
            st.error("1분봉 데이터를 가져올 수 없습니다.")
        else:
            df_1m = to_kst(df_1m)
            df_plot = df_1m.tail(n_candles)

            engine_out = engine_predict(
                df_1m=df_1m,
                df_2m=df_2m,
                models=models,
                feature_cols=feature_cols,
                horizons=HORIZONS,
                trend_window=trend_window,
                use_dst=use_dst,
            )

            prob_up = engine_out["prob_up"]
            pred_price = engine_out["pred_price"]
            pred_close = engine_out["pred_close"]
            last_price = engine_out["last_price"]
            last_time = engine_out["last_time"]

            # --- 30분 전 예측 복원 ---
            def get_back_30min():
                t_now = df_1m.index[-1]
                t_back = t_now - dt.timedelta(minutes=30)
                df1_back = df_1m[df_1m.index <= t_back]
                df2_back = df_2m[df_2m.index <= t_back]
                if len(df1_back) < 50 or len(df2_back) < 50:
                    return None
                back_out = engine_predict(
                    df_1m=df1_back,
                    df_2m=df2_back,
                    models=models,
                    feature_cols=feature_cols,
                    horizons=[30],
                    trend_window=trend_window,
                    use_dst=use_dst,
                )
                return {
                    "time": df1_back.index[-1],
                    "pred_30": list(back_out["pred_price"].values())[0],
                    "actual_now": last_price,
                }

            back30 = get_back_30min()

            # --- 차트 ---
            st.markdown("### 🕯 실시간 1분봉 캔들 + Horizon별 예상가")

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
                        name="1분봉",
                    )
                ]
            )

            x_pos = {1: 0.02, 3: 0.12, 5: 0.22, 10: 0.32, 15: 0.42,
                     30: 0.52, 60: 0.62, 120: 0.72, 300: 0.85}

            shapes = []
            annos = []
            for h, price in pred_price.items():
                if not np.isfinite(price):
                    continue
                shapes.append(
                    dict(
                        type="line", xref="paper", x0=0, x1=1,
                        yref="y", y0=price, y1=price,
                        line=dict(color="purple", width=1, dash="dot"),
                    )
                )
                annos.append(
                    dict(
                        xref="paper",
                        x=x_pos.get(h, 0.5),
                        y=price,
                        text=f"+{h}분",
                        yanchor="bottom",
                        showarrow=False,
                        font=dict(size=10, color="purple"),
                    )
                )

            if pred_close is not None and np.isfinite(pred_close):
                shapes.append(
                    dict(
                        type="line", xref="paper", x0=0, x1=1,
                        yref="y", y0=pred_close, y1=pred_close,
                        line=dict(color="black", width=1, dash="dash"),
                    )
                )
                annos.append(
                    dict(
                        xref="paper",
                        x=0.5,
                        y=pred_close,
                        text="종가예측",
                        yanchor="bottom",
                        showarrow=False,
                        font=dict(size=10, color="black"),
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
                modebar_remove=["zoom", "pan", "select", "lasso2d"],
            )

            st.plotly_chart(fig, use_container_width=True)

            colX, colY = st.columns([1.3, 1])
            with colX:
                st.subheader("💰 현재가 & 예상가")
                st.metric("현재가", f"{last_price:,.2f}")
                for h in HORIZONS:
                    if h in pred_price:
                        st.metric(f"+{h}분 예상", f"{pred_price[h]:,.2f}")
                if pred_close is not None:
                    st.metric("종가 예상", f"{pred_close:,.2f}")

            with colY:
                st.subheader("⏪ 30분 전 예측 복원")
                if back30 is None:
                    st.info("데이터 부족으로 30분 전 예측을 복원하지 못했습니다.")
                else:
                    st.write(f"예측 시점: {back30['time'].strftime('%H:%M')}")
                    st.write(f"그때의 +30분 예상가: {back30['pred_30']:.2f}")
                    st.write(f"현재 실제가: {back30['actual_now']:.2f}")

            st.caption(f"마지막 캔들 시각 (KST): {last_time}")

            if auto_refresh:
                time.sleep(5)
                st.rerun()


# ======================================
# 📅 TAB2 — 하루 힌드캐스트
# ======================================

with tab2:
    st.subheader("📅 하루 힌드캐스트 (과거에 4번 탭 엔진을 썼다면?)")

    if st.session_state["models"] is None or st.session_state["df_2m"] is None:
        st.warning("먼저 사이드바에서 엔진을 학습하세요.")
    else:
        ticker = st.session_state["ticker"]
        models = st.session_state["models"]
        feature_cols = st.session_state["features"]
        use_dst = st.session_state["use_dst"]
        trend_window = st.session_state["trend_window"]

        with st.spinner("최근 60일 1분봉 다운로드 중..."):
            df_1m_all = fetch_1min_intraday(ticker, days=7)  # yfinance 한계 때문에 7일
        if df_1m_all is None or df_1m_all.empty:
            st.error("1분봉 데이터를 가져올 수 없습니다.")
        else:
            df_1m_all = to_kst(df_1m_all)
            df_1m_all["date"] = df_1m_all.index.date
            days = sorted(df_1m_all["date"].unique())
            if len(days) == 0:
                st.error("거래일이 없습니다.")
            else:
                target_date = st.selectbox("어느 날짜를 테스트할까요?", days, index=len(days) - 1)
                st.write(f"선택한 날짜: **{target_date}**")

                day_df = df_1m_all[df_1m_all["date"] == target_date]
                prev_df = df_1m_all[df_1m_all["date"] < target_date]

                if len(day_df) < 100 or len(prev_df) < 200:
                    st.warning("이 날짜로 힌드캐스트하기에 데이터가 부족합니다.")
                else:
                    run_backtest = st.button("🔁 이 날짜로 힌드캐스트 실행")

                    if run_backtest:
                        # 2분봉 전체 생성
                        df_2m_all = df_1m_all.resample("2T").agg({
                            "Open": "first",
                            "High": "max",
                            "Low": "min",
                            "Close": "last",
                            "Volume": "sum",
                        }).dropna()
                        df_2m_all = to_kst(df_2m_all)

                        results = []

                        for i in range(40, len(day_df)):
                            cur_1m = day_df.iloc[:i]
                            cur_ts = cur_1m.index[-1]
                            cur_price = cur_1m["Close"].iloc[-1]

                            cur_2m = df_2m_all[df_2m_all.index < cur_ts]
                            if len(cur_2m) < 50:
                                continue

                            out = engine_predict(
                                df_1m=cur_1m,
                                df_2m=cur_2m,
                                models=models,
                                feature_cols=feature_cols,
                                horizons=HORIZONS,
                                trend_window=trend_window,
                                use_dst=use_dst,
                            )

                            for h in HORIZONS:
                                eval_ts = cur_ts + dt.timedelta(minutes=h)
                                # 가장 가까운 시각으로 매칭
                                idx = day_df.index.get_indexer([eval_ts], method="nearest")
                                if idx[0] == -1:
                                    continue
                                act_price = day_df["Close"].iloc[idx[0]]
                                results.append({
                                    "time": cur_ts,
                                    "horizon": h,
                                    "pred_price": out["pred_price"][h],
                                    "actual_price": act_price,
                                    "current_price": cur_price,
                                })

                        if not results:
                            st.error("예측 결과가 없습니다.")
                        else:
                            res_df = pd.DataFrame(results)
                            st.write(f"총 예측 샘플 수: {len(res_df)}")

                            perf_rows = []
                            for h in HORIZONS:
                                sub = res_df[res_df["horizon"] == h]
                                if len(sub) == 0:
                                    continue
                                pred = sub["pred_price"].values
                                act = sub["actual_price"].values
                                base = sub["current_price"].values

                                acc = ((pred > base) == (act > base)).mean()
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
                            st.subheader("📊 Horizon별 성능 요약")
                            st.dataframe(
                                perf_df.style.format({
                                    "accuracy": "{:.3f}",
                                    "MAE": "{:.3f}",
                                    "RMSE": "{:.3f}",
                                    "MAPE": "{:.3%}",
                                }),
                                use_container_width=True,
                            )

                            st.subheader("📉 예측 vs 실제 차트")
                            h_sel = st.selectbox("어떤 Horizon을 볼까?", HORIZONS)
                            view = res_df[res_df["horizon"] == h_sel]

                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(
                                x=view["time"],
                                y=view["actual_price"],
                                name="실제",
                                line=dict(color="red"),
                            ))
                            fig2.add_trace(go.Scatter(
                                x=view["time"],
                                y=view["pred_price"],
                                name="예측",
                                line=dict(color="blue", dash="dot"),
                            ))
                            st.plotly_chart(fig2, use_container_width=True)

                            avg_err = np.mean(view["pred_price"] - view["actual_price"])
                            bias = "상승 쪽으로 조금 과하게 보는 경향" if avg_err > 0 else "하락 쪽으로 조금 보수적인 경향"
                            st.subheader("🧠 간단 해석")
                            st.write(f"- 평균 예측 오차: {avg_err:.2f} → {bias}")
                            st.write("- Accuracy는 '방향 맞춘 비율', MAE/MAPE/RMSE는 가격 오차 크기를 나타냄.")
                    else:
                        st.info("버튼을 눌러 이 날짜에 대해 힌드캐스트를 실행하세요.")
