# app_micro.py

import time
import datetime as dt

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from zoneinfo import ZoneInfo

from core_micro import (
    fetch_2min_data,
    fetch_1min_intraday,
    build_feature_frame,
    build_targets,
    get_feature_target_matrices,
    train_models,
    predict_latest,
)

# ---------- 한글 폰트 설정 (Windows 기준: 굴림) ---------- #
matplotlib.rcParams["font.family"] = "Gulim"
matplotlib.rcParams["axes.unicode_minus"] = False


# ---------- 공통 유틸 함수들 (타임존 & 세션 처리) ---------- #
def to_kst(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance에서 받은 DataFrame의 인덱스를 무조건 KST(Asia/Seoul)로 변환.
    - 인덱스가 naive면 UTC로 가정 후 KST로 변환
    - tz-aware면 그대로 KST로 tz_convert
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    if df.index.tz is None:
        df = df.tz_localize("UTC")
    df = df.tz_convert("Asia/Seoul")
    return df


def get_kst_session_times(use_dst: bool) -> tuple[dt.time, dt.time]:
    """
    미국 정규장 개장/폐장 시각을 '한국시간(KST)' 기준으로 반환.
    use_dst=True  → 써머타임 적용 기준
    use_dst=False → 써머타임 미적용 기준
    """
    if use_dst:
        # US/Eastern 09:30~16:00 → KST 22:30 ~ 다음날 05:00
        open_kst = dt.time(22, 30)
        close_kst = dt.time(5, 0)
    else:
        # US/Eastern 09:30~16:00 → KST 23:30 ~ 다음날 06:00
        open_kst = dt.time(23, 30)
        close_kst = dt.time(6, 0)
    return open_kst, close_kst


def is_regular_session_kst(ts: pd.Timestamp, open_kst: dt.time, close_kst: dt.time) -> bool:
    """
    KST 기준 시각(ts)이 미국 '정규장' 시간대인지 여부.
    - 장이 밤에 열려서 새벽에 닫히므로, open_kst ~ 24:00, 00:00 ~ close_kst 두 구간을 하나로 본다.
    """
    t = ts.time()
    if t >= open_kst or t < close_kst:
        return True
    return False


def minutes_to_close_kst(ts: pd.Timestamp, open_kst: dt.time, close_kst: dt.time) -> int | None:
    """
    현재 시각(ts, KST 기준)에서 미국 정규장 '폐장시각'까지 남은 분 수 계산.
    정규장이 아니면 None 반환.
    """
    if ts.tz is None:
        ts = ts.tz_localize("Asia/Seoul")

    t = ts.time()
    if not is_regular_session_kst(ts, open_kst, close_kst):
        return None

    if t >= open_kst:
        close_dt = ts.replace(
            hour=close_kst.hour,
            minute=close_kst.minute,
            second=0,
            microsecond=0,
        ) + dt.timedelta(days=1)
    else:
        close_dt = ts.replace(
            hour=close_kst.hour,
            minute=close_kst.minute,
            second=0,
            microsecond=0,
        )

    delta_min = int((close_dt - ts).total_seconds() // 60)
    return max(delta_min, 0)


def get_session_label_kst(ts: pd.Timestamp, open_kst: dt.time, close_kst: dt.time) -> str:
    """
    한국시간(ts) 기준으로 프리장/정규장/애프터장 라벨 반환.
    """
    t = ts.time()
    if is_regular_session_kst(ts, open_kst, close_kst):
        return "정규장(Regular)"
    if t < open_kst:
        return "프리장(Pre-market)"
    return "애프터장(After-hours)"


def get_session_mask_kst(times: pd.Series, open_kst: dt.time, close_kst: dt.time) -> list[str]:
    """
    시계열 인덱스(KST 기준)에 대해 각 시점의 세션 라벨 목록을 반환.
    """
    labels: list[str] = []
    for ts in times:
        t = ts.time()
        if t >= open_kst or t < close_kst:
            labels.append("regular")
        elif t < open_kst:
            labels.append("premarket")
        else:
            labels.append("after")
    return labels


# ---------- 공통 엔진: 2분봉 → 피처/타깃 → 모델 학습 ---------- #
def run_training_pipeline(
    df_raw: pd.DataFrame,
    base_horizons: list[int],
    custom_h: int | None,
    random_state: int,
):
    """
    공통 엔진:
    - build_feature_frame
    - build_targets
    - get_feature_target_matrices
    - train_models
    """
    feat_df = build_feature_frame(df_raw)
    model_df, horizons = build_targets(
        feat_df,
        base_horizons=base_horizons,
        custom_horizon=int(custom_h) if custom_h else None,
    )
    X, y_dict, feature_cols = get_feature_target_matrices(model_df, horizons)
    models, metrics_df = train_models(X, y_dict, random_state=random_state)

    return {
        "feat_df": feat_df,
        "model_df": model_df,
        "horizons": horizons,
        "X": X,
        "y_dict": y_dict,
        "feature_cols": feature_cols,
        "models": models,
        "metrics": metrics_df,
    }


# ---------- 페이지 기본 설정 ---------- #
st.set_page_config(
    page_title="단타 예측 웹앱 (2분봉 엔진 + 1분봉 실시간 / 하루 힌드캐스트)",
    layout="wide",
)

st.title("⚡ 단타로 과자 먹자")
st.caption("2분봉 엔진 하나로 실시간 1분봉 예측 + 하루 힌드캐스트까지 한 번에")


# ---------- 세션 상태 초기화 ---------- #
def init_state():
    defaults = {
        "raw_df": None,          # 2분봉 데이터 (KST, tz-aware)
        "feat_df": None,
        "model_df": None,
        "horizons": None,
        "X": None,
        "y_dict": None,
        "feature_cols": None,
        "models": None,
        "metrics": None,
        "pred_log": None,          # 예측 로그 (DataFrame)
        "last_logged_time": None,  # 마지막으로 로그 찍은 1분봉 시각 (KST)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ---------- 사이드바 설정 ---------- #
with st.sidebar:
    st.header("⚙ 공통 설정 (엔진)")

    ticker = st.text_input("티커 (예: QQQ, SPY, AAPL 등)", value="QQQ")

    days = st.slider("최근 N일 (1~60일, 2분봉 학습용)", min_value=1, max_value=60, value=40, step=1)

    st.markdown("---")
    st.subheader("⏱ 예측 타임프레임 (2분봉 기반)")

    base_horizons = st.multiselect(
        "기본 예측 분 단위 (여러 개 선택 가능)",
        options=[5, 10, 30],
        default=[5, 10, 30],
    )

    custom_h = st.number_input(
        "사용자 정의 X분 (1~60분, 선택 사항)",
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
    use_dst = st.checkbox("미국 써머타임(DST) 적용", value=True)
    st.caption(
        "- ON: 미국 써머타임 기간 기준 (정규장 KST 22:30~05:00)\n"
        "- OFF: 써머타임 미적용 기준 (정규장 KST 23:30~06:00)"
    )

    st.markdown("---")
    st.caption("① 2분봉 데이터 다운로드 → ② 피처/타깃 생성 → ③ 모델 학습 → ④ 실시간 1분봉 시그널 → ⑤ 하루 힌드캐스트 평가")


# ---------- 탭 구성: 실시간 / 힌드캐스트 ---------- #
tab_live, tab_backtest = st.tabs(
    [
        "1️⃣ 실시간 시그널 (1분봉)",
        "2️⃣ 하루 힌드캐스트 테스트",
    ]
)


# ==================== 1) 실시간 시그널 탭 ==================== #
with tab_live:
    st.subheader("🚀 원클릭 파이프라인 + 실시간 시그널 (1분봉 / KST)")

    # ---- 1-1. 원클릭 파이프라인 실행 버튼 ---- #
    with st.expander("엔진 준비 (2분봉 다운로드 + 피처/타깃 + 모델 학습)", expanded=True):
        if st.button("🚀 2분봉 다운로드 + 피처/타깃 생성 + 모델 학습 (원클릭)"):
            with st.spinner("2분봉 다운로드 및 모델 학습 중..."):
                try:
                    df_raw = fetch_2min_data(ticker, days=days)
                    if df_raw is None or df_raw.empty:
                        raise ValueError("받아온 2분봉 데이터가 비어 있습니다.")
                    df_raw = to_kst(df_raw)

                    engine_out = run_training_pipeline(
                        df_raw=df_raw,
                        base_horizons=base_horizons,
                        custom_h=int(custom_h) if custom_h else None,
                        random_state=int(random_state),
                    )
                except Exception as e:
                    st.error(f"엔진 준비 중 오류 발생: {e}")
                else:
                    st.session_state["raw_df"] = df_raw
                    for k in ["feat_df", "model_df", "horizons",
                              "X", "y_dict", "feature_cols",
                              "models", "metrics"]:
                        st.session_state[k] = engine_out[k]
                    st.success(
                        f"엔진 준비 완료! ({ticker}, 최근 {days}일 2분봉, "
                        f"예측 horizon: {engine_out['horizons']})"
                    )

        metrics_df = st.session_state["metrics"]
        if metrics_df is not None:
            st.markdown("### 📊 엔진 성능 지표 (최근 30% 구간 테스트)")
            st.dataframe(
                metrics_df.style.format(
                    {
                        "MAE": "{:.4f}",
                        "RMSE": "{:.4f}",
                        "direction_acc": "{:.3f}",
                    }
                ),
                use_container_width=True,
            )

    st.markdown("---")

    # ---- 1-2. 실시간 시그널 엔진 준비 여부 체크 ---- #
    models = st.session_state["models"]
    model_df = st.session_state["model_df"]
    feature_cols = st.session_state["feature_cols"]
    horizons_engine = st.session_state["horizons"]

    if models is None or model_df is None or feature_cols is None or horizons_engine is None:
        st.warning("먼저 위에서 🚀 원클릭 버튼으로 엔진을 한 번 학습시켜 주세요.")
    else:
        # ----- 최신 2분봉 기준 예측 결과 (테이블) ----- #
        latest_row = model_df.iloc[-1]
        ret_preds = predict_latest(models, latest_row, feature_cols)  # {h: future_ret_pred}

        st.markdown("### 🔮 현재(가장 최근 2분봉, KST) 기준 예측 수익률 / 가격")

        last_close = float(latest_row["Close"])

        rows = []
        for h in sorted(ret_preds.keys()):
            r = ret_preds[h]
            price_pred = last_close * (1.0 + r)
            rows.append(
                {
                    "horizon_min": h,
                    "ret_pred": r,
                    "price_pred": price_pred,
                }
            )
        prob_df = pd.DataFrame(rows).set_index("horizon_min")
        st.dataframe(
            prob_df.style.format(
                {
                    "ret_pred": "{:.3%}",
                    "price_pred": "{:.2f}",
                }
            ),
            use_container_width=True,
        )

        st.markdown("---")

        # ----- 1-3. 실시간 1분봉 차트 & 현재가 + horizon별 예상가 ----- #
        st.markdown("### 🕯 1분봉 실시간 캔들 차트 (KST) + 현재가 + 회귀 기반 예상 가격")

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
            show_close_chk = st.checkbox("종가", value=False)

        horizon_flags = {
            1: show_1,
            3: show_3,
            10: show_10,
            30: show_30,
            60: show_60,
            120: show_120,
            300: show_300,
        }

        if refresh_now:
            st.rerun()

        # 1분봉 데이터 불러오기
        with st.spinner("1분봉 데이터 불러오는 중... (KST 변환)"):
            try:
                intraday_df = fetch_1min_intraday(ticker, days=3)
                if intraday_df is not None and not intraday_df.empty:
                    intraday_df = to_kst(intraday_df)
            except Exception as e:
                st.error(f"1분봉 데이터 다운로드 중 오류 발생: {e}")
                intraday_df = None

        if intraday_df is not None and not intraday_df.empty:
            # ---------- 차트 범위 선택: 최근 N캔들 vs 오늘 미국장 전체 ---------- #
            view_mode = st.radio(
                "차트 범위",
                ("최근 N캔들", "오늘 미국장(프리+데이+애프터)"),
                horizontal=True,
            )

            if view_mode == "최근 N캔들":
                df_plot = intraday_df.tail(n_candles).copy()
            else:
                # 오늘(가장 최근 캔들 기준) 미국 날짜의 프리+데이+애프터만 보기
                idx_et = intraday_df.index.tz_convert("America/New_York")
                us_date = idx_et[-1].date()
                mask_us = idx_et.date == us_date
                df_plot = intraday_df[mask_us].copy()

                # 혹시라도 비면 안전하게 최근 N캔들로 fallback
                if df_plot.empty:
                    df_plot = intraday_df.tail(n_candles).copy()

            last_price = df_plot["Close"].iloc[-1]
            last_time = df_plot.index[-1]  # KST

            open_kst, close_kst = get_kst_session_times(use_dst)

            # ===== horizon별 예측 가격 (회귀 기반, 선형 스케일링) ===== #
            preds: dict[int, float] = {}  # {horizon_min: pred_price}
            model_horizons = list(ret_preds.keys())

            def get_scaled_ret_for(target_min: int) -> float | None:
                """
                엔진이 가지고 있는 horizon 중 가장 가까운 h_model의
                future_ret_pred 를 가져와서
                target_min / h_model 비율만큼 선형 스케일링.
                """
                if not model_horizons:
                    return None
                nearest_h = min(model_horizons, key=lambda H: abs(H - target_min))
                base_ret = ret_preds.get(nearest_h, None)
                if base_ret is None:
                    return None
                scale = target_min / nearest_h
                return base_ret * scale

            for h_min, flag in horizon_flags.items():
                if not flag:
                    continue
                r_scaled = get_scaled_ret_for(h_min)
                if r_scaled is None:
                    continue
                preds[h_min] = float(last_price * (1.0 + r_scaled))

            # 종가 예측 (폐장까지 남은 시간 기준)
            pred_close = None
            if show_close_chk:
                minutes_to_close = minutes_to_close_kst(last_time, open_kst, close_kst)
                if minutes_to_close is not None and minutes_to_close > 0:
                    r_scaled = get_scaled_ret_for(minutes_to_close)
                    if r_scaled is not None:
                        pred_close = float(last_price * (1.0 + r_scaled))

            # ----- 예측 로그 저장 (5분 / 10분 / 1시간 / 6시간 / 1일) ----- #
            if st.session_state["pred_log"] is None:
                st.session_state["pred_log"] = pd.DataFrame(
                    columns=["made_at", "horizon_min", "base_price", "pred_price", "eval_time"]
                )

            last_logged = st.session_state.get("last_logged_time", None)
            if (last_logged is None) or (last_time > last_logged):
                log_horizons = [5, 10, 60, 360, 1440]
                new_rows = []
                for h_log in log_horizons:
                    r_scaled = get_scaled_ret_for(h_log)
                    if r_scaled is None:
                        continue
                    pred_price_log = float(last_price * (1.0 + r_scaled))
                    eval_time = last_time + dt.timedelta(minutes=h_log)
                    new_rows.append(
                        {
                            "made_at": last_time,
                            "horizon_min": h_log,
                            "base_price": last_price,
                            "pred_price": pred_price_log,
                            "eval_time": eval_time,
                        }
                    )
                if new_rows:
                    st.session_state["pred_log"] = pd.concat(
                        [st.session_state["pred_log"], pd.DataFrame(new_rows)],
                        ignore_index=True,
                    )
                    st.session_state["last_logged_time"] = last_time

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
                                line=dict(color="#FF8A8A"),
                                fillcolor="#FF8A8A",
                            ),
                            decreasing=dict(
                                line=dict(color="#6EA6FF"),
                                fillcolor="#6EA6FF",
                            ),
                            name="1분봉",
                        )
                    ]
                )

                # --- 세션 배경 (프리장 / 데이장 / 애프터장) 음영 추가 --- #
                session_colors = {
                    "premarket": "rgba(150, 200, 255, 0.12)",  # 연파랑
                    "regular": "rgba(150, 255, 150, 0.15)",    # 연초록(데이장)
                    "after": "rgba(180, 180, 180, 0.12)",      # 연회색
                }

                times = df_plot.index
                session_mask = get_session_mask_kst(times, open_kst, close_kst)

                shaded_regions = []
                start_idx = 0
                for i in range(1, len(times)):
                    if session_mask[i] != session_mask[i - 1]:
                        shaded_regions.append((start_idx, i - 1, session_mask[i - 1]))
                        start_idx = i
                shaded_regions.append((start_idx, len(times) - 1, session_mask[-1]))

                shapes = []
                for start, end, label in shaded_regions:
                    color = session_colors.get(label)
                    if color is None:
                        continue
                    shapes.append(
                        dict(
                            type="rect",
                            xref="x",
                            x0=times[start],
                            x1=times[end],
                            yref="paper",
                            y0=0,
                            y1=1,
                            fillcolor=color,
                            line_width=0,
                            layer="below",
                        )
                    )

                annotations = []

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

                title_suffix = "최근 {n}개".format(n=n_candles) if view_mode == "최근 N캔들" else "오늘 미국장"
                fig_c.update_layout(
                    dragmode=False,
                    xaxis=dict(fixedrange=True),
                    yaxis=dict(fixedrange=True),
                    modebar_remove=[
                        "zoom",
                        "select",
                        "lasso2d",
                        "pan",
                        "resetScale2d",
                    ],
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=10, r=10, t=40, b=40),
                    height=450,
                    title=f"{ticker} 1분봉 캔들 ({title_suffix}, KST)",
                    shapes=shapes,
                    annotations=annotations,
                )

                st.plotly_chart(fig_c, use_container_width=True)

            with info_col:
                st.markdown("#### 💰 현재가")
                st.metric(label="Price", value=f"{last_price:,.2f}")

                st.markdown("#### 🔮 회귀 기반 예상가")
                if preds:
                    for h_min in sorted(preds.keys()):
                        price = preds[h_min]
                        st.metric(label=f"+{h_min}분", value=f"{price:,.2f}")
                else:
                    st.write("예상가: 계산 불가 (데이터 또는 모델 예측 없음)")

                if pred_close is not None and np.isfinite(pred_close):
                    st.metric(label="종가 예상", value=f"{pred_close:,.2f}")

                st.markdown("#### 🕒 시각 (KST)")
                st.write(last_time.strftime("%Y-%m-%d %H:%M:%S"))

                session_label = get_session_label_kst(last_time, open_kst, close_kst)
                st.caption(f"현재 세션 추정: {session_label} (KST 기준)")

                st.markdown("---")
                st.caption(
                    "※ 모든 시간은 한국시간(KST, UTC+9) 기준입니다.\n"
                    "※ 정규장 시간대는 DST 체크박스에 따라 KST 22:30~05:00 또는 23:30~06:00으로 간주됩니다.\n"
                    "※ 예상 가격은 2분봉 엔진이 직접 예측한 '미래 수익률(%)'을 현재가에 곱해 계산한 값입니다."
                )

            st.markdown("#### 🔎 최근 1분봉 원시 데이터 (마지막 5개, KST)")
            st.dataframe(intraday_df.tail(5))
        else:
            st.info("1분봉 데이터를 가져오지 못했습니다. 티커/시간대를 다시 확인해 주세요.")

        if auto_refresh:
            time.sleep(5)
            st.rerun()


# ==================== 2) 하루 힌드캐스트 탭 ==================== #
with tab_backtest:
    st.header("📅 하루 힌드캐스트 테스트 (같은 2분봉 엔진으로 과거 하루 평가)")

    # 0) 엔진에 쓸 2분봉 원본이 준비됐는지 확인
    df_raw_global = st.session_state["raw_df"]
    if df_raw_global is None or df_raw_global.empty:
        st.warning("먼저 실시간 탭에서 🚀 원클릭 버튼으로 2분봉 데이터를 한 번 받아주세요.")
        st.stop()

    # 1) 며칠 전 '미국 기준 장 날짜'를 평가할지
    eval_offset_days = st.slider("며칠 전 장(미국 기준)을 평가할까요?", 1, 7, 6)
    st.info(
        f"{eval_offset_days}일 전 미국 기준 장 날짜를 평가일로 잡고, "
        f"그 전날까지의 2분봉으로 엔진을 학습시킨 뒤 "
        f"그날 프리장~정규장~애프터장을 하루 종일 예측했다고 가정해 평가합니다."
    )

    # 2) 날짜 계산 (UTC → US/Eastern 기준 장 날짜)
    now_utc = dt.datetime.now(dt.timezone.utc)
    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))

    eval_us_date = now_et.date() - dt.timedelta(days=eval_offset_days)       # 평가할 미국 장 날짜
    train_end_us_date = eval_us_date - dt.timedelta(days=1)                  # 그 전날까지로 학습

    st.write(f"📌 **훈련 종료일(미국 기준):** {train_end_us_date}")
    st.write(f"📌 **평가일(미국 기준):** {eval_us_date}")

    # 3) KST 인덱스를 US/Eastern으로 변환해서 날짜 마스크 생성
    idx_et = df_raw_global.index.tz_convert("America/New_York")

    train_mask = idx_et.date <= train_end_us_date
    eval_mask = idx_et.date == eval_us_date

    train_df = df_raw_global[train_mask]
    eval_df = df_raw_global[eval_mask]

    st.write(f"🔍 훈련용 캔들 수: {len(train_df)}")
    st.write(f"📈 평가일 캔들 수: {len(eval_df)}")

    if train_df is None or train_df.empty or len(train_df) < 200:
        st.error("훈련 데이터가 부족합니다. (최소 200캔들 필요, days 슬라이더를 늘려보세요.)")
        st.stop()

    if eval_df is None or eval_df.empty or len(eval_df) < 50:
        st.error("평가일 데이터가 너무 적습니다. (최소 50캔들 필요)")
        st.stop()

    # 4) 같은 엔진으로 다시 학습 (과거 cutoff까지)
    st.subheader("🔧 엔진 학습 (훈련 종료일까지만 사용)")

    try:
        engine_bt = run_training_pipeline(
            df_raw=train_df,
            base_horizons=base_horizons,
            custom_h=int(custom_h) if custom_h else None,
            random_state=int(random_state),
        )
    except Exception as e:
        st.error(f"과거 구간 엔진 학습 중 오류 발생: {e}")
        st.stop()

    models_bt = engine_bt["models"]
    feature_cols_bt = engine_bt["feature_cols"]
    horizons_bt = engine_bt["horizons"]
    metrics_bt = engine_bt["metrics"]

    st.write(f"사용 horizon(분): {horizons_bt}")

    if metrics_bt is not None:
        st.markdown("### 📊 이 힌드캐스트에서 사용한 엔진 성능 (훈련 데이터 내 테스트)")
        st.dataframe(
            metrics_bt.style.format(
                {"MAE": "{:.4f}", "RMSE": "{:.4f}", "direction_acc": "{:.3f}"}
            ),
            use_container_width=True,
        )

    # 5) 평가일 전체에 대해: h분 뒤 가격 예측 vs 실제
    st.subheader("🔮 하루 종일 예측 실행 중...")

    # 5-1) 평가일 피처 (같은 엔진 피처 생성 함수 사용)
    feat_eval_full = build_feature_frame(eval_df)
    feat_eval_full = feat_eval_full.dropna()

    if feat_eval_full is None or feat_eval_full.empty:
        st.error("평가일 데이터에서 유효한 피처를 만들지 못했습니다.")
        st.stop()

    # 5-2) 평가용 공통 구조
    close_raw = eval_df["Close"]
    if isinstance(close_raw, pd.DataFrame):
        close_raw = close_raw.iloc[:, 0]
    close_series = pd.to_numeric(close_raw, errors="coerce")

    idx_positions = {ts: i for i, ts in enumerate(eval_df.index)}

    # 분 → 2분봉 steps 변환 함수 (core와 동일 로직)
    def minutes_to_steps(h_min: int) -> int:
        steps = int(round(h_min / 2.0))
        return max(1, steps)

    results = []

    for ts in feat_eval_full.index:
        if ts not in idx_positions:
            continue
        pos = idx_positions[ts]

        # 현재가
        cur_val = close_series.iloc[pos]
        if not np.isfinite(cur_val):
            continue
        cur_price = float(cur_val)

        # 피처 벡터
        feat_row = feat_eval_full.loc[ts, feature_cols_bt]
        if feat_row.isna().any():
            continue
        X_row = feat_row.values.reshape(1, -1)

        # 각 horizon마다 예측
        for h in horizons_bt:
            # 미래 수익률 회귀 예측
            ret_pred = float(models_bt[h].predict(X_row)[0])
            pred_price = cur_price * (1.0 + ret_pred)

            # 실제 h분 뒤 가격
            steps = minutes_to_steps(h)
            target_idx = pos + steps
            if target_idx < len(close_series):
                actual_val = close_series.iloc[target_idx]
                actual_price = float(actual_val) if np.isfinite(actual_val) else None
            else:
                actual_price = None

            results.append(
                {
                    "time": ts,
                    "horizon": h,
                    "ret_pred": ret_pred,
                    "current_price": cur_price,
                    "pred_price": pred_price,
                    "actual_price": actual_price,
                }
            )

    res_df = pd.DataFrame(results)
    st.success("하루 전체 예측 완료!")

    # 6) horizon별 성능 요약 (정량 지표)
    st.subheader("📊 성능 요약")

    perf_rows = []
    for h in horizons_bt:
        sub = res_df[res_df["horizon"] == h].copy()
        if sub.empty:
            continue

        actual = pd.to_numeric(sub["actual_price"], errors="coerce").to_numpy()
        pred = pd.to_numeric(sub["pred_price"], errors="coerce").to_numpy()
        cur = pd.to_numeric(sub["current_price"], errors="coerce").to_numpy()

        mask = np.isfinite(actual) & np.isfinite(pred) & np.isfinite(cur)
        if mask.sum() == 0:
            continue

        actual = actual[mask]
        pred = pred[mask]
        cur = cur[mask]

        # 수익률 기준으로 다시 계산
        actual_ret = (actual / cur) - 1.0
        pred_ret = (pred / cur) - 1.0

        mae = float(np.mean(np.abs(actual_ret - pred_ret)))
        mape = float(np.mean(np.abs(actual_ret - pred_ret) / (np.abs(actual_ret) + 1e-9)))
        dir_acc = float((np.sign(actual_ret) == np.sign(pred_ret)).mean())

        perf_rows.append(
            {
                "horizon_min": h,
                "samples": int(mask.sum()),
                "direction_acc": dir_acc,
                "MAE_ret": mae,
                "MAPE_ret": mape,
            }
        )

    if perf_rows:
        perf_df = pd.DataFrame(perf_rows)
        st.dataframe(
            perf_df.style.format(
                {"direction_acc": "{:.3f}", "MAE_ret": "{:.4f}", "MAPE_ret": "{:.2%}"}
            ),
            use_container_width=True,
        )
    else:
        st.write("성능을 계산할 수 있는 유효한 샘플이 없습니다.")

    # 7) n분 뒤 예측 차트 vs 실제 차트 + 오차 그래프
    st.subheader("📉 예측 차트 vs 실제 차트")

    if len(horizons_bt) == 0:
        st.write("표시할 horizon이 없습니다.")
        st.stop()

    h_sel = st.selectbox("어떤 horizon(몇 분 뒤)을 볼까요?", sorted(horizons_bt))

    view = res_df[res_df["horizon"] == h_sel].copy()
    view["actual_num"] = pd.to_numeric(view["actual_price"], errors="coerce")
    view["pred_num"] = pd.to_numeric(view["pred_price"], errors="coerce")
    view["cur_num"] = pd.to_numeric(view["current_price"], errors="coerce")
    view = view.dropna(subset=["actual_num", "pred_num", "cur_num"])

    if view.empty:
        st.write("선택한 horizon에 대해 표시할 데이터가 없습니다.")
    else:
        # 미래 시간축 생성 (ts + h_sel 분)
        future_times = view["time"] + pd.to_timedelta(h_sel, unit="m")

        fig_price = go.Figure()
        fig_price.add_trace(
            go.Scatter(
                x=future_times,
                y=view["pred_num"],
                name=f"{h_sel}분 뒤 예상가",
                line=dict(color="#6EA6FF", dash="dot"),
            )
        )
        fig_price.add_trace(
            go.Scatter(
                x=future_times,
                y=view["actual_num"],
                name=f"{h_sel}분 뒤 실제가격",
                line=dict(color="#FF8A8A"),
            )
        )

        fig_price.update_layout(
            title=f"{ticker} — {h_sel}분 뒤 예측 vs 실제 (실제 시간축 기준, KST)",
            xaxis_title="실제 시각 (KST)",
            yaxis_title=f"{h_sel}분 뒤 가격",
            legend=dict(orientation="h"),
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_price, use_container_width=True)

        err = view["actual_num"] - view["pred_num"]
        fig_err = go.Figure()
        fig_err.add_trace(
            go.Scatter(
                x=view["time"],
                y=err,
                name="오차(실제 - 예측)",
                line=dict(color="#B7AC8D"),
            )
        )
        fig_err.add_hline(y=0, line=dict(color="#CCCCCC", width=1, dash="dot"))
        fig_err.update_layout(
            title=f"{h_sel}분 뒤 예측 오차 (실제 - 예측)",
            xaxis_title="예측 시점 (KST)",
            yaxis_title="오차",
            height=260,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_err, use_container_width=True)

        # 8) 선택한 horizon에 대한 자동 해석
        st.markdown("### 🧠 해석")

        actual = view["actual_num"].to_numpy()
        pred = view["pred_num"].to_numpy()
        cur = view["cur_num"].to_numpy()
        samples = len(actual)
        actual_ret = actual / cur - 1.0
        pred_ret = pred / cur - 1.0

        dir_acc = float((np.sign(actual_ret) == np.sign(pred_ret)).mean())
        mae = float(np.mean(np.abs(actual_ret - pred_ret)))
        mape = float(np.mean(np.abs(actual_ret - pred_ret) / (np.abs(actual_ret) + 1e-9)))

        st.write(f"- 샘플 수: **{samples}개**")
        st.write(f"- 방향 예측 정확도: **{dir_acc*100:.1f}%**")
        st.write(f"- MAE(수익률 기준): **{mae:.4f}**")
        st.write(f"- MAPE(수익률 기준): **{mape*100:.2f}%**")

        st.markdown("---")

        if dir_acc > 0.6:
            st.success("📈 방향은 꽤 잘 맞는 편입니다. 다른 필터(거래량, 지표)와 함께 쓰면 단타 시그널로 쓸 만한 수준입니다.")
        elif dir_acc > 0.52:
            st.info("➖ 방향이 약간 우위 정도입니다. 단독 매매보다는 보조 지표 느낌으로 쓰는 게 현실적입니다.")
        else:
            st.warning("📉 방향 예측력이 거의 코인 플립 수준이거나 그 이하입니다. 이 horizon은 실전에 쓰기 어렵습니다.")

        if mape < 0.3:
            st.success("🎯 수익률 기준 오차도 30% 미만이라, 대략적인 ‘방향+강도’ 감을 잡는 데는 쓸 수 있습니다.")
        else:
            st.warning("⚠ 오차가 큰 편이라, 정확한 진입/청산 가격보다는 '방향' 중심으로만 참고하는 편이 낫습니다.")
