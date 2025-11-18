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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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
    # 예: open=22:30, close=05:00 이면
    # 22:30~24:00, 00:00~05:00 모두 정규장
    if t >= open_kst or t < close_kst:
        return True
    return False


def minutes_to_close_kst(ts: pd.Timestamp, open_kst: dt.time, close_kst: dt.time) -> int | None:
    """
    현재 시각(ts, KST 기준)에서 미국 정규장 '폐장시각'까지 남은 분 수 계산.
    정규장이 아니면 None 반환.
    """
    if ts.tz is None:
        # 안전장치: KST로 가정
        ts = ts.tz_localize("Asia/Seoul")

    t = ts.time()
    if not is_regular_session_kst(ts, open_kst, close_kst):
        return None

    # 폐장 시간은 항상 '다음날 새벽' 기준으로 계산
    if t >= open_kst:
        # 예: 밤 23시 → 다음날 새벽 close_kst 시각이 폐장
        close_dt = ts.replace(
            hour=close_kst.hour,
            minute=close_kst.minute,
            second=0,
            microsecond=0,
        ) + dt.timedelta(days=1)
    else:
        # 이미 0시~close_kst 사이(새벽)인 경우 → 같은 날 close_kst
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
    # 정규장 이전이면 프리장, 이후면 애프터장으로 간단하게 처리
    if t < open_kst:
        return "프리장(Pre-market)"
    return "애프터장(After-hours)"


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
        "raw_df": None,          # 2분봉 데이터 (KST)
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
    st.header("⚙ 설정")

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
    use_dst = st.checkbox("미국 써머타임(DST) 적용", value=True)
    st.caption(
        "- ON: 미국 써머타임 기간 기준 (정규장 KST 22:30~05:00)\n"
        "- OFF: 써머타임 미적용 기준 (정규장 KST 23:30~06:00)"
    )

    st.markdown("---")
    st.caption("① 2분봉 데이터 다운로드 → ② 피처/타깃 생성 → ③ 모델 학습 → ④ 1분봉 실시간 시그널 → ⑤ 예측 정확도 확인")


# ---------- 메인 탭 구성 ---------- #
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1️⃣ 데이터 다운로드 (2분봉)",
        "2️⃣ 피처 & 타깃 생성",
        "3️⃣ 모델 학습",
        "4️⃣ 실시간 시그널 (1분봉)",
        "5️⃣ 얼마나 정확했나?",
    ]
)


# ==================== 1) 데이터 탭 ==================== #
with tab1:
    st.subheader("1️⃣ 2분봉 데이터 다운로드 (프리/데이/애프터 포함, KST 변환)")

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("📥 2분봉 데이터 불러오기"):
            with st.spinner("데이터 다운로드 중..."):
                try:
                    df_raw = fetch_2min_data(ticker, days=days)
                    if df_raw is None or df_raw.empty:
                        raise ValueError("받아온 데이터가 비어 있습니다.")
                    df_raw = to_kst(df_raw)
                except Exception as e:
                    st.error(f"데이터 다운로드/변환 중 오류 발생: {e}")
                else:
                    st.session_state["raw_df"] = df_raw
                    st.success(
                        f"{ticker} 최근 {days}일 2분봉 데이터 다운로드 완료! (KST 변환, 주말 제외)"
                    )

    with col2:
        df_raw = st.session_state["raw_df"]
        if df_raw is not None:
            st.write("🔹 데이터 샘플 (최근 10개, 인덱스=KST)")
            st.dataframe(df_raw.tail(10))
        else:
            st.info("좌측에서 데이터를 먼저 다운로드하세요.")

    if st.session_state["raw_df"] is not None:
        df_raw = st.session_state["raw_df"]
        st.markdown("---")
        st.write("📊 종가 간단 라인 차트 (최근 500캔들, KST)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_raw["Close"].tail(500))
        ax.set_title(f"{ticker} 2분봉 종가 (최근 500캔들, KST)")
        ax.set_xlabel("시간 (KST)")
        ax.set_ylabel("가격")
        st.pyplot(fig)


# ==================== 2) 피처 & 타깃 생성 탭 ==================== #
with tab2:
    st.subheader("2️⃣ 피처 & 타깃 생성 (2분봉 기반, KST 인덱스)")

    df_raw = st.session_state["raw_df"]
    if df_raw is None:
        st.warning("먼저 2분봉 데이터를 다운로드 해주세요. (탭 1)")
    else:
        st.write(
            f"티커: **{ticker}**, 최근 **{days}일** 2분봉(KST) 기준으로 "
            f"피처/타깃을 생성합니다."
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
            st.markdown("### 🔍 피처/타깃 데이터 샘플 (최근 10개)")
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
    st.subheader("3️⃣ 모델 학습 (2분봉 피처 기반, RandomForest)")

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


# ==================== 4) 실시간 시그널 탭 (1분봉, KST) ==================== #
with tab4:
    st.subheader("4️⃣ 실시간 시그널 (1분봉 / KST 기준 / 모델 보정 예상가 + 30분 전 예측 리뷰)")

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

        st.markdown("### 🔮 현재(가장 최근 2분봉, KST) 기준 예측 결과")

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
        st.markdown("### 🕯 1분봉 실시간 캔들 차트 (KST) + 현재가 + 모델 보정 예상 가격")

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

        # 수동 새로고침 버튼 → 즉시 rerun
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
            df_plot = intraday_df.tail(n_candles).copy()

            last_price = df_plot["Close"].iloc[-1]
            last_time = df_plot.index[-1]  # KST

            open_kst, close_kst = get_kst_session_times(use_dst)

            # ===== 최근 추세 기반 여러 시간대 예상 가격 + 모델 확률 보정 ===== #
            reg_window = min(50, len(df_plot))
            y_arr = df_plot["Close"].tail(reg_window).values
            x_arr = np.arange(reg_window)

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
                slope, intercept = np.polyfit(x_arr, y_arr, 1)

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
                        base_w = 0.3  # 최소 추세 비중
                        confidence = 2 * abs(p_up - 0.5)  # 0~1
                        w = base_w + (1 - base_w) * confidence
                        w = float(np.clip(w, 0.0, 1.0))

                        p_adj = (1 - w) * last_price + w * p_trend
                        preds[h_min] = p_adj

                # 2) 종가 예상 (KST 기준 미국 폐장 시각)
                if show_close_chk:
                    minutes_to_close = minutes_to_close_kst(last_time, open_kst, close_kst)
                    if minutes_to_close is not None and minutes_to_close > 0:
                        p_trend_close = last_price + slope * minutes_to_close
                        p_up_close = get_nearest_model_prob(minutes_to_close)

                        if p_up_close is None:
                            pred_close = p_trend_close
                        else:
                            base_w = 0.3
                            confidence = 2 * abs(p_up_close - 0.5)
                            w = base_w + (1 - base_w) * confidence
                            w = float(np.clip(w, 0.0, 1.0))
                            pred_close = (1 - w) * last_price + w * p_trend_close
                    else:
                        pred_close = None

                # ----- 예측 로그 저장 (5분 / 10분 / 1시간 / 6시간 / 1일) ----- #
                if st.session_state["pred_log"] is None:
                    st.session_state["pred_log"] = pd.DataFrame(
                        columns=["made_at", "horizon_min", "base_price", "pred_price", "eval_time"]
                    )

                last_logged = st.session_state.get("last_logged_time", None)

                # 같은 1분봉 캔들에 대해 중복 로그 안 남기도록: 새로운 캔들일 때만 기록
                if (last_logged is None) or (last_time > last_logged):
                    log_horizons = [5, 10, 60, 360, 1440]  # 5분, 10분, 1시간, 6시간, 1일

                    new_rows = []
                    for h_log in log_horizons:
                        # 1) 단순 추세 기반 예상가
                        p_trend_h = last_price + slope * h_log

                        # 2) 해당 시간에 가장 가까운 모델 horizon 확률
                        p_up_h = get_nearest_model_prob(h_log)

                        if p_up_h is None:
                            p_adj_h = p_trend_h
                        else:
                            base_w = 0.3
                            confidence = 2 * abs(p_up_h - 0.5)  # 0~1
                            w_h = base_w + (1 - base_w) * confidence
                            w_h = float(np.clip(w_h, 0.0, 1.0))
                            p_adj_h = (1 - w_h) * last_price + w_h * p_trend_h

                        eval_time = last_time + dt.timedelta(minutes=h_log)

                        new_rows.append(
                            {
                                "made_at": last_time,
                                "horizon_min": h_log,
                                "base_price": last_price,
                                "pred_price": p_adj_h,
                                "eval_time": eval_time,
                            }
                        )

                    if new_rows:
                        st.session_state["pred_log"] = pd.concat(
                            [st.session_state["pred_log"], pd.DataFrame(new_rows)],
                            ignore_index=True,
                        )
                        st.session_state["last_logged_time"] = last_time

            # ===== 30분 전에 예상했던 현재가 (과거 예측 검증, KST 기준) ===== #
            back_result = None
            try:
                t_now = intraday_df.index[-1]  # KST
                t_back = t_now - dt.timedelta(minutes=30)

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
                    idx_candidates = df2.index[df2.index <= t_back]
                    if len(idx_candidates) > 0:
                        idx_back = idx_candidates[-1]
                        past_row = df2.loc[idx_back]
                        past_probs = predict_latest(models, past_row, feature_cols)

                        model_hs_back = list(past_probs.keys())
                        nearest_h_back = min(model_hs_back, key=lambda H: abs(H - 30))
                        p_up_back = past_probs[nearest_h_back]

                        base_w = 0.3
                        confidence = 2 * abs(p_up_back - 0.5)
                        w_back = base_w + (1 - base_w) * confidence
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
                back_result = None

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
                                line=dict(color="#FF8A8A"),   # 파스텔 레드
                                fillcolor="#FF8A8A",
                            ),
                            decreasing=dict(
                                line=dict(color="#6EA6FF"),   # 파스텔 블루
                                fillcolor="#6EA6FF",
                            ),
                            name="1분봉",
                        )
                    ]
                )

                # 드래그 줌 없애고, Zoom in/out 버튼은 유지
                fig_c.update_layout(
                    dragmode=False,
                    xaxis=dict(fixedrange=True),
                    yaxis=dict(fixedrange=True),
                    modebar_remove=[
                        "zoom",       # drag zoom
                        "select",
                        "lasso2d",
                        "pan",
                        "resetScale2d",
                    ],
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
                    title=f"{ticker} 1분봉 캔들 (최근 {n_candles}개, KST)",
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
                        f"30분 전 시점: {back_result['time_back'].strftime('%Y-%m-%d %H:%M')}"
                    )
                    st.write(f"그때 30분 뒤 예상가: {back_result['pred']:.2f}")
                    st.write(f"현재 실제가: {back_result['actual']:.2f}")
                    st.write(
                        f"오차: {back_result['error']:+.2f} ({back_result['error_pct']*100:+.2f}%)"
                    )
                else:
                    st.write("30분 전 예측값을 계산할 수 있는 데이터가 부족합니다.")

                st.markdown("#### 🕒 시각 (KST)")
                st.write(last_time.strftime("%Y-%m-%d %H:%M:%S"))

                session_label = get_session_label_kst(last_time, open_kst, close_kst)
                st.caption(f"현재 세션 추정: {session_label} (KST 기준)")

                st.markdown("---")
                st.caption(
                    "※ 모든 시간은 한국시간(KST, UTC+9) 기준입니다.\n"
                    "※ 정규장 시간대는 DST 체크박스에 따라 KST 22:30~05:00 또는 23:30~06:00으로 간주됩니다.\n"
                    "※ 예상 가격은 최근 1분봉 추세 + 2분봉 모델 상승 확률을 함께 반영한 단순 보정값입니다."
                )

            st.markdown("#### 🔎 최근 1분봉 원시 데이터 (마지막 5개, KST)")
            st.dataframe(intraday_df.tail(5))
        else:
            st.info("1분봉 데이터를 가져오지 못했습니다. 티커/시간대를 다시 확인해 주세요.")

        # 자동 새로고침 로직 (간단한 5초 주기)
        if auto_refresh:
            time.sleep(5)
            st.rerun()


# ============================
# 📊 5번 탭: 하루 힌드캐스트 테스트
# ============================

with tab5:
    st.header("📅 하루 힌드캐스트 테스트 (과거 하루 예측 시뮬레이션)")

    # ----- UI: 몇 일 전 하루를 테스트할지 -----
    eval_offset_days = st.slider("며칠 전 하루를 평가할까요?", 1, 7, 6)
    st.info(f"{eval_offset_days}일 전 하루를 예측해보고 실제와 비교합니다.")

    # ----- 현재 시각 -----
    now = dt.datetime.now(dt.timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))

    # ----- 평가 날짜 정의 -----
    eval_date = (now.date() - dt.timedelta(days=eval_offset_days))
    train_end_date = (now.date() - dt.timedelta(days=eval_offset_days + 1))

    st.write(f"📌 **평가할 날짜:** {eval_date}")
    st.write(f"📌 **훈련 데이터 종료일:** {train_end_date}")

        # =============================
    # 1) 훈련 데이터 로딩 (train_end_date까지)
    # =============================
    def load_train_df():
        """
        하루 힌드캐스트용 훈련 데이터:
        - fetch_2min_data()로 최근 60일 2분봉을 받는다.
        - KST로 변환 후, train_end_date 이전까지만 사용.
        """
        df = fetch_2min_data(ticker, days=60)
        if df is None or df.empty:
            return df

        # 이미 1번 탭에서 쓰던 것과 동일한 방식으로 KST 변환
        df = to_kst(df)

        # 평가일 직전까지만 사용 (train_end_date 기준)
        df = df[df.index.date <= train_end_date]

        return df.dropna()

    train_df = load_train_df()

    # ----- 피처 생성 -----
    def make_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        2분봉/1분봉 DataFrame(df)에서 피처 프레임 생성.
        - Close, Volume 컬럼 기준
        - 인덱스: df.index (DatetimeIndex, KST)
        """
    # 이상한 입력이면 바로 빈 DF 반환
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
             return pd.DataFrame()

        if "Close" not in df.columns:
             return pd.DataFrame()

    # ---- Close 처리: Series / DataFrame 모두 커버 ----
        close_raw = df["Close"]
        if isinstance(close_raw, pd.DataFrame):
        # 멀티인덱스에서 ('Close', 티커) 형태면 한 컬럼만 사용
             close_raw = close_raw.iloc[:, 0]
        close = pd.to_numeric(close_raw, errors="coerce")

    # ---- Volume 처리: 없으면 NaN 시리즈, DataFrame이면 첫 컬럼 ----
        if "Volume" in df.columns:
            vol_raw = df["Volume"]
            if isinstance(vol_raw, pd.DataFrame):
                vol_raw = vol_raw.iloc[:, 0]
        else:
            vol_raw = pd.Series(index=df.index, data=np.nan)

        vol = pd.to_numeric(vol_raw, errors="coerce")

    # 🔹 먼저 인덱스만 가진 빈 DF 생성
        X = pd.DataFrame(index=df.index)

    # 🔹 컬럼 하나씩 추가 (전부 1차원 Series라 안전)
        X["ret1"] = close.pct_change()
        X["ma5"] = close.rolling(5).mean()
        X["ma20"] = close.rolling(20).mean()
        X["vol"] = vol
        X["trend"] = close.diff()

    # 초반부 NaN, 이상치 제거
        X = X.dropna()

        return X




    # horizon 설정 (5, 10, 30분 등 필요하면 변경)
    horizons = [5, 10, 30]

    # 타깃 생성
    def make_target(df, horizon):
        return (df["Close"].shift(-horizon) > df["Close"]).astype(int)

        # ----- X, y 생성 -----
    X_train = make_features(train_df)

    if X_train is None or X_train.empty or len(X_train) < 50:
        st.error("훈련 데이터에서 유효한 피처를 만들지 못했습니다. (샘플 수 부족)")
        st.stop()

    y_train_dict = {
        h: make_target(train_df, h).loc[X_train.index]
        for h in horizons
    }


    # =============================
    # 2) 모델 학습
    # =============================
    st.subheader("🔧 모델 학습 중...")
    models = {}
    for h in horizons:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42
        )
        rf.fit(X_train, y_train_dict[h])
        models[h] = rf

    st.success("모델 학습 완료!")

    # =============================
    # 3) 평가일 하루 전체 데이터 로드
    # =============================
    def load_eval_day():
        df = yf.download(
            ticker,
            start=eval_date,
            end=(eval_date + dt.timedelta(days=1)),
            interval="2m",
            prepost=True,
            progress=False
        )
        if df is None or df.empty:
            return df
        df = to_kst(df)
        df = df[df.index.date == eval_date]
        return df.dropna()

    eval_df = load_eval_day()
    st.write(f"📈 평가일 데이터 개수: {len(eval_df) if eval_df is not None else 0}")

    if eval_df is None or eval_df.empty or len(eval_df) < 50:
        st.error("평가일 데이터가 너무 적음.")
        st.stop()

    # =============================
    # 4) 하루 종일 예측 루프
    # =============================
    st.subheader("🔮 하루 종일 예측 실행 중...")

    results = []

    close_series = eval_df["Close"]

    for t_idx in range(20, len(eval_df)):

        # 시점 t까지의 데이터만 사용
        hist = eval_df.iloc[:t_idx]

        X_hist = make_features(hist)
        if len(X_hist) < 20:
            continue

        cur_time = hist.index[-1]
        cur_close = hist["Close"].iloc[-1]

        for h in horizons:
            rf = models[h]

            # 방향 확률
            prob = rf.predict_proba(X_hist.iloc[-1:])[0, 1]

            # 실제 가격 (t+h)
            if t_idx + h < len(eval_df):
                actual_price = close_series.iloc[t_idx + h]
            else:
                actual_price = None

            results.append({
                "time": cur_time,
                "horizon": h,
                "pred_prob": prob,
                "current_price": cur_close,
                "actual_price": actual_price,
            })

    res_df = pd.DataFrame(results)

    st.success("하루 전체 예측 완료!")

        # =============================
    # 5) 성능 계산
    # =============================
    st.subheader("📊 성능 요약")

    perf_rows = []
    for h in horizons:
        # 해당 horizon만 추출
        sub = res_df[res_df["horizon"] == h].copy()

        # 실제/현재/확률이 모두 있는 행만 사용
        sub = sub.dropna(subset=["actual_price", "current_price", "pred_prob"])
        if sub.empty:
            continue

        # 숫자로 강제 변환 (이 단계에서 object/문자/이상 타입 정리)
        sub["actual_price_num"] = pd.to_numeric(sub["actual_price"], errors="coerce")
        sub["current_price_num"] = pd.to_numeric(sub["current_price"], errors="coerce")
        sub["pred_prob_num"] = pd.to_numeric(sub["pred_prob"], errors="coerce")

        sub = sub.dropna(subset=["actual_price_num", "current_price_num", "pred_prob_num"])
        if sub.empty:
            continue

        # numpy 배열로 꺼내서 순수 수치 연산 (pandas 비교 버그 회피)
        actual_price = sub["actual_price_num"].to_numpy()
        current_price = sub["current_price_num"].to_numpy()
        pred_prob = sub["pred_prob_num"].to_numpy()

        actual_dir = (actual_price > current_price).astype(int)
        pred_dir = (pred_prob > 0.5).astype(int)

        acc = (actual_dir == pred_dir).mean()
        mae = np.abs(actual_price - current_price).mean()
        mape = (np.abs(actual_price - current_price) / current_price).mean()

        perf_rows.append({
            "horizon": h,
            "samples": len(sub),
            "accuracy": acc,
            "MAE": mae,
            "MAPE": mape,
        })

    if perf_rows:
        perf_df = pd.DataFrame(perf_rows)
        st.dataframe(perf_df, use_container_width=True)
    else:
        st.write("성능을 계산할 수 있는 유효한 샘플이 없습니다.")


    perf_df = pd.DataFrame(perf_rows)
    st.dataframe(perf_df, use_container_width=True)

        # =============================
    # 6) 차트 시각화
    # =============================
    st.subheader("📉 예측 vs 실제 차트")

    h_sel = st.selectbox("어떤 horizon을 볼까요?", horizons)

    view_df = res_df[res_df["horizon"] == h_sel].copy()
    view_df = view_df.dropna(subset=["actual_price", "current_price", "pred_prob"])

    if view_df.empty:
        st.write("선택한 horizon에 대해 표시할 데이터가 없습니다.")
    else:
        view_df["actual_price_num"] = pd.to_numeric(view_df["actual_price"], errors="coerce")
        view_df["current_price_num"] = pd.to_numeric(view_df["current_price"], errors="coerce")
        view_df["pred_prob_num"] = pd.to_numeric(view_df["pred_prob"], errors="coerce")

        view_df = view_df.dropna(subset=["actual_price_num", "current_price_num", "pred_prob_num"])
        if view_df.empty:
            st.write("선택한 horizon에 대해 표시할 데이터가 없습니다.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=view_df["time"],
                y=view_df["current_price_num"],
                name="현재가",
                line=dict(color="gray")
            ))
            fig.add_trace(go.Scatter(
                x=view_df["time"],
                y=view_df["actual_price_num"],
                name="실제 H분 뒤 가격",
                line=dict(color="red")
            ))
            fig.add_trace(go.Scatter(
                x=view_df["time"],
                y=view_df["current_price_num"] * (1 + view_df["pred_prob_num"] * 0.004),
                name="예측 경향선",
                line=dict(color="blue", dash="dot")
            ))

            st.plotly_chart(fig, use_container_width=True)

