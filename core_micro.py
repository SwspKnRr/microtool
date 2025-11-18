
# core_micro.py
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------- 1분봉 실시간/단기 데이터 다운로드 (최대 7일) ---------- #
def fetch_1min_intraday(ticker: str, days: int = 3) -> pd.DataFrame:
    """
    최근 days일간 1분봉 데이터 (실시간 시그널용)
    - yfinance 1분봉 최대 7일
    - prepost=True 로 프리/애프터 포함
    - 주말 제거
    """
    if days > 7:
        days = 7

    df = yf.download(
        ticker,
        period=f"{days}d",
        interval="1m",
        auto_adjust=True,
        prepost=True,
        progress=False,
    )

    if df is None or df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title)

    if hasattr(df.index, "tz"):
        df = df.tz_localize(None)

    df = df.sort_index()

    # 주말 제거 (토:5, 일:6)
    df = df[df.index.dayofweek < 5]

    return df


# ---------- 2분봉 데이터 다운로드 (최대 60일) ---------- #
def fetch_2min_data(ticker: str, days: int = 60) -> pd.DataFrame:
    """
    최근 days일간 2분봉 데이터 다운로드 (yfinance 사용)
    - interval="2m"
    - prepost=True 로 프리/애프터 포함
    - 주말(토/일) 제거
    """
    if days > 60:
        days = 60  # yfinance 2m 최대 60일 제한

    df = yf.download(
        ticker,
        period=f"{days}d",
        interval="2m",
        auto_adjust=True,
        prepost=True,      # 프리장/애프터장 포함
        progress=False,
    )

    if df is None or df.empty:
        return df

    # MultiIndex 정리 (티커 포함되어 올 수 있음)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title)  # open -> Open 등

    # 타임존 제거 (naive datetime으로)
    if hasattr(df.index, "tz"):
        df = df.tz_localize(None)

    df = df.sort_index()

    # 주말 제거 (토:5, 일:6)
    df = df[df.index.dayofweek < 5]

    return df


# ---------- RSI 계산 ---------- #
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ---------- 피처 생성 ---------- #
def build_feature_frame(
    df_raw: pd.DataFrame,
    sma_windows=(3, 5, 10),
    rsi_periods=(2, 5),
) -> pd.DataFrame:
    df = df_raw.copy()

    # 기본 수익률 (2분 수익률)
    df["return_2m"] = df["Close"].pct_change()

    # 단기 이동평균
    for w in sma_windows:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()

    # 단기 모멘텀
    df["mom_1"] = df["Close"].diff(1)
    df["mom_3"] = df["Close"].diff(3)

    # RSI
    for p in rsi_periods:
        df[f"rsi_{p}"] = compute_rsi(df["Close"], period=p)

    # 거래량 스파이크 (Z-Score)
    vol_mean = df["Volume"].rolling(30).mean()
    vol_std = df["Volume"].rolling(30).std()
    df["vol_z"] = (df["Volume"] - vol_mean) / (vol_std + 1e-9)
    df["vol_spike"] = (df["vol_z"] > 2).astype(int)

    # 시간 관련 피처
    df["minute"] = df.index.minute
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek

    # 장 구분 (미국 기준 대략적인 구분)
    # 프리장: 4~9시, 정규장: 9~16시, 애프터: 16~24 & 0~4
    df["is_regular"] = ((df["hour"] >= 9) & (df["hour"] < 16)).astype(int)
    df["is_premarket"] = ((df["hour"] >= 4) & (df["hour"] < 9)).astype(int)
    df["is_after"] = ((df["hour"] >= 16) | (df["hour"] < 4)).astype(int)

    return df


# ---------- 타깃(미래 수익률) 생성 ---------- #
def _minutes_to_steps(h_min: int) -> int:
    """
    2분봉 기준으로, h_min (분)을 몇 캔들 뒤로 볼지 변환.
    ex) 5분 → 2~3캔들 중 2캔들로 round.
    """
    steps = int(round(h_min / 2.0))
    return max(1, steps)


def build_targets(
    df_feat: pd.DataFrame,
    base_horizons=(5, 10, 30),
    custom_horizon: int | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    df = df_feat.copy()

    horizons = list(sorted(set(base_horizons)))
    if custom_horizon is not None and custom_horizon > 0:
        horizons.append(int(custom_horizon))
        horizons = sorted(set(horizons))

    steps_dict = {h: _minutes_to_steps(h) for h in horizons}

    # ★ 여기서 타깃 수익률 계산 + 클리핑
    CLIP = 0.1  # ±10% 이상은 잘라버리기

    for h, steps in steps_dict.items():
        future_price = df["Close"].shift(-steps)
        future_ret = future_price / df["Close"] - 1.0

        # 미친 값들 잘라버리기
        future_ret = future_ret.clip(-CLIP, CLIP)

        df[f"future_ret_{h}"] = future_ret

    max_steps = max(steps_dict.values())
    df = df.iloc[:-max_steps].copy()
    df = df.dropna()

    return df, horizons



# ---------- X, y 생성 ---------- #
def get_feature_target_matrices(
    df_model: pd.DataFrame,
    horizons: list[int],
) -> tuple[np.ndarray, dict[int, np.ndarray], list[str]]:
    """
    df_model에서 피처 컬럼과 타깃(future_ret_{h})를 분리
    """
    exclude_prefixes = ("future_ret_",)
    exclude_exact = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

    feature_cols = []
    for c in df_model.columns:
        if c in exclude_exact:
            continue
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        feature_cols.append(c)

    X = df_model[feature_cols].values

    y_dict: dict[int, np.ndarray] = {}
    for h in horizons:
        y_dict[h] = df_model[f"future_ret_{h}"].values

    return X, y_dict, feature_cols


# ---------- 모델 학습 (회귀) ---------- #
def train_models(
    X: np.ndarray,
    y_dict: dict[int, np.ndarray],
    random_state: int = 42,
) -> tuple[dict[int, RandomForestRegressor], pd.DataFrame]:
    """
    각 horizon 별로 RandomForestRegressor 학습
    - 시간 순서를 고려해서 앞 70% train, 뒤 30% test
    - 메트릭: MAE, RMSE, 방향 정확도(수익률 부호)
    """
    n = X.shape[0]
    if n < 200:
        raise ValueError("데이터가 너무 적어서 모델 학습이 어렵습니다. (최소 200 샘플 권장)")

    split_idx = int(n * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]

    models: dict[int, RandomForestRegressor] = {}
    rows = []

    for h, y in y_dict.items():
        y_train, y_test = y[:split_idx], y[split_idx:]

        reg = RandomForestRegressor(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=8,
            random_state=random_state,
            n_jobs=-1,
        )
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        # 방향 정확도: 수익률의 부호 기준
        sign_true = np.sign(y_test)
        sign_pred = np.sign(y_pred)
        dir_acc = float((sign_true == sign_pred).mean())

        models[h] = reg
        rows.append(
            {
                "horizon_min": h,
                "MAE": mae,
                "RMSE": rmse,
                "direction_acc": dir_acc,
                "support": int(y_test.shape[0]),
            }
        )

    metrics_df = pd.DataFrame(rows).set_index("horizon_min")
    return models, metrics_df


# ---------- 최신 시점에 대한 예측 (미래 수익률) ---------- #
def predict_latest(
    models: dict[int, RandomForestRegressor],
    latest_row: pd.Series,
    feature_cols: list[str],
) -> dict[int, float]:
    """
    최신 1개 row에 대해 각 horizon 별 미래 수익률 예측값 반환.
    반환값: {horizon_min: future_ret_pred}
    """
    X_latest = latest_row[feature_cols].values.reshape(1, -1)

    rets: dict[int, float] = {}
    for h, reg in models.items():
        r = float(reg.predict(X_latest)[0])
        rets[h] = r

    return rets