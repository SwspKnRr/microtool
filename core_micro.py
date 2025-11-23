# core_micro.py
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


# ---------- 1분봉 시세/거래 데이터 다운로드 (최대 7일) ---------- #
def fetch_1min_intraday(ticker: str, days: int = 3) -> pd.DataFrame:
    """
    최근 days동안 1분봉 데이터 (실시간/로그용)
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

    # 주말 제거 (토, 일)
    df = df[df.index.dayofweek < 5]

    return df


# ---------- 2분봉 데이터 다운로드 (최대 60일) ---------- #
def fetch_2min_data(ticker: str, days: int = 60) -> pd.DataFrame:
    """
    최근 days동안 2분봉 데이터 다운로드 (yfinance 사용)
    - interval="2m"
    - prepost=True 로 프리/애프터 포함
    - 주말 제거
    """
    if days > 60:
        days = 60  # yfinance 2m 최대 60일 제한

    df = yf.download(
        ticker,
        period=f"{days}d",
        interval="2m",
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
    df = df[df.index.dayofweek < 5]

    return df


# ---------- 인디케이터들 ---------- #
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range for volatility; uses High/Low/Close.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def winsorize_iqr(series: pd.Series, whisker: float = 1.5, clip_abs: float = 0.2) -> pd.Series:
    """
    Soften extreme tails: clamp to IQR whisker bounds and also hard cap to clip_abs.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - whisker * iqr
    upper = q3 + whisker * iqr

    series = series.clip(lower, upper)
    series = series.clip(-clip_abs, clip_abs)
    return series


# ---------- 피처 생성 ---------- #
def build_feature_frame(
    df_raw: pd.DataFrame,
    sma_windows=(3, 5, 10),
    ema_windows=(5, 10),
    rsi_periods=(2, 5),
    bollinger_window: int = 20,
) -> pd.DataFrame:
    df = df_raw.copy()

    # 기본 수익률(2분 수익률) + 로그수익률
    df["return_2m"] = df["Close"].pct_change()
    df["log_return_2m"] = np.log(df["Close"]).diff()

    # 이동평균/EMA
    for w in sma_windows:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()
    for w in ema_windows:
        df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

    # 단기 모멘텀
    df["mom_1"] = df["Close"].diff(1)
    df["mom_3"] = df["Close"].diff(3)

    # RSI
    for p in rsi_periods:
        df[f"rsi_{p}"] = compute_rsi(df["Close"], period=p)

    # Bollinger Bands
    bb_mid = df["Close"].rolling(bollinger_window).mean()
    bb_std = df["Close"].rolling(bollinger_window).std()
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (bb_mid + 1e-9)

    # Rolling volatility (returns 기반)
    ret = df["Close"].pct_change()
    df["vol_std_10"] = ret.rolling(10).std()
    df["vol_std_30"] = ret.rolling(30).std()

    # ATR
    df["atr_14"] = compute_atr(df, period=14)

    # VWAP (누적)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol_cum = df["Volume"].cumsum().replace(0, np.nan)
    df["vwap"] = (typical_price * df["Volume"]).cumsum() / vol_cum

    # OHLC 편차 (종가 대비)
    df["high_close_pct"] = (df["High"] - df["Close"]) / (df["Close"] + 1e-9)
    df["low_close_pct"] = (df["Close"] - df["Low"]) / (df["Close"] + 1e-9)
    df["open_close_pct"] = (df["Open"] - df["Close"]) / (df["Close"] + 1e-9)

    # 거래량 - 스파이크 (Z-Score)
    vol_mean = df["Volume"].rolling(30).mean()
    vol_std = df["Volume"].rolling(30).std()
    df["vol_z"] = (df["Volume"] - vol_mean) / (vol_std + 1e-9)
    df["vol_spike"] = (df["vol_z"] > 2).astype(int)

    # 시간 관련 피처
    df["minute"] = df.index.minute
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek

    # 세션 구분 (미국 기준 시계열적 구분)
    df["is_regular"] = ((df["hour"] >= 9) & (df["hour"] < 16)).astype(int)
    df["is_premarket"] = ((df["hour"] >= 4) & (df["hour"] < 9)).astype(int)
    df["is_after"] = ((df["hour"] >= 16) | (df["hour"] < 4)).astype(int)

    return df


# ---------- 타깃(미래 수익률) 생성 ---------- #
def _minutes_to_steps(h_min: int) -> int:
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

    for h, steps in steps_dict.items():
        future_price = df["Close"].shift(-steps)
        future_ret = future_price / df["Close"] - 1.0
        future_ret = winsorize_iqr(future_ret)
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


# ---------- 모델 학습 (RF/HGB + 방향 분류기) ---------- #
def _train_best_regressor(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    n_splits: int,
) -> tuple[object, dict[str, float]]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    candidates = {
        "rf": RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=6,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hgb": HistGradientBoostingRegressor(random_state=random_state),
    }

    best_name = None
    best_model = None
    best_rmse = np.inf
    best_metrics = {"MAE": np.nan, "RMSE": np.nan, "direction_acc": np.nan}

    for name, base_model in candidates.items():
        maes = []
        rmses = []
        dirs = []
        for train_idx, test_idx in tscv.split(X):
            reg = clone(base_model)
            reg.fit(X[train_idx], y[train_idx])
            y_pred = reg.predict(X[test_idx])
            maes.append(mean_absolute_error(y[test_idx], y_pred))
            rmses.append(np.sqrt(mean_squared_error(y[test_idx], y_pred)))
            dirs.append(np.sign(y[test_idx]) == np.sign(y_pred))
        mae_avg = float(np.mean(maes))
        rmse_avg = float(np.mean(rmses))
        dir_acc = float(np.mean([d.mean() for d in dirs])) if dirs else np.nan

        if rmse_avg < best_rmse:
            best_rmse = rmse_avg
            best_name = name
            best_model = clone(base_model)
            best_metrics = {"MAE": mae_avg, "RMSE": rmse_avg, "direction_acc": dir_acc}

    best_model.fit(X, y)
    return best_model, best_metrics


def train_models(
    X: np.ndarray,
    y_dict: dict[int, np.ndarray],
    random_state: int = 42,
) -> tuple[dict[int, object], dict[int, LogisticRegression], pd.DataFrame]:
    n = X.shape[0]
    if n < 200:
        raise ValueError("데이터가 너무 적어 모델 학습이 어렵습니다. (최소 200 샘플 권장)")

    n_splits = min(5, max(2, n // 150))

    models: dict[int, object] = {}
    dir_models: dict[int, LogisticRegression] = {}
    rows = []

    for h, y in y_dict.items():
        reg_model, reg_metrics = _train_best_regressor(X, y, random_state, n_splits)
        models[h] = reg_model

        y_dir = (y > 0).astype(int)
        dir_clf = LogisticRegression(max_iter=500, solver="lbfgs")
        dir_clf.fit(X, y_dir)
        dir_models[h] = dir_clf

        # 방향 분류기 정확도 (마지막 폴드 기준)
        dir_acc_clf = np.nan
        try:
            last_train_idx, last_test_idx = list(TimeSeriesSplit(n_splits=n_splits).split(X))[-1]
            dir_pred = dir_clf.predict(X[last_test_idx])
            dir_acc_clf = float((dir_pred == y_dir[last_test_idx]).mean())
        except Exception:
            dir_acc_clf = np.nan

        rows.append(
            {
                "horizon_min": h,
                "MAE": reg_metrics["MAE"],
                "RMSE": reg_metrics["RMSE"],
                "direction_acc_reg": reg_metrics["direction_acc"],
                "direction_acc_clf": dir_acc_clf,
                "direction_acc": reg_metrics["direction_acc"],
                "support": int(len(y)),
            }
        )

    metrics_df = pd.DataFrame(rows).set_index("horizon_min")
    return models, dir_models, metrics_df


# ---------- 최신 시점의 미래 수익률 예측 ---------- #
def predict_latest(
    models: dict[int, object],
    latest_row: pd.Series,
    feature_cols: list[str],
    dir_models: dict[int, LogisticRegression] | None = None,
    min_confidence: float = 0.55,
) -> tuple[dict[int, float], dict[int, float]]:
    X_latest = latest_row[feature_cols].values.reshape(1, -1)

    rets: dict[int, float] = {}
    dir_probs: dict[int, float] = {}

    for h, reg in models.items():
        r = float(reg.predict(X_latest)[0])

        if dir_models is not None and h in dir_models:
            clf = dir_models[h]
            prob_up = float(clf.predict_proba(X_latest)[0][1])
            sign_clf = 1.0 if prob_up >= 0.5 else -1.0
            confidence = max(prob_up, 1.0 - prob_up)
            dir_probs[h] = prob_up

            # 방향 불일치 시 약화 + 방향 보정
            if np.sign(r) != sign_clf:
                r = abs(r) * sign_clf * 0.5
            # 자신감 낮으면 크기 축소
            if confidence < min_confidence:
                r *= 0.5

        rets[h] = r

    return rets, dir_probs
