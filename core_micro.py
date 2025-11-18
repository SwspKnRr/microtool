# core_micro.py (V2)

import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# -----------------------------------------
# 1) 1분봉 데이터
# -----------------------------------------
def fetch_1min_intraday(ticker: str, days: int = 3):
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

    df = df[df.index.dayofweek < 5]
    return df


# -----------------------------------------
# 2) 2분봉 다운로드
# -----------------------------------------
def fetch_2min_data(ticker: str, days: int = 60):
    if days > 60:
        days = 60

    df = yf.download(
        ticker,
        period=f"{days}d",
        interval="2m",
        auto_adjust=True,
        prepost=True,
        progress=False,
    )
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title)

    if hasattr(df.index, "tz"):
        df = df.tz_localize(None)

    df = df.sort_index()
    df = df[df.index.dayofweek < 5]
    return df


# -----------------------------------------
# 3) RSI
# -----------------------------------------
def compute_rsi(s: pd.Series, period=14):
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ru = up.rolling(period).mean()
    rd = down.rolling(period).mean()
    rs = ru / rd
    rsi = 100 - (100 / (1 + rs))
    return rsi


# -----------------------------------------
# 4) Feature Frame
# -----------------------------------------
def build_feature_frame(df_raw):
    df = df_raw.copy()

    df["return_2m"] = df["Close"].pct_change()

    for w in (3, 5, 10):
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()

    df["mom_1"] = df["Close"].diff(1)
    df["mom_3"] = df["Close"].diff(3)

    for p in (2, 5):
        df[f"rsi_{p}"] = compute_rsi(df["Close"], p)

    vol_mean = df["Volume"].rolling(30).mean()
    vol_std = df["Volume"].rolling(30).std()
    df["vol_z"] = (df["Volume"] - vol_mean) / (vol_std + 1e-9)
    df["vol_spike"] = (df["vol_z"] > 2).astype(int)

    df["minute"] = df.index.minute
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek

    df["is_regular"] = ((df["hour"] >= 9) & (df["hour"] < 16)).astype(int)
    df["is_premarket"] = ((df["hour"] >= 4) & (df["hour"] < 9)).astype(int)
    df["is_after"] = ((df["hour"] >= 16) | (df["hour"] < 4)).astype(int)

    return df


# -----------------------------------------
# 5) Target 생성
# -----------------------------------------
def _minutes_to_steps(h_min):
    steps = int(round(h_min / 2.0))
    return max(1, steps)


def build_targets(df_feat, base_horizons, custom_horizon):
    df = df_feat.copy()

    horizons = list(sorted(set(base_horizons)))
    if custom_horizon:
        horizons.append(int(custom_horizon))
    horizons = sorted(set(horizons))

    steps_dict = {h: _minutes_to_steps(h) for h in horizons}

    CLIP = 0.1

    for h, stp in steps_dict.items():
        fut = df["Close"].shift(-stp)
        r = fut / df["Close"] - 1
        df[f"future_ret_{h}"] = r.clip(-CLIP, CLIP)

    df = df.iloc[: -max(steps_dict.values())]
    df = df.dropna()

    return df, horizons


# -----------------------------------------
# 6) X, y 분리
# -----------------------------------------
def get_feature_target_matrices(df_model, horizons):
    exclude_exact = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
    exclude_prefix = ("future_ret_",)

    feat_cols = []
    for c in df_model.columns:
        if c in exclude_exact:
            continue
        if any(c.startswith(p) for p in exclude_prefix):
            continue
        feat_cols.append(c)

    X = df_model[feat_cols].values

    y_dict = {}
    for h in horizons:
        y_dict[h] = df_model[f"future_ret_{h}"].values

    return X, y_dict, feat_cols


# -----------------------------------------
# 7) RandomForest 회귀 모델 학습
# -----------------------------------------
def train_models(X, y_dict, random_state):
    n = X.shape[0]
    if n < 200:
        raise ValueError("샘플 부족")

    split = int(n * 0.7)
    X_tr, X_te = X[:split], X[split:]

    rows = []
    models = {}

    for h, y in y_dict.items():
        y_tr, y_te = y[:split], y[split:]

        reg = RandomForestRegressor(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=8,
            random_state=random_state,
            n_jobs=-1,
        )
        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)

        mae = mean_absolute_error(y_te, y_pred)
        rmse = mean_squared_error(y_te, y_pred) ** 0.5
        dacc = (np.sign(y_te) == np.sign(y_pred)).mean()

        rows.append({
            "horizon_min": h,
            "MAE": mae,
            "RMSE": rmse,
            "direction_acc": dacc,
            "support": len(y_te)
        })

        models[h] = reg

    return models, pd.DataFrame(rows).set_index("horizon_min")


# -----------------------------------------
# 8) 최신 1캔들 예측
# -----------------------------------------
def predict_latest(models, latest_row, feat_cols):
    X = latest_row[feat_cols].values.reshape(1, -1)
    out = {}
    for h, reg in models.items():
        out[h] = float(reg.predict(X)[0])
    return out


# -----------------------------------------
# 9) 스케일링 통계 기반 (A)
# -----------------------------------------
def compute_scaling_A(res_df):
    groups = res_df.groupby("horizon")

    scales = {}
    for h, g in groups:
        pred = g["pred_price"]
        actual = g["actual_price"]
        m = (actual / pred).median()
        if np.isfinite(m):
            scales[h] = float(m)
        else:
            scales[h] = 1.0
    return scales


# -----------------------------------------
# 10) 스케일링 ML 기반 (B)
# -----------------------------------------
def compute_scaling_B(res_df):
    scales = compute_scaling_A(res_df)
    hs = sorted(scales.keys())
    xs = np.array(hs).reshape(-1, 1)
    ys = np.array([scales[h] for h in hs])

    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    reg.fit(xs, ys)

    return reg