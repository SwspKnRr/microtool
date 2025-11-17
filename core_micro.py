# core_micro.py
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


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


# ---------- 1분봉 데이터 다운로드 ---------- #
def fetch_1min_data(ticker: str, days: int = 30) -> pd.DataFrame:
    """
    최근 days일간 1분봉 데이터 다운로드 (yfinance 사용)
    """
    df = yf.download(
        ticker,
        period=f"{days}d",
        interval="1m",
        auto_adjust=True,
        progress=False,
    )

    # yfinance는 MultiIndex(티커 포함)로 올 때도 있으니 정리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns=str.title)  # open -> Open 등

    # 인덱스를 DateTime으로 보장
    df = df.tz_localize(None) if hasattr(df.index, "tz") else df
    df = df.sort_index()

    return df


# ---------- 피처 생성 ---------- #
def build_feature_frame(
    df_raw: pd.DataFrame,
    sma_windows=(3, 5, 10),
    rsi_periods=(2, 5),
) -> pd.DataFrame:
    df = df_raw.copy()

    # 기본 수익률
    df["return_1m"] = df["Close"].pct_change()

    # 단기 이동평균
    for w in sma_windows:
        df[f"sma_{w}"] = df["Close"].rolling(w).mean()

    # 단기 모멘텀 (종가 차이)
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

    # 장 구분 (미국 ETF 기준, 한국이면 바꿔야 함)
    #  - 정규장: 9:30~16:00 (뉴욕)
    #  - 여기서는 그냥 시간만 대충 구분용으로 씀 (엄밀하게 하려면 TZ 변환 필요)
    df["is_regular"] = ((df["hour"] >= 9) & (df["hour"] < 16)).astype(int)
    df["is_premarket"] = ((df["hour"] >= 4) & (df["hour"] < 9)).astype(int)
    df["is_after"] = ((df["hour"] >= 16) | (df["hour"] < 4)).astype(int)

    return df


# ---------- 타깃(미래 수익률 & 상승 여부) 생성 ---------- #
def build_targets(
    df_feat: pd.DataFrame,
    base_horizons=(5, 10, 30),
    custom_horizon: int | None = None,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, list[int]]:
    """
    base_horizons + custom_horizon(선택) 에 대해
    - future_ret_{h}
    - y_{h}  (미래 수익률 > threshold 이면 1, 아니면 0)
    를 생성
    """
    df = df_feat.copy()

    horizons = list(sorted(set(base_horizons)))
    if custom_horizon is not None and custom_horizon > 0:
        horizons.append(int(custom_horizon))
        horizons = sorted(set(horizons))

    for h in horizons:
        future_price = df["Close"].shift(-h)
        df[f"future_ret_{h}"] = future_price / df["Close"] - 1.0
        df[f"y_{h}"] = (df[f"future_ret_{h}"] > threshold).astype(int)

    # 미래 데이터가 없는 마지막 max(h) 구간 제거 + NaN 제거
    max_h = max(horizons)
    df = df.iloc[:-max_h].copy()
    df = df.dropna()

    return df, horizons


# ---------- X, y 생성 ---------- #
def get_feature_target_matrices(
    df_model: pd.DataFrame,
    horizons: list[int],
) -> tuple[np.ndarray, dict[int, np.ndarray], list[str]]:
    """
    df_model에서 피처 컬럼과 타깃(y_{h})를 분리
    """
    # 타깃/미래수익/원시 OHLC 제외한 피처들만 사용
    exclude_prefixes = ("future_ret_", "y_")
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
        y_dict[h] = df_model[f"y_{h}"].values

    return X, y_dict, feature_cols


# ---------- 모델 학습 ---------- #
def train_models(
    X: np.ndarray,
    y_dict: dict[int, np.ndarray],
    random_state: int = 42,
) -> tuple[dict[int, RandomForestClassifier], pd.DataFrame]:
    """
    각 horizon 별로 RandomForestClassifier 학습
    - 시간 순서를 고려해서 앞 70% train, 뒤 30% test
    """
    n = X.shape[0]
    if n < 200:  # 대충 최소 샘플수
        raise ValueError("데이터가 너무 적어서 모델 학습이 어렵습니다. (최소 200 샘플 권장)")

    split_idx = int(n * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]

    models: dict[int, RandomForestClassifier] = {}
    rows = []

    for h, y in y_dict.items():
        y_train, y_test = y[:split_idx], y[split_idx:]

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))

        models[h] = clf
        rows.append(
            {
                "horizon_min": h,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "support": int(y_test.shape[0]),
            }
        )

    metrics_df = pd.DataFrame(rows).set_index("horizon_min")
    return models, metrics_df


# ---------- 최신 시점에 대한 예측 ---------- #
def predict_latest(
    models: dict[int, RandomForestClassifier],
    latest_row: pd.Series,
    feature_cols: list[str],
) -> dict[int, float]:
    """
    최신 1개 row에 대해 각 horizon 별 상승 확률 (클래스 1 확률) 반환
    """
    X_latest = latest_row[feature_cols].values.reshape(1, -1)

    probs: dict[int, float] = {}
    for h, clf in models.items():
        # predict_proba[:, 1] -> 상승(1)일 확률
        p = float(clf.predict_proba(X_latest)[0, 1])
        probs[h] = p

    return probs
