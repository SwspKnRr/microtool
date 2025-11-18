# core_micro.py
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt

#############################################
# 1. 데이터 다운로드 (1분봉)
#############################################
def fetch_1min_intraday(ticker: str, days: int = 10):
    """
    1분봉 데이터를 n일치 가져온다.
    yfinance는 7일 이상 1분봉 제공 안 하므로 1일씩 나눠서 병합한다.
    """

    frames = []
    for i in range(days):
        end = dt.datetime.now() - dt.timedelta(days=i)
        start = end - dt.timedelta(days=1)
        df = yf.download(
            ticker,
            interval="1m",
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
        )
        if df is not None and len(df) > 0:
            df["date"] = df.index
            frames.append(df)

    if len(frames) == 0:
        return pd.DataFrame()

    df_all = pd.concat(frames).sort_index()
    df_all = df_all[~df_all.index.duplicated(keep="first")]
    return df_all


#############################################
# 2. 간단한 예측 모델 (5분, 15분, 종가)
#############################################
def predict_horizon(df, horizon_min: int = 5):
    """
    horizon_min 분 뒤의 가격 변화를 예측하는 간단한 baseline 모델
    모델: 최근 N분 평균 기울기 사용
    """
    if df.empty:
        return None

    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["slope"] = df["Close"].diff()

    window = horizon_min  # 매우 단순 baseline
    df[f"pred_{horizon_min}m"] = df["slope"].rolling(window).mean()

    return df


#############################################
# 3. 예측 결과 생성
#############################################
def generate_prediction(df, horizon_list=[5, 15]):
    """
    여러 horizon 한번에 수행
    """
    result = {}
    for h in horizon_list:
        temp = predict_horizon(df, horizon_min=h)
        result[h] = temp
    return result


#############################################
# 4. 과거 예측 정확도 검증
#############################################
def backtest_prediction(df, horizon=5):
    """
    (5번탭용) 과거 예측이 실제로 맞았는지 검증
    """
    df = df.copy()
    df["future"] = df["Close"].shift(-horizon)
    df["future_ret"] = df["future"].pct_change()

    # 예측값
    df["pred"] = df["Close"].diff().rolling(horizon).mean()

    # 방향 정확도
    df["pred_dir"] = np.sign(df["pred"])
    df["real_dir"] = np.sign(df["future"] - df["Close"])

    accuracy = (df["pred_dir"] == df["real_dir"]).mean()
    return accuracy, df


#############################################
# 5. 전체 파이프라인 (4번탭 일괄 실행)
#############################################
def run_full_pipeline(ticker="SPY", days=10):
    df = fetch_1min_intraday(ticker, days=days)
    if df.empty:
        return None

    pred = generate_prediction(df, horizon_list=[5, 15])
    return {
        "raw": df,
        "predictions": pred,
    }
