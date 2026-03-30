import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .evaluation import evaluate_predictions


warnings.filterwarnings("ignore")


def _fit_sarima(series, order, seasonal_order):
    ## series la mot chuoi thoi giian
    ## order: (p,d,q), seasonal_order: (P,D,Q,m)
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def train_and_forecast_all(data,order=(2, 1, 2),seasonal_order=(1, 0, 1, 7),test_days=30,forecast_horizon=30):
    ## Cac quan an do co bien dong doanh thu va nguyn lieu rat manh, va co tuan theo hang tuan
    ## do vay, sarima cho performance tot hon arima, nhung ton thoi gian train hon    
    metrics_rows = []
    test_pred_rows = []
    future_rows = []
    ## du doan 2 target chinh la doanh thu va luong nguyen lieu
    targets = ["revenue_vnd", "raw_material_kg"]

    ## loop qua tung nha hang va tung target de train SARIMA va forecast
    for (restaurant_id, restaurant_name), store_df in data.groupby(["restaurant_id", "restaurant_name"], sort=True):
        store_df = store_df.sort_values("date").copy()
        store_df = store_df.set_index("date")

        for target in targets:
            ### cat train va test dataset
            series = store_df[target].asfreq("D")
            train_series = series.iloc[:-test_days]
            test_series = series.iloc[-test_days:]

            ### train model
            model = _fit_sarima(train_series, order, seasonal_order)
            test_forecast = model.get_forecast(steps=test_days).predicted_mean

            mae, rmse = evaluate_predictions(y_true=test_series.values, y_pred=test_forecast.values)

            ### them metric vao list
            metrics_rows.append(
                {
                    "restaurant_id": restaurant_id,
                    "restaurant_name": restaurant_name,
                    "target": target,
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "train_size": len(train_series),
                    "test_size": len(test_series),
                    "mae": round(mae, 2),
                    "rmse": round(rmse, 2),
                }
            )

            ## tao predict set cho tung cua hang de visualize sau nay
            for dt, actual, pred in zip(test_series.index, test_series.values, test_forecast.values):
                test_pred_rows.append(
                    {
                        "restaurant_id": restaurant_id,
                        "restaurant_name": restaurant_name,
                        "target": target,
                        "date": dt,
                        "actual_value": round(float(actual), 2),
                        "predicted_value": round(float(pred), 2),
                    }
                )

            ## train model tren toan bo tap du lieu
            full_model = _fit_sarima(series, order, seasonal_order)
            ## du doan tuong lai
            future_pred = full_model.get_forecast(steps=forecast_horizon).predicted_mean
            ## tao ra cac ngay trong tuong lai
            future_dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")

            ## luu ket qua forecast
            for dt, val in zip(future_dates, future_pred.values):
                future_rows.append(
                    {
                        "restaurant_id": restaurant_id,
                        "restaurant_name": restaurant_name,
                        "target": target,
                        "date": dt,
                        "forecast_value": round(float(val), 2),
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    test_pred_df = pd.DataFrame(test_pred_rows)
    future_df = pd.DataFrame(future_rows)
    return metrics_df, test_pred_df, future_df
