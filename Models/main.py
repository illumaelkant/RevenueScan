from pathlib import Path

import pandas as pd

from ARIMA.preprocessing import load_and_preprocess
from ARIMA.training import train_and_forecast_all
from ARIMA.visualization import save_forecast_plots
from SARIMA.training import train_and_forecast_all as train_and_forecast_all_sarima


def main():
    root = Path(__file__).resolve().parent
    USE_SARIMA = True  ## co dung sarima, chay va ghi de ket qua phan arima len output

    ## Folder configuration
    raw_data_path = root/"data"/"raw"/"fnb_synthetic_10_stores_1y.csv"
    processed_data_path = root/"data"/"processed"/"fnb_daily_clean.csv"

    output_forecasts = root/"output"/"forecasts"
    output_metrics = root/"output"/"metrics"
    output_plots = root / "output" / "plots"


    for p in [output_forecasts, output_metrics, output_plots]:
        p.mkdir(parents=True, exist_ok=True)

    ## Running the pipeline
    print("Running preprocessing")
    clean_df = load_and_preprocess(raw_data_path,processed_data_path)

    if USE_SARIMA == False:
        print("Training SARIMA...")
        metrics_df, test_pred_df, future_forecast_df = train_and_forecast_all_sarima(
            data=clean_df,
            ## for best results, da thu va param tot nhat la (2,1,2) va (1,0,1,7) cho du lieu tuan cua quan
            order=(2, 1, 2),
            seasonal_order=(1, 0, 1, 7),
            test_days=30,
            forecast_horizon=30,
        )
    else:
        print("Training ARIMA...")
        metrics_df, test_pred_df, future_forecast_df = train_and_forecast_all(
            data=clean_df,
            ## change param here for best results
            arima_order=(2, 1, 2),
            ## test day is 30
            test_days=30,
            forecast_horizon=30,
        )

    ## exporting metrics
    print("[Exporting metrics and forecast outputs...")
    metrics_path = root/"output"/"metrics"/"metrics_by_store_target.csv"
    metrics_df.to_csv(metrics_path, index=False)

    metrics_overall = (
        metrics_df.groupby("target", as_index=False)[["mae", "rmse"]]
        .mean()
        .rename(columns={"mae": "avg_mae", "rmse": "avg_rmse"})
    )
    metrics_overall.to_csv(root/"output"/"metrics""metrics_overall.csv", index=False)

    test_pred_df.to_csv(root/"output"/"forecasts"/"test_predictions.csv", index=False)
    future_forecast_df.to_csv(root/"output"/"forecasts"/"forecast_next_30_days.csv", index=False)

    ### visualize plot
    save_forecast_plots(
        data=clean_df,
        test_predictions=test_pred_df,
        future_forecasts=future_forecast_df,
        save_dir=output_plots,
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()