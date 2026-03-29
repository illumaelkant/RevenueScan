from pathlib import Path

import pandas as pd

from src.data_generator import generate_synthetic_dataset
from src.preprocessing import load_and_preprocess
from src.training import train_and_forecast_all
from src.visualization import save_forecast_plots


def build_inventory_plan(forecast_df: pd.DataFrame) -> pd.DataFrame:
    material_fc = forecast_df[forecast_df["target"] == "raw_material_kg"].copy()
    material_fc["safety_stock_kg"] = material_fc["forecast_value"].mul(0.15).round(2)
    material_fc["recommended_purchase_kg"] = (
        material_fc["forecast_value"] + material_fc["safety_stock_kg"]
    ).round(2)
    return material_fc[
        [
            "restaurant_id",
            "restaurant_name",
            "date",
            "forecast_value",
            "safety_stock_kg",
            "recommended_purchase_kg",
        ]
    ]


def build_business_summary(forecast_df: pd.DataFrame) -> pd.DataFrame:
    revenue_fc = (
        forecast_df[forecast_df["target"] == "revenue_vnd"]
        .groupby(["restaurant_id", "restaurant_name"], as_index=False)["forecast_value"]
        .sum()
        .rename(columns={"forecast_value": "forecast_revenue_30d_vnd"})
    )

    material_fc = (
        forecast_df[forecast_df["target"] == "raw_material_kg"]
        .groupby(["restaurant_id", "restaurant_name"], as_index=False)["forecast_value"]
        .sum()
        .rename(columns={"forecast_value": "forecast_material_30d_kg"})
    )

    summary = revenue_fc.merge(material_fc, on=["restaurant_id", "restaurant_name"], how="inner")
    summary["expected_daily_revenue_vnd"] = (
        summary["forecast_revenue_30d_vnd"] / 30.0
    ).round(0)
    summary["expected_daily_material_kg"] = (
        summary["forecast_material_30d_kg"] / 30.0
    ).round(2)
    return summary.sort_values("forecast_revenue_30d_vnd", ascending=False)


def main() -> None:
    root = Path(__file__).resolve().parent

    raw_data_path = root / "data" / "raw" / "fnb_synthetic_10_stores_1y.csv"
    processed_data_path = root / "data" / "processed" / "fnb_daily_clean.csv"

    output_forecasts = root / "output" / "forecasts"
    output_metrics = root / "output" / "metrics"
    output_plots = root / "output" / "plots"

    for p in [output_forecasts, output_metrics, output_plots]:
        p.mkdir(parents=True, exist_ok=True)

    print("[1/5] Generating synthetic data for 10 F&B stores (1 year)...")
    generate_synthetic_dataset(output_csv=raw_data_path, start_date="2025-01-01", days=365, seed=42)

    print("[2/5] Running preprocessing...")
    clean_df = load_and_preprocess(raw_csv=raw_data_path, processed_csv=processed_data_path)

    print("[3/5] Training ARIMA and forecasting targets...")
    metrics_df, test_pred_df, future_forecast_df = train_and_forecast_all(
        data=clean_df,
        arima_order=(2, 1, 2),
        test_days=30,
        forecast_horizon=30,
    )

    print("[4/5] Exporting metrics and forecast outputs...")
    metrics_path = output_metrics / "metrics_by_store_target.csv"
    metrics_df.to_csv(metrics_path, index=False)

    metrics_overall = (
        metrics_df.groupby("target", as_index=False)[["mae", "rmse"]]
        .mean()
        .rename(columns={"mae": "avg_mae", "rmse": "avg_rmse"})
    )
    metrics_overall.to_csv(output_metrics / "metrics_overall.csv", index=False)

    test_pred_df.to_csv(output_forecasts / "test_predictions.csv", index=False)
    future_forecast_df.to_csv(output_forecasts / "forecast_next_30_days.csv", index=False)

    inventory_plan = build_inventory_plan(future_forecast_df)
    inventory_plan.to_csv(output_forecasts / "inventory_plan_next_30_days.csv", index=False)

    business_summary = build_business_summary(future_forecast_df)
    business_summary.to_csv(output_forecasts / "business_decision_summary.csv", index=False)

    print("[5/5] Creating visual reports...")
    save_forecast_plots(
        data=clean_df,
        test_predictions=test_pred_df,
        future_forecasts=future_forecast_df,
        save_dir=output_plots,
    )

    print("\nPipeline completed successfully.")
    print(f"- Metrics: {output_metrics}")
    print(f"- Forecasts: {output_forecasts}")
    print(f"- Plots: {output_plots}")


if __name__ == "__main__":
    main()
