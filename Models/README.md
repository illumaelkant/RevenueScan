# RevenueScan Models - ARIMA Forecasting for F&B SMEs

This folder is a complete personal project for forecasting:
- Daily revenue
- Daily raw material demand

Scope:
- 10 simulated F&B stores
- 365 days of realistic synthetic data
- ARIMA forecasting pipeline with MAE and RMSE evaluation

## Project Structure

```text
Models/
  main.py
  requirements.txt
  README.md
  data/
    raw/
      fnb_synthetic_10_stores_1y.csv
    processed/
      fnb_daily_clean.csv
  output/
    forecasts/
      test_predictions.csv
      forecast_next_30_days.csv
      inventory_plan_next_30_days.csv
      business_decision_summary.csv
    metrics/
      metrics_by_store_target.csv
      metrics_overall.csv
    plots/
      *.png
  src/
    __init__.py
    data_generator.py
    preprocessing.py
    training.py
    evaluation.py
    visualization.py
```

## CV Mapping

- Built a revenue forecasting system for F&B SMEs using ARIMA models.
- Designed full ML pipeline: data preprocessing, training, and evaluation (MAE, RMSE).
- Provided demand forecasts to support inventory planning and business decisions.

## How To Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the full pipeline:

```bash
python main.py
```

## What Recruiters Can Check Quickly

- Final metrics: `output/metrics/metrics_by_store_target.csv`
- Overall average performance: `output/metrics/metrics_overall.csv`
- 30-day forecast output: `output/forecasts/forecast_next_30_days.csv`
- Inventory recommendation: `output/forecasts/inventory_plan_next_30_days.csv`
- Executive summary for decisions: `output/forecasts/business_decision_summary.csv`
- Forecast charts: `output/plots/`
