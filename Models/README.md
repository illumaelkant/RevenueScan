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
## How To Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the full pipeline:

```bash
python main.py
```

