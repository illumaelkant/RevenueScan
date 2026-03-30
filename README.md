# REVENUESCAN – TIME-SERIES FORECASTING SYSTEM

## Introduction
Welcome to **RevenueScan**! This project is a comprehensive time-series forecasting system specifically designed for Small and Medium Enterprises (SMEs) in the Food & Beverage (F&B) industry.

As the Technical Leader, I architected and built this system from the ground up to empower F&B businesses to anticipate customer demand, optimize their inventory, and make data-driven decisions.

## Key Features & Contributions
* **Advanced Machine Learning Architecture**: Structured the backend using modular for Data Generation, Preprocessing, Training, Evaluation, and Visualization into distinct functional components.
* **ARIMA-based Revenue Forecasting**: Built the core revenue/demand forecasting engine utilizing robust Statistical Autoregressive Integrated Moving Average (ARIMA) models.
* **Comprehensive ML Pipeline**: 
  * **Data Preprocessing** (`src/preprocessing.py`): Time-series indexing, interpolation, and 80/20 train/test splits.
  * **Training** (`src/training.py`): Fitting ARIMA models to historical revenue data.
  * **Evaluation & Insight Extraction** (`src/evaluation.py`): Employs MAE and RMSE metrics to rigorously score models. Orchestrator script outputs specific inventory decisions based on demand.
  * **Automated Visualization** (`src/visualization.py`): Plots are programmatically saved to the `/plots/` directory to track actual vs predicted volumes.

## Machine Learning Backend (Python Pipeline)

The core intelligent engine operates entirely within the `Models/` folder.

### Repository Structure
```
Models/
├── data/                           # Simulated CSV datasets
├── plots/                          # Saved prediction charts (PNGs)
├── src/                            # Modularized pipeline scripts
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── visualization.py
├── main.py                         # Master pipeline execution script
└── requirements.txt
```

### Running the Forecasting Model:
1. Navigate to the model folder: `cd Models`
2. Install dependencies: `pip install -r requirements.txt`
3. Execute the pipeline for all target restaurants: `python main.py`
---

## Frontend Web Application (React App)

A user-friendly dashboard provides Sales Analytics and allows managers to view various sales metrics and insights.

### To run the dashboard locally, follow these steps:
1. Navigate to the frontend directory: `cd revenuescan`
2. Install dependencies: `npm install`
3. Start the development server: `npm start`
4. Open your browser and access the application at [http://localhost:3000](http://localhost:3000)

## License
This project is licensed under the MIT License.
