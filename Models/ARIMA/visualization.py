from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_forecast_plots(data,test_predictions,future_forecasts,save_dir):
    ## path configuration
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for (restaurant_id, restaurant_name), store_df in data.groupby(["restaurant_id", "restaurant_name"]):
        store_df = store_df.sort_values("date")

        ## ve va luu bieu do cho tung cua hang va tung target
        for target in ["revenue_vnd", "raw_material_kg"]:
            pred_df = test_predictions[
                (test_predictions["restaurant_id"] == restaurant_id)
                & (test_predictions["target"] == target)
            ].copy()
            pred_df["date"] = pd.to_datetime(pred_df["date"])

            future_df = future_forecasts[
                (future_forecasts["restaurant_id"] == restaurant_id)
                & (future_forecasts["target"] == target)
            ].copy()
            future_df["date"] = pd.to_datetime(future_df["date"])

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(store_df["date"], store_df[target], label="historical", color="#1f77b4", alpha=0.8)
            ax.plot(pred_df["date"], pred_df["predicted_value"], label="test prediction", color="#d62728")
            ax.plot(
                future_df["date"],
                future_df["forecast_value"],
                label="next 30 days forecast",
                color="#2ca02c",
                linestyle="--",
            )

            ax.set_title(f"{restaurant_name} - {target}")
            ax.set_xlabel("date")
            ax.set_ylabel(target)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend()
            fig.tight_layout()

            safe_target = target.replace("_", "-")
            outfile = save_path / f"{restaurant_id}_{safe_target}.png"
            fig.savefig(outfile, dpi=200)
            plt.close(fig)