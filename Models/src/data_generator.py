from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class StoreProfile:
    restaurant_id: str
    restaurant_name: str
    avg_daily_customers: int
    avg_ticket_vnd: int
    ingredient_kg_per_customer: float
    weekend_lift: float
    seasonal_sensitivity: float


def _store_profiles() -> list[StoreProfile]:
    return [
        StoreProfile("S01", "Pho Sunrise", 180, 85000, 0.42, 1.10, 0.35),
        StoreProfile("S02", "Saigon Com Tam", 220, 70000, 0.38, 1.12, 0.30),
        StoreProfile("S03", "Central Bun Bo", 160, 90000, 0.40, 1.08, 0.32),
        StoreProfile("S04", "Lotus Noodle House", 210, 78000, 0.36, 1.15, 0.28),
        StoreProfile("S05", "Mekong Grill", 145, 145000, 0.58, 1.25, 0.55),
        StoreProfile("S06", "Riverbank Seafood", 130, 190000, 0.72, 1.35, 0.62),
        StoreProfile("S07", "Green Bowl Vegan", 155, 98000, 0.34, 1.09, 0.27),
        StoreProfile("S08", "Hanoi Bun Cha", 200, 92000, 0.44, 1.18, 0.41),
        StoreProfile("S09", "Street Eats Hub", 240, 67000, 0.33, 1.22, 0.33),
        StoreProfile("S10", "Family Rice Kitchen", 190, 76000, 0.39, 1.14, 0.29),
    ]


def generate_synthetic_dataset(
    output_csv: str | Path,
    start_date: str = "2025-01-01",
    days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=days, freq="D")

    holiday_dates = {
        "2025-01-01",
        "2025-01-29",
        "2025-01-30",
        "2025-04-30",
        "2025-05-01",
        "2025-09-02",
    }

    rows = []
    for store in _store_profiles():
        for i, date in enumerate(dates):
            dow = date.dayofweek
            is_weekend = int(dow >= 5)
            is_holiday = int(date.strftime("%Y-%m-%d") in holiday_dates)
            promo_flag = int(rng.random() < 0.12)

            yearly_wave = 1.0 + store.seasonal_sensitivity * np.sin(2.0 * np.pi * i / 365.0)
            trend = 1.0 + 0.0007 * i
            weekend_factor = store.weekend_lift if is_weekend else 1.0
            holiday_factor = 1.35 if is_holiday else 1.0
            promo_factor = 1.18 if promo_flag else 1.0

            rain_index = float(np.clip(rng.normal(0.38, 0.2), 0.0, 1.0))
            weather_factor = 1.0 - (0.20 * rain_index)

            customer_noise = rng.normal(0, store.avg_daily_customers * 0.07)
            customers = (
                store.avg_daily_customers
                * yearly_wave
                * trend
                * weekend_factor
                * holiday_factor
                * promo_factor
                * weather_factor
            ) + customer_noise
            customers = int(max(20, round(customers)))

            ticket_noise = rng.normal(0, store.avg_ticket_vnd * 0.04)
            avg_ticket = max(35000, store.avg_ticket_vnd + ticket_noise)
            revenue_vnd = float(customers * avg_ticket)

            ingredient_noise = rng.normal(0, store.ingredient_kg_per_customer * 0.06)
            ingredient_per_customer = max(0.15, store.ingredient_kg_per_customer + ingredient_noise)
            raw_material_kg = float(customers * ingredient_per_customer)

            rows.append(
                {
                    "date": date,
                    "restaurant_id": store.restaurant_id,
                    "restaurant_name": store.restaurant_name,
                    "day_of_week": dow,
                    "is_weekend": is_weekend,
                    "is_holiday": is_holiday,
                    "promo_flag": promo_flag,
                    "rain_index": round(rain_index, 3),
                    "customer_count": customers,
                    "avg_ticket_vnd": round(avg_ticket, 0),
                    "raw_material_kg": round(raw_material_kg, 2),
                    "revenue_vnd": round(revenue_vnd, 0),
                }
            )

    df = pd.DataFrame(rows).sort_values(["restaurant_id", "date"]).reset_index(drop=True)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
