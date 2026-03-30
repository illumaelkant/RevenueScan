from pathlib import Path

import pandas as pd


def _clip_outliers(group, column):
    ### Lay 1% nguong duoi va 99% nguong tren, tranh outlier qua khac biet
    ## gay anh huong den mo hinh ARMIA
    q_low = group[column].quantile(0.01)
    q_high = group[column].quantile(0.99)
    return group[column].clip(lower=q_low, upper=q_high)


def load_and_preprocess(raw_csv,processed_csv):
    df = pd.read_csv(raw_csv)
    ## change to datetime type
    df["date"] = pd.to_datetime(df["date"])

    numeric_cols = [
        "customer_count",
        "avg_ticket_vnd",
        "raw_material_kg",
        "revenue_vnd",
        "rain_index",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ## sort by restaurant_id and date
    df = df.sort_values(["restaurant_id", "date"]).reset_index(drop=True)

    ## fill NaN bang noi suy
    # neu van thieu thi thay bang gia tri phia sau va gia tri phia truoc
    for col in numeric_cols:
        df[col] = (
            df.groupby("restaurant_id", group_keys=False)[col]
            .apply(lambda s: s.interpolate(method="linear").bfill().ffill())
            .reset_index(level=0, drop=True)
        )

    ## cut outlier
    for target in ["revenue_vnd", "raw_material_kg"]:
        df[target] = df.groupby("restaurant_id", group_keys=False).apply(
            lambda g: _clip_outliers(g, target)
        )

    ## them cac cot thoi gian (freature engineering)
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)


    out_path = Path(processed_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
