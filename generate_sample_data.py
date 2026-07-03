"""
Generates a realistic synthetic e-commerce sales dataset for demo/testing purposes.
Run this once to create sample_data/sales_data.csv, which analyzer.py can then process.

Usage:
    python generate_sample_data.py
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)

CATEGORIES = ["Electronics", "Home & Garden", "Clothing", "Sports", "Books"]
REGIONS = ["North", "South", "East", "West"]


def generate_sales_data(n_days: int = 365, start_date: str = "2025-01-01") -> pd.DataFrame:
    """Create a synthetic daily sales dataset with trend, weekly seasonality and noise."""
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    rows = []
    for date in dates:
        day_of_year = date.dayofyear
        # Upward trend over the year
        trend = day_of_year * 2.5
        # Weekly seasonality: weekends sell more
        weekend_boost = 150 if date.weekday() >= 5 else 0
        # A holiday-season boost in November/December
        season_boost = 300 if date.month in (11, 12) else 0

        n_orders_today = RNG.integers(15, 40)
        for _ in range(n_orders_today):
            category = RNG.choice(CATEGORIES, p=[0.3, 0.2, 0.25, 0.15, 0.1])
            region = RNG.choice(REGIONS)
            base_price = {
                "Electronics": 180,
                "Home & Garden": 60,
                "Clothing": 45,
                "Sports": 70,
                "Books": 18,
            }[category]

            noise = RNG.normal(0, base_price * 0.15)
            daily_factor = 1 + (trend + weekend_boost + season_boost) / 10000
            revenue = max(round((base_price + noise) * daily_factor, 2), 5.0)

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "category": category,
                "region": region,
                "units_sold": int(RNG.integers(1, 5)),
                "revenue": revenue,
            })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = generate_sales_data()
    df.to_csv("sample_data/sales_data.csv", index=False)
    print(f"Generated {len(df)} rows -> sample_data/sales_data.csv")
