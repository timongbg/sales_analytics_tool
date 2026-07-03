"""
Sales Analytics & Forecasting Tool
-----------------------------------
Loads a sales CSV (date, category, region, units_sold, revenue), runs exploratory
analysis, trains a simple linear trend forecast for future revenue, and generates
a self-contained HTML report with embedded charts.

Usage:
    python analyzer.py --input sample_data/sales_data.csv --output report.html
    python analyzer.py --input sample_data/sales_data.csv --forecast-days 30
"""

import argparse
import base64
import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, just save to buffer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"date", "category", "region", "units_sold", "revenue"}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load and validate the sales CSV."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    return df


def fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("date", as_index=False)["revenue"].sum().sort_values("date")
    return daily


def chart_revenue_over_time(daily: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(daily["date"], daily["revenue"], color="#2F5496", linewidth=1.2)
    ax.set_title("Daily Revenue Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    fig.autofmt_xdate()
    return fig_to_base64(fig)


def chart_revenue_by_category(df: pd.DataFrame) -> str:
    by_cat = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(by_cat.index, by_cat.values, color="#4472C4")
    ax.set_title("Total Revenue by Category")
    ax.set_ylabel("Revenue")
    plt.xticks(rotation=30, ha="right")
    return fig_to_base64(fig)


def chart_revenue_by_region(df: pd.DataFrame) -> str:
    by_region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(by_region.values, labels=by_region.index, autopct="%1.0f%%", startangle=90)
    ax.set_title("Revenue Share by Region")
    return fig_to_base64(fig)


def forecast_revenue(daily: pd.DataFrame, forecast_days: int = 30) -> dict:
    """
    Simple linear-trend forecast on daily revenue.
    Not a substitute for a production time-series model (e.g. Prophet/ARIMA),
    but a transparent, fast baseline that's easy to explain to non-technical clients.
    """
    daily = daily.copy()
    daily["day_index"] = np.arange(len(daily))

    X = daily[["day_index"]].values
    y = daily["revenue"].values

    split = int(len(daily) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    mae = None
    if len(X_test) > 0:
        preds_test = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds_test)

    # Refit on all data for the actual future forecast
    model.fit(X, y)
    future_index = np.arange(len(daily), len(daily) + forecast_days).reshape(-1, 1)
    future_preds = model.predict(future_index)
    future_dates = pd.date_range(
        start=daily["date"].max() + pd.Timedelta(days=1), periods=forecast_days
    )

    return {
        "model": model,
        "mae": mae,
        "future_dates": future_dates,
        "future_preds": future_preds,
        "historical": daily,
    }


def chart_forecast(forecast: dict) -> str:
    daily = forecast["historical"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(daily["date"], daily["revenue"], color="#2F5496", linewidth=1, label="Historical")
    ax.plot(
        forecast["future_dates"],
        forecast["future_preds"],
        color="#ED7D31",
        linewidth=1.5,
        linestyle="--",
        label="Forecast",
    )
    ax.set_title("Revenue Forecast (Linear Trend Baseline)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend()
    fig.autofmt_xdate()
    return fig_to_base64(fig)


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sales Analytics Report</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; color: #1a1a1a; background: #fafafa; }
  h1 { color: #2F5496; }
  h2 { color: #2F5496; border-bottom: 2px solid #eee; padding-bottom: 6px; margin-top: 40px; }
  .kpi-row { display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }
  .kpi-box { background: white; border-radius: 8px; padding: 18px 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); min-width: 160px; }
  .kpi-value { font-size: 26px; font-weight: 700; color: #2F5496; }
  .kpi-label { font-size: 13px; color: #666; margin-top: 4px; }
  img { max-width: 100%; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin: 12px 0; }
  .note { font-size: 13px; color: #777; font-style: italic; }
  footer { margin-top: 50px; font-size: 12px; color: #999; }
</style>
</head>
<body>
  <h1>📊 Sales Analytics Report</h1>
  <p class="note">Generated automatically from {{ n_rows }} transaction records
     covering {{ date_min }} to {{ date_max }}.</p>

  <div class="kpi-row">
    <div class="kpi-box">
      <div class="kpi-value">{{ total_revenue }}</div>
      <div class="kpi-label">Total Revenue</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-value">{{ total_units }}</div>
      <div class="kpi-label">Units Sold</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-value">{{ avg_daily_revenue }}</div>
      <div class="kpi-label">Avg. Daily Revenue</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-value">{{ top_category }}</div>
      <div class="kpi-label">Top Category</div>
    </div>
  </div>

  <h2>Revenue Over Time</h2>
  <img src="data:image/png;base64,{{ chart_time }}" alt="Revenue over time">

  <h2>Revenue by Category</h2>
  <img src="data:image/png;base64,{{ chart_category }}" alt="Revenue by category">

  <h2>Revenue by Region</h2>
  <img src="data:image/png;base64,{{ chart_region }}" alt="Revenue by region">

  <h2>{{ forecast_days }}-Day Forecast</h2>
  <img src="data:image/png;base64,{{ chart_forecast }}" alt="Revenue forecast">
  <p class="note">
    Baseline linear-trend model{% if mae %}, mean absolute error on the held-out
    validation split: {{ mae }}{% endif %}. For production use cases, this can be
    upgraded to a proper time-series model (e.g. Prophet, ARIMA) with the same
    report layout.
  </p>

  <footer>Report generated with the Sales Analytics & Forecasting Tool.</footer>
</body>
</html>
"""


def build_report(df: pd.DataFrame, forecast_days: int, output_path: str) -> None:
    daily = build_daily_revenue(df)

    chart_time = chart_revenue_over_time(daily)
    chart_category = chart_revenue_by_category(df)
    chart_region = chart_revenue_by_region(df)

    forecast = forecast_revenue(daily, forecast_days=forecast_days)
    chart_forecast_b64 = chart_forecast(forecast)

    top_category = df.groupby("category")["revenue"].sum().idxmax()

    template = Template(REPORT_TEMPLATE)
    html = template.render(
        n_rows=f"{len(df):,}",
        date_min=daily["date"].min().strftime("%Y-%m-%d"),
        date_max=daily["date"].max().strftime("%Y-%m-%d"),
        total_revenue=f"€{df['revenue'].sum():,.0f}",
        total_units=f"{df['units_sold'].sum():,}",
        avg_daily_revenue=f"€{daily['revenue'].mean():,.0f}",
        top_category=top_category,
        chart_time=chart_time,
        chart_category=chart_category,
        chart_region=chart_region,
        chart_forecast=chart_forecast_b64,
        forecast_days=forecast_days,
        mae=f"€{forecast['mae']:.2f}" if forecast["mae"] is not None else None,
    )

    Path(output_path).write_text(html, encoding="utf-8")
    logger.info("Report saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Analyze sales data and generate an HTML report.")
    parser.add_argument("--input", default="sample_data/sales_data.csv", help="Path to input CSV.")
    parser.add_argument("--output", default="report.html", help="Path to output HTML report.")
    parser.add_argument("--forecast-days", type=int, default=30, help="Number of days to forecast.")
    args = parser.parse_args()

    logger.info("Loading data from %s", args.input)
    df = load_data(args.input)

    logger.info("Building report...")
    build_report(df, args.forecast_days, args.output)


if __name__ == "__main__":
    main()
