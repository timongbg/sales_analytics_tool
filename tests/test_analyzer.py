"""Unit tests for analyzer.py (no external files needed, uses in-memory data)."""

import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyzer import (
    build_daily_revenue,
    forecast_revenue,
    load_data,
    build_report,
    REQUIRED_COLUMNS,
)


def make_sample_df(n_days=40):
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for i, date in enumerate(dates):
        rows.append({
            "date": date,
            "category": "Electronics" if i % 2 == 0 else "Books",
            "region": "North",
            "units_sold": 2,
            "revenue": 50 + i * 1.5,
        })
    return pd.DataFrame(rows)


def test_load_data_validates_columns():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("date,category,region,units_sold\n2025-01-01,Books,North,1\n")
        path = f.name

    try:
        with pytest.raises(ValueError):
            load_data(path)
    finally:
        os.unlink(path)


def test_load_data_valid_file():
    df = make_sample_df()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name

    try:
        loaded = load_data(path)
        assert REQUIRED_COLUMNS.issubset(set(loaded.columns))
        assert len(loaded) == len(df)
    finally:
        os.unlink(path)


def test_build_daily_revenue_aggregates_correctly():
    df = make_sample_df(n_days=5)
    daily = build_daily_revenue(df)
    assert len(daily) == 5
    assert "revenue" in daily.columns
    assert daily["revenue"].sum() == pytest.approx(df["revenue"].sum())


def test_forecast_revenue_returns_expected_shape():
    df = make_sample_df(n_days=60)
    daily = build_daily_revenue(df)
    forecast = forecast_revenue(daily, forecast_days=10)

    assert len(forecast["future_dates"]) == 10
    assert len(forecast["future_preds"]) == 10
    # Revenue trends upward in the synthetic data, forecast should continue that trend
    assert forecast["future_preds"][-1] > forecast["historical"]["revenue"].iloc[0]


def test_build_report_creates_html_file():
    df = make_sample_df(n_days=40)
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "test_report.html")
        build_report(df, forecast_days=7, output_path=output_path)

        assert os.path.exists(output_path)
        content = open(output_path, encoding="utf-8").read()
        assert "Sales Analytics Report" in content
        assert "base64" in content  # charts embedded
