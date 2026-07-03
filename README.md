# Sales Analytics & Forecasting Tool

A Python tool that turns raw sales data (CSV) into a polished, self-contained
HTML report — complete with KPIs, visualizations, and a baseline revenue
forecast. Built as a practical demonstration of an end-to-end data science
workflow: **data validation → exploratory analysis → visualization → forecasting → automated reporting.**

## Why this project

Turning messy transaction data into a clear, shareable report is one of the
most commonly requested freelance data tasks. This project shows the full
pipeline a client would actually pay for — not just a notebook, but a
finished deliverable.

## What it does

1. **Loads & validates** a sales CSV (`date, category, region, units_sold, revenue`)
2. **Computes KPIs**: total revenue, units sold, average daily revenue, top category
3. **Generates visualizations**: revenue over time, revenue by category, revenue by region
4. **Trains a baseline forecast** (linear trend) for the next N days, with a
   held-out validation MAE (mean absolute error) for transparency
5. **Outputs a single HTML file** — no server needed, easy to email or host anywhere

## Demo

```bash
$ python analyzer.py --input sample_data/sales_data.csv --forecast-days 30
15:42:32 [INFO] Loading data from sample_data/sales_data.csv
15:42:32 [INFO] Building report...
15:42:33 [INFO] Report saved to report.html
```

Open `report.html` in any browser to see the KPI dashboard, charts, and forecast.

## Installation

```bash
git clone https://github.com/timongbg/sales-analytics-tool.git
cd sales-analytics-tool
pip install -r requirements.txt
```

## Usage

```bash
# Generate the included synthetic sample dataset (optional, already included)
python generate_sample_data.py

# Run the analysis on the sample data
python analyzer.py --input sample_data/sales_data.csv --output report.html

# Use your own data (must have columns: date, category, region, units_sold, revenue)
python analyzer.py --input my_sales.csv --output my_report.html --forecast-days 60
```

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## Notes on the forecasting approach

The forecast uses a simple linear-trend model (`scikit-learn LinearRegression`)
on daily aggregated revenue. This is intentionally a transparent, fast baseline
— easy to explain to non-technical stakeholders — rather than a black-box model.
For production use cases with strong seasonality, this is designed to be a
drop-in point for a more advanced model (e.g. Prophet, SARIMA, or a gradient-boosted
model with engineered seasonal features) while keeping the same report layout.

## Tech stack

- Python 3.10+
- `pandas` / `numpy` — data handling
- `matplotlib` — visualization
- `scikit-learn` — forecasting model
- `jinja2` — HTML report templating
- `pytest` — testing

## License

MIT — feel free to fork and adapt.
