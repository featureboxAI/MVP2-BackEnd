# # Forecast from Jan 2025
# ####### Just ETS(A,A,A), ETS(A,A,N), SES for all skus

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────

SHEET_NAME = "SPROUTS"
ITEM_COL = "Item"
HIST_END = pd.Timestamp("2024-12-01")        # Train until Dec 2024
FORECAST_START = pd.Timestamp("2025-01-01")  # Forecast Jan–May 2025
SEASONAL_P = 12

# # ── 1) LOAD FILE ────────────────────────────────────────────────────────

def generate_forecasts(forecast_file):
    print("\n=== Starting Forecast Generation ===")
    print("Forecast file:", forecast_file)
    df = pd.read_excel(forecast_file, sheet_name="SPROUTS", engine="openpyxl")
    df[ITEM_COL] = df[ITEM_COL].astype(str).str.strip().str.upper()
    # ── 2) PARSE COLUMNS ───────────────────────────────
    parsed_cols = []
    for col in df.columns:
        if isinstance(col, pd.Timestamp):
            parsed_cols.append(col)
            continue
        try:
            parsed = pd.to_datetime(col, format="%b-%y", errors="coerce")
            if not pd.isna(parsed):
                parsed_cols.append(parsed)
                continue
            parsed = pd.to_datetime(col, format="%b %Y", errors="coerce")
            if not pd.isna(parsed):
                parsed_cols.append(parsed)
                continue
            parsed = pd.to_datetime(col, errors="coerce")
            if not pd.isna(parsed):
                parsed_cols.append(parsed)
                continue
        except:
            pass
        parsed_cols.append(col)
    df.columns = parsed_cols
    # ── 3) IDENTIFY COLUMNS ─────────────────────────────
    month_cols = sorted([col for col in df.columns if isinstance(col, pd.Timestamp)])
    hist_cols = [col for col in month_cols if col <= HIST_END]
    fcast_cols = [col for col in month_cols if FORECAST_START <= col <= FORECAST_START + pd.DateOffset(months=4)]
    print("\n📅 Forecast months:", fcast_cols)
    if not hist_cols or not fcast_cols:
        return {
            "status": "error",
            "message": "Missing historical or forecast columns in the Excel sheet"
        }
    # ── 4) FORMAT HISTORICAL ─────────────────────────────
    hist_df = df.set_index(ITEM_COL)[hist_cols].T.apply(pd.to_numeric, errors='coerce')
    hist_df.index = pd.DatetimeIndex(hist_cols)
    hist_df.index.name = "month"
    prov_index = pd.DatetimeIndex(fcast_cols)
    prov_fcasts = {
        sku: df.set_index(ITEM_COL).loc[sku, fcast_cols].values.astype(float)
        for sku in df[ITEM_COL]
    }
    results = []
    pred_fcasts = {}
    for sku in df[ITEM_COL]:
        # print(f"\n🔄 Checking SKU: {sku}")
        if sku not in hist_df.columns:
            print(f" {sku} not found in hist_df.columns")
            continue
        # print(f"{sku} - Series shape: {hist_df[sku].shape}, NaNs: {hist_df[sku].isna().sum()}")
        # print(f"{sku} - Last data point: {hist_df[sku].dropna().index.max()}")
        series = hist_df[sku].dropna()
        if series.index.max() < HIST_END:
            print(f" Skipping {sku}: last entry {series.index.max()} < HIST_END {HIST_END}")
            continue
        train = series.loc[:HIST_END]
        train.index = pd.date_range(start=train.index[0], periods=len(train), freq="MS")
        h = len(prov_index)
        if train.empty or h == 0:
            continue
        prov_vals = prov_fcasts.get(sku)
        if prov_vals is None or len(prov_vals) != h:
            continue
        # ── MODELS ─────────────────────
        candidates = {
            "ETS(A,A,A)": {"trend": "add", "seasonal": "add"},
            "ETS(A,A,N)": {"trend": "add", "seasonal": None},
            "SES":        {"trend": None, "seasonal": None}
        }
        best_model = None
        best_forecast = None
        best_mape = float("inf")
        for name, params in candidates.items():
            try:
                if params["seasonal"] and len(train) < SEASONAL_P * 2:
                    continue
                model = ExponentialSmoothing(
                    train,
                    trend=params["trend"],
                    seasonal=params["seasonal"],
                    seasonal_periods=SEASONAL_P if params["seasonal"] else None,
                    freq="MS"
                ).fit(optimized=True)
                forecast = model.forecast(h).values
                mape = mean_absolute_percentage_error(
                    np.nan_to_num(prov_vals),
                    np.nan_to_num(forecast)
                ) * 100
                if mape < best_mape:
                    best_model = name
                    best_mape = mape
                    best_forecast = forecast
            except Exception as e:
                print(f" Model {name} failed: {str(e)}")
                continue
        if best_model:
            pred_fcasts[sku] = best_forecast
            results.append({
                "SKU": sku,
                "Model": best_model,
                "MAPE%": best_mape
            })
    if not results:
        print(" No models selected. Returning error.")
        return {
            "status": "error",
            "message": "No SKUs could be forecasted due to missing or invalid data."
        }
    err_df = pd.DataFrame(results).set_index("SKU").sort_values("MAPE%", ascending=False)

    # ── DISPLAY FINAL MODEL SELECTION ───────────────────────
    print("\n Final model selection:")
    print(err_df)


    # ── EXPORT ─────────────────────
    downloads_dir = Path.home() / "Downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = downloads_dir / f"forecasts_{timestamp}.xlsx"

    forecast_data = []
    dates = pd.date_range(FORECAST_START, periods=5, freq='MS')
    for sku, values in pred_fcasts.items():
        for date, val in zip(dates, values):
            forecast_data.append({"SKU": sku, "Date": date, "Forecast": val})

    df_forecast = pd.DataFrame(forecast_data)
    df_metrics = err_df.reset_index()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_forecast.pivot(index="SKU", columns="Date", values="Forecast").to_excel(writer, sheet_name="Forecasts")
        df_metrics.to_excel(writer, sheet_name="Metrics", index=False)

    return {
        "status": "success",
        "forecast_file": str(output_path),
        "message": f"Forecasts generated for {len(pred_fcasts)} SKUs"
    }







    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_path = f"forecast_outputs/forecasts_{timestamp}.xlsx"
    # forecast_data = []
    # dates = pd.date_range(FORECAST_START, periods=5, freq='MS')
    # for sku, values in pred_fcasts.items():
    #     for date, val in zip(dates, values):
    #         forecast_data.append({"SKU": sku, "Date": date, "Forecast": val})
    # df_forecast = pd.DataFrame(forecast_data)
    # df_metrics = err_df.reset_index()
    # with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    #     df_forecast.pivot(index="SKU", columns="Date", values="Forecast").to_excel(writer, sheet_name="Forecasts")
    #     df_metrics.to_excel(writer, sheet_name="Metrics", index=False)
    # return {
    #     "status": "success",
    #     "forecast_file": output_path,
    #     "message": f"Forecasts generated for {len(pred_fcasts)} SKUs"
    # }
