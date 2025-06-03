# Forecast from Jan 2025
####### Just ETS(A,A,A), ETS(A,A,N), SES for all skus

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error

# ── CONFIG ──────────────────────────────────────────────────────────────

SHEET_NAME = "SPROUTS"
ITEM_COL = "Item"
HIST_END = pd.Timestamp("2024-12-01")        # Train until Dec 2024
FORECAST_START = pd.Timestamp("2025-01-01")  # Forecast Jan–May 2025
SEASONAL_P = 12

# ── 1) LOAD FILE ────────────────────────────────────────────────────────
def generate_forecasts(forecast_file):
    print("\n=== Starting Forecast Generation ===")
    print("Forecast file:", forecast_file)

    # Load the first sheet automatically
    df = pd.read_excel(forecast_file, sheet_name= "SPROUTS", engine="openpyxl")
    print("\nOriginal columns in Excel:")
    for col in df.columns:
        print(f"- {col} ({type(col)})")
    
    df[ITEM_COL] = df[ITEM_COL].astype(str).str.strip().str.upper()

    # ── 2) COERCE DATETIME COLUMNS ──────────────────────────────────────────
    parsed_cols = []
    print("\nAttempting to parse dates...")
    for col in df.columns:
        # If column is already a Timestamp, keep as-is
        if isinstance(col, pd.Timestamp):
            print(f"Column {col} is already a Timestamp")
            parsed_cols.append(col)
            continue
            
        # Try parsing string like "Jan-25" or "Jan 2025"
        try:
            # First try MMM-YY format
            parsed = pd.to_datetime(col, format="%b-%y", errors="coerce")
            if not pd.isna(parsed):
                print(f"Successfully parsed {col} as MMM-YY format: {parsed}")
                parsed_cols.append(parsed)
                continue
                
            # Then try MMM YYYY format
            parsed = pd.to_datetime(col, format="%b %Y", errors="coerce")
            if not pd.isna(parsed):
                print(f"Successfully parsed {col} as MMM YYYY format: {parsed}")
                parsed_cols.append(parsed)
                continue
                
            # Finally try any other date format
            parsed = pd.to_datetime(col, errors="coerce")
            if not pd.isna(parsed):
                print(f"Successfully parsed {col} as generic date format: {parsed}")
                parsed_cols.append(parsed)
                continue
                
            print(f"Could not parse {col} as a date")
        except Exception as e:
            print(f"Error parsing {col}: {str(e)}")
            
        # If all parsing attempts failed, keep original column name
        parsed_cols.append(col)

    df.columns = parsed_cols

    print("\nFinal parsed columns:")
    for col in df.columns:
        print(f"- {col} ({type(col)})")

    # Get all date columns and sort them
    month_cols = sorted([col for col in df.columns if isinstance(col, pd.Timestamp)])
    print("\nDate columns found:", month_cols)
    
    # Filter historical columns (up to Dec 2024)
    hist_cols = [col for col in month_cols if col <= HIST_END]
    print("\nHistorical columns (<= Dec 2024):", hist_cols)
    
    # Filter forecast columns (Jan-May 2025)
    fcast_cols = []
    for col in month_cols:
        if (col.year == FORECAST_START.year and 
            col.month >= FORECAST_START.month and 
            col.month <= FORECAST_START.month + 4):
            fcast_cols.append(col)
    print("\nForecast columns (Jan-May 2025):", fcast_cols)

    if not hist_cols or not fcast_cols:
        print("\nERROR: Missing required date columns!")
        print(f"Historical columns found: {len(hist_cols)}")
        print(f"Forecast columns found: {len(fcast_cols)}")
        return {
            "status": "error",
            "message": "Missing historical or forecast columns in the Excel sheet"
        }
    
    # ── 3) TRANSFORM HISTORICAL DATA ────────────────────────────────────────
    hist_df = df.set_index(ITEM_COL)[hist_cols].T.astype(float)
    hist_df.index = pd.DatetimeIndex(hist_cols)
    hist_df.index.name = "month"

    # ── 4) EXTRACT PROVIDED FORECASTS ───────────────────────────────────────
    prov_index = pd.DatetimeIndex(fcast_cols)
    prov_fcasts = {
        sku: df.set_index(ITEM_COL).loc[sku, fcast_cols].values.astype(float)
        for sku in df[ITEM_COL]
    }

    # ── 5) TRY MULTIPLE MODELS AND SELECT BEST BY MAPE ──────────────────────
    results = []
    pred_fcasts = {}

    for sku in df[ITEM_COL]:
        if sku not in hist_df.columns:
            continue

        series = hist_df[sku].dropna()
        if series.index.max() < HIST_END:
            continue

        train = series.loc[:HIST_END]
        h = len(prov_index)
        if train.empty or h == 0:
            continue

        prov_vals = prov_fcasts.get(sku)
        if prov_vals is None or len(prov_vals) != h:
            continue

        # Candidate models
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

            except:
                continue

        if best_model is None:
            continue

        pred_fcasts[sku] = best_forecast
        results.append({
            "SKU": sku,
            "Model": best_model,
            "MAPE%": best_mape
        })

    # ── 6) DISPLAY RESULTS ───────────────────────────────────────────────────
    err_df = pd.DataFrame(results).set_index("SKU").sort_values("MAPE%", ascending=False)

    for sku in err_df.index:
        m = err_df.loc[sku, "MAPE%"]
        c = err_df.loc[sku, "Model"]
        print(f"{sku} | Model: {c} | MAPE: {m:.2f}%")

    # Save forecast + metrics to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"forecast_outputs/forecasts_{timestamp}.xlsx"

    forecast_data = []
    dates = pd.date_range(FORECAST_START, periods=5, freq='MS')
    for sku, values in pred_fcasts.items():
        for date, val in zip(dates, values):
            forecast_data.append({"SKU": sku, "Date": date, "Forecast": val})

    df_forecast = pd.DataFrame(forecast_data)
    df_metrics = err_df.reset_index()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_forecast.pivot(index="Date", columns="SKU", values="Forecast").to_excel(writer, sheet_name="Forecasts")
        df_metrics.to_excel(writer, sheet_name="Metrics", index=False)

    return {
        "status": "success",
        "forecast_file": output_path,
        "message": f"Forecasts generated for {len(pred_fcasts)} SKUs"
    }

    # ── 7) PLOT TOP 5 FORECASTS ──────────────────────────────────────────────
    # for sku in err_df.tail(5).index:
    #     plt.figure(figsize=(8, 3))
    #     plt.plot(hist_df.index, hist_df[sku], "C0-", label="History")
    #     plt.plot(prov_index, prov_fcasts[sku], "C1o-", label="Provided")
    #     plt.plot(prov_index, pred_fcasts[sku], "C2s--", label="Ours")
    #     m = err_df.loc[sku, "MAPE%"]
    #     c = err_df.loc[sku, "Model"]
    #     plt.title(f"{sku} | {c} | MAPE={m:.1f}%")
    #     plt.xticks(rotation=45)
    #     plt.ylabel("Units")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
