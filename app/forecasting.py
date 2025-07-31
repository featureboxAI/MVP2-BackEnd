import pandas as pd
import numpy as np
import time
import multiprocessing as mp
import os
import warnings
import sys
import psutil
from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from pmdarima import auto_arima
from pykalman import KalmanFilter
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# -------------------------------
# Metrics & Helper Functions
# -------------------------------
def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100 * np.mean(np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom))

def nrmse(y_true, y_pred):
    return 100 * (np.sqrt(np.mean((y_true - y_pred)**2)) / (y_true.max() - y_true.min()))

# At this point:
#   non_exog_series holds every (sheet,item): orig_series
#   exog_series     holds only those for which consumption exists
# You can now pass non_exog_series into your blocks 4–5,
# and exog_series into your blocks 6–7.

#  Model wrappers for non-exogenous
# -----------------------------------------

def fit_holt_winters(train, test):
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
    fit = model.fit(optimized=True)
    pred = fit.forecast(len(test))
    return pred, nrmse(test, pred)

def fit_ets(train, test):
    model = ETSModel(train, error='add', trend='add', seasonal='add', seasonal_periods=12)
    fit = model.fit(disp=False)
    pred = fit.forecast(len(test))
    return pd.Series(pred, index=test.index), nrmse(test, pred)

def fit_pmdarima(train, test):
    train_clean, test_clean = train.dropna(), test.dropna()
    # Fallback for very short series
    if len(train_clean) < 3:
        fcast = np.repeat(train_clean.iloc[-1] if len(train_clean)>0 else 0, len(test))
        return pd.Series(fcast, index=test.index), np.inf
    
    # First attempt seasonal ARIMA with custom p/d/q ranges (try ACF and PACF methods: statsmodels statstools))
    try:
        model = auto_arima(
            train_clean,
            start_p=1, max_p=1,      #range of length of ACF values
            start_q=1, max_q=1,
            d=None, D=None,
            seasonal=False,        
            max_order=2,           
            stepwise=True,
            error_action='ignore',
            suppress_warnings=True,
            trace=False,
            n_jobs=1,
            information_criterion='bic'
        )
    except ValueError:
        last = train_clean.iloc[-1] if len(train_clean) > 0 else 0
        preds = np.repeat(last, len(test))
        return pd.Series(preds, index=test.index), np.inf

    vals = model.predict(n_periods=len(test_clean))
    pred = pd.Series(index=test.index, dtype=float)
    pred.loc[test_clean.index] = vals
    return pred, nrmse(test_clean, vals)

def fit_kalman(train, test):
    clean = train.dropna()
    if len(clean) < 2:
        last = clean.iloc[-1] if len(clean)>0 else 0
        return pd.Series(np.repeat(last, len(test)), index=test.index), np.inf
    kf = KalmanFilter(initial_state_mean=clean.iloc[0], n_dim_obs=1)
    states, _ = kf.em(clean.values).smooth(clean.values)
    last = states[-1]
    pred = pd.Series(np.repeat(last, len(test)), index=test.index)
    return pred, nrmse(test, pred)

def fit_prophet(train, test):
    df = pd.DataFrame({'ds': train.index, 'y': train.values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=len(test), freq='MS')
    forecast = model.predict(future)
    preds = forecast['yhat'].iloc[-len(test):].values
    return pd.Series(preds, index=test.index), smape(test, preds)

def fit_random_forest(X_train, y_train, X_test, test_index):
    # Grid search over specified RF hyperparameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_leaf': [1, 2, 4]
    }
    grid = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    pred = grid.predict(X_test)
    return pd.Series(pred, index=test_index), nrmse(y_train, y_train)

def fit_gradient_boost(X_train, y_train, X_test, test_index):
    # Grid search over specified GBM hyperparameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    }
    grid = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)
    pred = grid.predict(X_test) 
    return pd.Series(pred, index=test_index), nrmse(y_train, y_train)

def fit_lstm(series, train, test, look_back=6, epochs=100, batch_size=20, units=20, dropout=0.2):
    def create_dataset(data, lb):
        X, y = [], []
        for i in range(len(data) - lb):
            X.append(data[i:(i+lb)])
            y.append(data[i+lb])
        return np.array(X), np.array(y)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1,1))
    train_scaled = scaled[:len(train)]
    test_scaled = scaled[len(train)-look_back:]
    X_train, y_train = create_dataset(train_scaled, look_back)
    X_test, _ = create_dataset(test_scaled, look_back)
    model = Sequential([LSTM(units, input_shape=(look_back,1)), Dropout(dropout), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0,
              callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)])
    preds = model.predict(X_test).flatten()
    fcast = scaler.inverse_transform(preds.reshape(-1,1)).flatten()[-len(test):]
    return pd.Series(fcast, index=test.index), nrmse(test, fcast)

def fit_pmdarima_exog(train, test, exog_train, exog_test):
    """ARIMA with an external regressor."""
    # drop missing just in case
    y_train = train.dropna()
    ex_train = exog_train.loc[y_train.index]
    model = auto_arima(
        y=y_train,
        exogenous=ex_train,
        seasonal=True, m=12,
        error_action='ignore', suppress_warnings=True,
        stepwise=True
    )
    # forecast n steps with exogenous test
    preds = model.predict(n_periods=len(test), exogenous=exog_test)
    return pd.Series(preds, index=test.index), nrmse(test, preds)


def fit_prophet_exog(train, test, exog_train, exog_test):
    """Prophet with an external regressor."""
    # prepare history
    df_hist = pd.DataFrame({
        'ds': train.index,
        'y': train.values,
        'cons': exog_train.values
    }).dropna()
    model = Prophet()
    model.add_regressor('cons')
    model.fit(df_hist)

    # future frame
    future = pd.DataFrame({
        'ds': pd.date_range(start=test.index[0], periods=len(test), freq='MS'),
        'cons': exog_test.values
    })
    forecast = model.predict(future)
    preds = forecast['yhat'].values
    return pd.Series(preds, index=test.index), smape(test, preds)

# block 5: Non-Exogenous Models Driver
# --------------------------------------
def run_non_exog_driver(series_dict):
    df_metrics, df_forecasts = [], []
    keys = list(series_dict.keys())
    sheets = sorted({s for s, _ in keys})
    total_sheets = len(sheets)
    sheet_to_items = {s: [i for (s2, i) in keys if s2 == s] for s in sheets}

    for si, (sheet, item) in enumerate(keys, 1):
        ii = sheet_to_items[sheet].index(item) + 1
        print(f"[NON-EXOG] Sheet {si}/{total_sheets}: '{sheet}' — Item {ii}/{len(sheet_to_items[sheet])}: {item}")
        
        ts = series_dict[(sheet, item)]
        ts = ts["2018-01-01":"2025-05-01"]
        split = int(len(ts) * 0.8)
        train, test = ts.iloc[:split], ts.iloc[split:]

        # helper for logging and dual-error
        def log_run(name, fn, *args):
            proc = psutil.Process(os.getpid())
            ta, tu = os.cpu_count(), proc.num_threads()
            m0, t0 = proc.memory_info().rss, time.time()
            pr, _ = fn(*args)
            n_err = nrmse(test, pr)
            s_err = smape(test, pr)
            return pr, n_err, s_err

        # run each model on train/test
        preds, errs_nrmse, errs_smape = {}, {}, {}
        for name, fn in [
            ('hw', fit_holt_winters),
            ('ets', fit_ets),
            ('arima', fit_pmdarima),
            ('kf', fit_kalman),
            ('prophet', fit_prophet)
        ]:
            pr, n_err, s_err = log_run(name, fn, train, test)
            preds[name] = pr
            errs_nrmse[name] = n_err
            errs_smape[name] = s_err

        # ML & LSTM windowed models
        for lb in [6]:
            Xtr, ytr, Xte, idxs = [], [], [], []
            for k in range(lb, len(ts)):
                w, y = ts.values[k-lb:k], ts.values[k]
                if np.isnan(y) or np.isnan(w).any(): continue
                if k < split:
                    Xtr.append(w); ytr.append(y)
                else:
                    Xte.append(w); idxs.append(ts.index[k])
            if Xtr and Xte:
                Xtr_arr, ytr_arr, Xte_arr = map(np.array, (Xtr, ytr, Xte))
                for model_name, fn in [('rf', fit_random_forest), ('gb', fit_gradient_boost)]:
                    pr, n_err, s_err = log_run(f"{model_name}_{lb}", fn, Xtr_arr, ytr_arr, Xte_arr, idxs)
                    preds[f"{model_name}_{lb}"] = pr
                    errs_nrmse[f"{model_name}_{lb}"] = n_err
                    errs_smape[f"{model_name}_{lb}"] = s_err
                pr, n_err, s_err = log_run(f"lstm_{lb}", fit_lstm, ts, train, test, lb)
                preds[f"lstm_{lb}"] = pr
                errs_nrmse[f"lstm_{lb}"] = n_err
                errs_smape[f"lstm_{lb}"] = s_err

        # select best model among those supporting multi-step forecasting
        candidates = {
            m: errs_nrmse[m] for m in errs_nrmse
            if m in {'arima','hw','ets','prophet'} or m.startswith(('rf_','gb_','lstm_'))
        }
        # filter out zero or NaN errors
        valid = {m: e for m, e in candidates.items() if e > 0 and not np.isnan(e)}
        if valid:
            best = min(valid, key=valid.get)
        else:
            best = min(candidates, key=candidates.get)
        print(f"    ==> Best: {best} err={errs_nrmse[best]:.2f}%")

        
        # prepare future index
        fut_idx = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='MS')

        # full-series forecast
        if best == 'arima':
            m = auto_arima(ts, seasonal=True, m=12, stepwise=True, n_jobs=-1)
            fc = pd.Series(m.predict(n_periods=12), index=fut_idx)

        elif best == 'hw':
            fc = ExponentialSmoothing(
                ts, trend='add', seasonal='add', seasonal_periods=12
            ).fit(optimized=True).forecast(12)

        elif best == 'ets':
            m = ETSModel(
                ts, error='add', trend='add', seasonal='add', seasonal_periods=12
            ).fit(disp=False)
            fc = pd.Series(m.forecast(12), index=fut_idx)

        elif best == 'prophet':
            df_hist = pd.DataFrame({'ds': ts.index, 'y': ts.values}).dropna()
            m = Prophet().fit(df_hist)
            future = m.make_future_dataframe(periods=12, freq='MS')
            forecast = m.predict(future)
            fc = pd.Series(forecast['yhat'].iloc[-12:].values, index=future['ds'].iloc[-12:])

        elif best.startswith(('rf_','gb_','lstm_')):
            # determine lookback
            lb = int(best.split('_')[1])
            arr = ts.values

            # build full-history datasets
            def make_XY(arr, lb):
                X, y = [], []
                for i in range(lb, len(arr)):
                    X.append(arr[i-lb:i]); y.append(arr[i])
                return np.array(X), np.array(y)

            X_full, y_full = make_XY(arr, lb)

            # Drop any windows containing NaNs
            mask = (~np.isnan(X_full).any(axis=1)) & (~np.isnan(y_full))
            X_full, y_full = X_full[mask], y_full[mask]

            # fit the chosen model on full history
            if best.startswith('rf'):
                model = RandomForestRegressor(
                    n_estimators=100, max_depth=10
                ).fit(X_full, y_full)
                reshape = (1, lb)
            elif best.startswith('gb'):
                model = GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=5
                ).fit(X_full, y_full)
                reshape = (1, lb)
            else:  # LSTM
                X_full_l = X_full.reshape(-1, lb, 1)
                model = Sequential([
                    LSTM(50, input_shape=(lb,1)),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile('adam', 'mse')
                model.fit(X_full_l, y_full, epochs=100, batch_size=32, verbose=0)
                reshape = (1, lb, 1)

            # recursive multi-step
            history = arr.tolist()
            future_preds = []
            for _ in range(12):
                x = np.array(history[-lb:]).reshape(reshape)
                p = model.predict(x)[0]
                future_preds.append(p)
                history.append(p)
            fc = pd.Series(future_preds, index=fut_idx)

        else:
            # should not occur, but safe fallback
            last = ts.iloc[-1]
            fc = pd.Series(np.repeat(last, 12), index=fut_idx)

        # record metrics
        row = {'sheet': sheet, 'item': item, 'bucket': 'non_exog'}
        for m in preds:
            row[f'nrmse_{m}'] = errs_nrmse[m]
            row[f'smape_{m}'] = errs_smape[m]
        df_metrics.append(row)

        # record forecasts
        df_forecasts.append(pd.DataFrame({
            'sheet': sheet, 'item': item, 'bucket': 'non_exog',
            'model': best, 'ds': fc.index, 'forecast': fc.values
        }))

    # return pd.DataFrame(df_metrics), pd.concat(df_forecasts, ignore_index=True)
    return pd.DataFrame(df_metrics), pd.concat(df_forecasts, ignore_index=True),  None
    
def run_exog_driver(series_dict):
    df_metrics, df_forecasts = [], []

    
    keys = list(series_dict.keys())
    sheets = sorted({s for s, _ in keys})
    total_sheets = len(sheets)
    sheet_to_items = {s: [i for (s2, i) in keys if s2 == s] for s in sheets}
   

    for si, (sheet, item) in enumerate(keys, 1):
        ii = sheet_to_items[sheet].index(item) + 1
        print(f"[EXOG] Sheet {si}/{total_sheets}: '{sheet}' — Item {ii}/{len(sheet_to_items[sheet])}: {item}")
        orig_ts, cons_ts = series_dict[(sheet, item)]
        orig_ts = orig_ts["2018-01-01":"2025-05-01"].asfreq('MS')
        cons_ts = cons_ts.reindex(orig_ts.index).asfreq('MS').fillna(method='ffill')
        split = int(len(orig_ts) * 0.8)
        train, test = orig_ts.iloc[:split], orig_ts.iloc[split:]
        ex_train = cons_ts.iloc[:split].to_frame('cons')
        ex_test  = cons_ts.iloc[split:].to_frame('cons')

        # helper for logging exog models
        def log_exog(name, fn, *args):
            proc = psutil.Process(os.getpid())
            ta, tu = os.cpu_count(), proc.num_threads()
            m0, t0 = proc.memory_info().rss, time.time()
            pr, _ = fn(*args)
            n_err = nrmse(test, pr)
            s_err = smape(test, pr)
            t1, m1 = time.time(), proc.memory_info().rss
            print(f"    [{name:^12}] threads_avail={ta} threads_used={tu} "
                  f"time={(t1-t0):.1f}s mem={(m1-m0)/1e6:.1f}MB")
            return pr, n_err, s_err

        preds, errs_nrmse, errs_smape = {}, {}, {}
        for name, fn, args in [
            ('arima_exog', fit_pmdarima_exog, (train, test, ex_train, ex_test)),
            ('prophet_exog', fit_prophet_exog, (train, test, ex_train['cons'], ex_test['cons']))
        ]:
            pr, n_err, s_err = log_exog(name, fn, *args)
            preds[name] = pr
            errs_nrmse[name] = n_err
            errs_smape[name] = s_err

        # choose best exog model
        valid = {m: e for m, e in errs_nrmse.items() if e > 0 and not np.isnan(e)}
        best = min(valid, key=valid.get) if valid else min(errs_nrmse, key=errs_nrmse.get)
        print(f"    ==> Best exog: {best} err={errs_nrmse[best]:.2f}%")

        # full-series forecast
        fut_idx = pd.date_range(orig_ts.index[-1] + pd.offsets.MonthBegin(), periods=12, freq='MS')
        
        if best == 'arima_exog':
            df_full = pd.DataFrame({'y': orig_ts}).join(cons_ts.to_frame('cons')).dropna()
            y_full, ex_full = df_full['y'], df_full.drop(columns='y')
            try:
                m_full = auto_arima(y=y_full, exogenous=ex_full,
                                    seasonal=True, m=12, stepwise=True, n_jobs=-1)
                fc = pd.Series(m_full.predict(n_periods=12, exogenous=cons_ts.iloc[-12:].values.reshape(-1,1)),
                               index=fut_idx)
            except ValueError:
                last = y_full.iloc[-1] if len(y_full)>0 else 0
                fc = pd.Series(np.repeat(last, 12), index=fut_idx)

        else:  # prophet_exog
            df_hist = pd.DataFrame({
                'ds': orig_ts.index,
                'y': orig_ts.values,
                'cons': cons_ts.values
            }).dropna()
            m = Prophet(); m.add_regressor('cons'); m.fit(df_hist)
            fut = pd.DataFrame({
                'ds': fut_idx,
                'cons': cons_ts.iloc[-12:].fillna(method='ffill').values
            })
            fc = pd.Series(m.predict(fut)['yhat'].values, index=fut_idx)

        # record both errors
        row = {'sheet': sheet, 'item': item, 'bucket': 'exog'}
        for m in preds:
            row[f'nrmse_{m}'] = errs_nrmse[m]
            row[f'smape_{m}'] = errs_smape[m]
        df_metrics.append(row)

        # record forecasts
        df_forecasts.append(pd.DataFrame({
            'sheet': sheet, 'item': item, 'bucket': 'exog',
            'model': best, 'ds': fc.index, 'forecast': fc.values
        }))


    
    return pd.DataFrame(df_metrics), pd.concat(df_forecasts, ignore_index=True),None

def generate_forecasts(filepath: str, cons_path: str = None, az_path: str = None):

    global df_core, non_exog_series, exog_series

    print(f"[DEBUG] generate_forecasts() called with:")
    print(f"        core_path = {filepath}")
    print(f"        cons_path = {cons_path}")
    print(f"        az_path   = {az_path}")

    # ─────────────────────────────────────────────
    # Load the uploaded Excel file 
    # ─────────────────────────────────────────────
    print("[DEBUG] Loading core Excel file...")
    df_core = pd.read_excel(filepath, sheet_name=None)
    print(f"[DEBUG] Loaded {len(df_core)} sheets from core Excel")
    
    df_cons = {}
    if cons_path:
        print("[DEBUG] Loading consumption Excel file...")
        df_cons = pd.read_excel(cons_path, sheet_name=None)
        print(f"[DEBUG] Loaded {len(df_cons)} sheets from consumption Excel")

    df_amaz = None
    if az_path:
        df_amaz = pd.read_excel(az_path)

    if df_amaz is not None:
        df_amaz['month'] = df_amaz['month'].str.strip().str[:3].str.title()
        df_amaz['ds'] = pd.to_datetime(df_amaz['year'].astype(str) + '-' + df_amaz['month'], format='%Y-%b')
        df_amaz_pivot = df_amaz.pivot_table(index='item', columns='ds', values='Total Units').sort_index(axis=1)
        az_cons_workbook = {'AMAZ': df_amaz_pivot} 

    # ─────────────────────────────────────────────
    # Build non_exog_series and exog_series
    # ─────────────────────────────────────────────
    non_exog_series = {}
    exog_series = {}

    for raw_name, df_orig in df_core.items():
        sheet_name = raw_name.strip().upper()
        if sheet_name == 'DPI':
            continue

        df_o = df_orig.set_index(df_orig.columns[0])
        df_o.columns = pd.to_datetime(df_o.columns)
        df_o = df_o.loc[:, ~df_o.columns.duplicated()]
        df_o.index = df_o.index.astype(str).str.strip().str.upper()

        if sheet_name in df_cons:
            df_c = df_cons[sheet_name].set_index(df_cons[sheet_name].columns[0])
            df_c.columns = pd.to_datetime(df_c.columns)
            df_c = df_c.loc[:, ~df_c.columns.duplicated()]
            df_c.index = df_c.index.astype(str).str.strip().str.upper()
        elif sheet_name in az_cons_workbook:
            df_c = az_cons_workbook[sheet_name].copy()
            df_c.index = df_c.index.astype(str).str.strip().str.upper()
        else:
            df_c = None

        for item, row in df_o.iterrows():
            ts_orig = row.dropna().asfreq('MS')
            if df_c is None or item not in df_c.index:
                non_exog_series[(sheet_name, item)] = ts_orig
            else:
                ts_cons = df_c.loc[item].dropna().asfreq('MS')
                exog_series[(sheet_name, item)] = (ts_orig, ts_cons)

    #=== Limited SKUs for testing ===
    def _limit_dict(orig_dict, n=10):
        return dict(list(orig_dict.items())[:n])

    non_exog_series = _limit_dict(non_exog_series, n=10)

    exog_series = _limit_dict(exog_series, n=10)
    # ─────────────────────────────────────────────
    # Run non-exog and exog forecast drivers
    # ─────────────────────────────────────────────
    print(f"[DEBUG] Loaded {len(df_core)} core sheets")
    print(f"[DEBUG] Prepared {len(non_exog_series)} non-exog series")
    print(f"[DEBUG] Prepared {len(exog_series)} exog series")

    
    
    non_metrics, non_forecasts, _ = run_non_exog_driver(non_exog_series) 
    ex_metrics, ex_forecasts, _ = run_exog_driver(exog_series)

    # Combine outputs
    metrics = pd.concat([non_metrics, ex_metrics], ignore_index=True)
    forecasts = pd.concat([non_forecasts, ex_forecasts], ignore_index=True)

    # ─────────────────────────────────────────────
    # Save to timestamped Excel file
    # ─────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"combined_results_{timestamp}.xlsx")  # timestamped file

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        forecasts.to_excel(writer, sheet_name="Forecasts", index=False)
        metrics.to_excel(writer, sheet_name="Metrics", index=False)

    # ─────────────────────────────────────────────
    # Return a summary dictionary
    # ─────────────────────────────────────────────
    return {
        "status": "success",
        "forecast_file": str(output_path.resolve()),  # full path for download
        "message": f"Forecasts generated for {metrics['item'].nunique()} SKUs",
        "metrics_shape": metrics.shape,
        "forecast_shape": forecasts.shape
    }
