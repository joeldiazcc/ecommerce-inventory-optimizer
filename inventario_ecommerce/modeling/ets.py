"""ETS (Holt-Winters) para top-N SKUs clase A."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from inventario_ecommerce import config


def class_a_top_skus(abc: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    """Top-N clase A por ventas del último trimestre."""
    n = config.CLASS_A_ETS_TOP_N if top_n is None else top_n
    class_a = abc.loc[abc["ABCClass"] == "A"].sort_values(
        "TotalSales", ascending=False
    )
    if n is not None:
        class_a = class_a.head(int(n))
    return class_a[[config.COL_STOCK_CODE, config.COL_DESCRIPTION]].drop_duplicates()


def _sku_series(daily: pd.DataFrame, stock, desc) -> pd.Series:
    mask = (daily[config.COL_STOCK_CODE] == stock) & (
        daily[config.COL_DESCRIPTION] == desc
    )
    y = (
        daily.loc[mask]
        .sort_values("Date")
        .set_index("Date")["QuantitySold"]
        .astype(float)
    )
    if y.empty:
        return y
    y.index = pd.DatetimeIndex(y.index).normalize()
    return y.asfreq("D", fill_value=0.0)


def fit_ets_horizon(y: pd.Series, horizon: int) -> pd.Series | None:
    """Pronóstico Holt-Winters aditivo (tendencia amortiguada + estacionalidad semanal)."""
    clean = y.dropna()
    if len(clean) < config.ETS_MIN_TRAIN_DAYS:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                clean,
                trend="add",
                damped_trend=True,
                seasonal="add",
                seasonal_periods=config.ETS_SEASONAL_PERIODS,
                initialization_method="estimated",
            )
            fitted = model.fit(optimized=True)
            fc = fitted.forecast(horizon)
        values = np.clip(np.asarray(fc, dtype=float), 0.0, None)
        if not np.isfinite(values).all():
            return None
        return pd.Series(values, index=fc.index)
    except Exception:
        return None


def forecast_ets_class_a(
    daily_sku_demand: pd.DataFrame,
    abc: pd.DataFrame,
    horizon_days: int = 30,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Forecast 30d por SKU clase A. `forecast_daily` es la media del horizonte."""
    rows: list[dict] = []
    for stock, desc in class_a_top_skus(abc, top_n).itertuples(index=False):
        y = _sku_series(daily_sku_demand, stock, desc)
        fc = fit_ets_horizon(y, horizon_days)
        if fc is None:
            continue
        daily_mean = float(fc.mean())
        rows.append(
            {
                config.COL_STOCK_CODE: stock,
                config.COL_DESCRIPTION: desc,
                "forecast_daily": daily_mean,
                "forecast_30d": daily_mean * horizon_days,
                "forecast_model": "ets",
            }
        )
    return pd.DataFrame(rows)


def overlay_ets_forecast(
    baseline_forecast: pd.DataFrame,
    ets_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Sustituye la media móvil por ETS donde el ajuste exista."""
    out = baseline_forecast.copy()
    if "forecast_model" not in out.columns:
        out["forecast_model"] = "ma30"
    if ets_forecast is None or ets_forecast.empty:
        return out

    merged = out.merge(
        ets_forecast,
        on=[config.COL_STOCK_CODE, config.COL_DESCRIPTION],
        how="left",
        suffixes=("", "_ets"),
    )
    use_ets = merged["forecast_daily_ets"].notna()
    merged.loc[use_ets, "forecast_daily"] = merged.loc[use_ets, "forecast_daily_ets"]
    merged.loc[use_ets, "forecast_30d"] = merged.loc[use_ets, "forecast_30d_ets"]
    merged.loc[use_ets, "forecast_model"] = "ets"
    drop_cols = [c for c in merged.columns if c.endswith("_ets")]
    return merged.drop(columns=drop_cols)


def temporal_backtest_ets_vs_baseline(
    daily_sku_demand: pd.DataFrame,
    abc: pd.DataFrame,
    horizon_days: int = 30,
    lookback_days: int = 30,
    top_n: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara MAE de media móvil 30d vs ETS en el mismo holdout, solo top-N clase A."""
    daily = daily_sku_demand.copy()
    daily["Date"] = pd.to_datetime(daily["Date"]).dt.normalize()
    max_date = daily["Date"].max()
    cutoff = max_date - pd.Timedelta(days=horizon_days)
    train = daily[daily["Date"] <= cutoff]
    test = daily[daily["Date"] > cutoff]
    if train.empty or test.empty:
        raise ValueError("No hay suficientes datos para backtest temporal.")

    records: list[dict] = []
    err_ma_parts: list[pd.Series] = []
    err_ets_parts: list[pd.Series] = []
    targets = class_a_top_skus(abc, top_n)
    n_targets = len(targets)

    for i, (stock, desc) in enumerate(targets.itertuples(index=False), start=1):
        y_train = _sku_series(train, stock, desc)
        y_test = _sku_series(test, stock, desc)
        if y_test.empty:
            continue

        tail = y_train.iloc[-lookback_days:] if len(y_train) else y_train
        ma = float(tail.mean()) if len(tail) else 0.0
        pred_ma = pd.Series(ma, index=y_test.index)

        fc = fit_ets_horizon(y_train, horizon_days)
        if fc is None:
            pred_ets = pred_ma.copy()
            used = "ma30_fallback"
        else:
            pred_ets = pd.Series(
                np.asarray(fc, dtype=float)[: len(y_test)],
                index=y_test.index[: len(fc)],
            )
            pred_ets = pred_ets.reindex(y_test.index).fillna(ma).clip(lower=0.0)
            used = "ets"

        err_ma = (y_test - pred_ma).abs()
        err_ets = (y_test - pred_ets).abs()
        err_ma_parts.append(err_ma)
        err_ets_parts.append(err_ets)

        if i == 1 or i % 25 == 0 or i == n_targets:
            print(f"ETS backtest {i}/{n_targets}", flush=True)

        mae_ma = float(err_ma.mean())
        mae_ets = float(err_ets.mean())
        records.append(
            {
                config.COL_STOCK_CODE: stock,
                config.COL_DESCRIPTION: desc,
                "MAE_ma30": mae_ma,
                "MAE_ets": mae_ets,
                "RealMeanDaily": float(y_test.mean()),
                "PredMeanDaily_ma30": float(pred_ma.mean()),
                "PredMeanDaily_ets": float(pred_ets.mean()),
                "ets_used": used,
                "ets_wins": mae_ets < mae_ma,
                "DaysEval": int(len(y_test)),
            }
        )

    by_sku = pd.DataFrame(records)
    concat_ma = pd.concat(err_ma_parts) if err_ma_parts else pd.Series(dtype=float)
    concat_ets = pd.concat(err_ets_parts) if err_ets_parts else pd.Series(dtype=float)
    n_fit = int((by_sku["ets_used"] == "ets").sum()) if not by_sku.empty else 0
    n_win = int(by_sku["ets_wins"].sum()) if not by_sku.empty else 0
    top = config.CLASS_A_ETS_TOP_N if top_n is None else top_n

    global_metrics = pd.DataFrame(
        [
            {
                "CutoffDate": cutoff.date().isoformat(),
                "EvalStartDate": (cutoff + pd.Timedelta(days=1)).date().isoformat(),
                "EvalEndDate": max_date.date().isoformat(),
                "SKUsEval": len(by_sku),
                "SKUsETSFit": n_fit,
                "SKUsETSBetter": n_win,
                "GlobalMAE_ma30": float(concat_ma.mean()) if len(concat_ma) else float("nan"),
                "GlobalMAE_ets": float(concat_ets.mean()) if len(concat_ets) else float("nan"),
                "HorizonDays": horizon_days,
                "LookbackDays": lookback_days,
                "TopN": int(top),
            }
        ]
    )
    return by_sku, global_metrics
