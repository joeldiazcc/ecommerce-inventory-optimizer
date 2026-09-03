"""Limpieza, agregaciones y features para análisis de inventario."""

import numpy as np
import pandas as pd

from inventario_ecommerce import config


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia nulos, cancela devoluciones y tipa columnas clave."""
    cleaned = df.copy()

    # Normalizar nombres si vienen en formato Online Retail clásico
    rename_map = {
        "InvoiceNo": config.COL_INVOICE,
        "UnitPrice": config.COL_PRICE,
        "CustomerID": config.COL_CUSTOMER_ID,
    }
    cleaned = cleaned.rename(columns={k: v for k, v in rename_map.items() if k in cleaned.columns})

    required = [
        config.COL_STOCK_CODE,
        config.COL_DESCRIPTION,
        config.COL_QUANTITY,
        config.COL_INVOICE_DATE,
        config.COL_PRICE,
    ]
    missing = [c for c in required if c not in cleaned.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    cleaned[config.COL_INVOICE_DATE] = pd.to_datetime(
        cleaned[config.COL_INVOICE_DATE], errors="coerce"
    )

    # Nulos críticos: fecha, producto o precio
    cleaned = cleaned.dropna(
        subset=[
            config.COL_STOCK_CODE,
            config.COL_DESCRIPTION,
            config.COL_INVOICE_DATE,
            config.COL_PRICE,
        ]
    )

    cleaned[config.COL_DESCRIPTION] = cleaned[config.COL_DESCRIPTION].astype(str).str.strip()
    cleaned = cleaned[cleaned[config.COL_DESCRIPTION] != ""]

    # Filtrar devoluciones / cantidades o precios no válidos
    cleaned = cleaned[
        (cleaned[config.COL_QUANTITY] > 0) & (cleaned[config.COL_PRICE] >= 0)
    ]

    # Quitar líneas que no representan stock físico de inventario.
    stock_upper = cleaned[config.COL_STOCK_CODE].astype(str).str.upper().str.strip()
    desc_lower = cleaned[config.COL_DESCRIPTION].astype(str).str.lower()
    non_product_mask = stock_upper.isin(config.NON_PRODUCT_STOCK_CODES) | (
        desc_lower.str.contains("|".join(config.NON_PRODUCT_DESC_KEYWORDS), regex=True)
    )
    cleaned = cleaned[~non_product_mask]

    cleaned["Sales"] = cleaned[config.COL_QUANTITY] * cleaned[config.COL_PRICE]
    return cleaned.reset_index(drop=True)


def sales_by_product_last_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ventas totales por producto en el último trimestre disponible."""
    if df.empty:
        raise ValueError("El DataFrame está vacío tras la limpieza.")

    max_date = df[config.COL_INVOICE_DATE].max()
    start_quarter = (max_date - pd.DateOffset(months=3)) + pd.Timedelta(days=1)
    last_q = df[
        (df[config.COL_INVOICE_DATE] >= start_quarter)
        & (df[config.COL_INVOICE_DATE] <= max_date)
    ].copy()

    summary = (
        last_q.groupby([config.COL_STOCK_CODE, config.COL_DESCRIPTION], as_index=False)
        .agg(
            QuantitySold=(config.COL_QUANTITY, "sum"),
            TotalSales=("Sales", "sum"),
            NTransactions=(config.COL_INVOICE_DATE, "count"),
        )
        .sort_values("TotalSales", ascending=False)
        .reset_index(drop=True)
    )
    summary.attrs["period_start"] = start_quarter
    summary.attrs["period_end"] = max_date
    return summary


def build_daily_sku_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la demanda diaria por SKU."""
    daily = df.copy()
    daily["Date"] = daily[config.COL_INVOICE_DATE].dt.floor("D")
    result = (
        daily.groupby(
            ["Date", config.COL_STOCK_CODE, config.COL_DESCRIPTION], as_index=False
        ).agg(
            QuantitySold=(config.COL_QUANTITY, "sum"),
            DailySales=("Sales", "sum"),
            NTransactions=(config.COL_INVOICE, "nunique"),
        )
    )
    return result.sort_values(["Date", config.COL_STOCK_CODE]).reset_index(drop=True)


def compute_abc_classification(
    sales_summary: pd.DataFrame,
    sales_col: str = "TotalSales",
) -> pd.DataFrame:
    """Clasifica SKUs por contribución acumulada de ventas (A/B/C)."""
    abc = sales_summary.copy()
    if abc.empty:
        return abc

    abc = abc.sort_values(sales_col, ascending=False).reset_index(drop=True)
    total_sales = abc[sales_col].sum()
    abc["SalesShare"] = np.where(total_sales > 0, abc[sales_col] / total_sales, 0.0)
    abc["CumSalesShare"] = abc["SalesShare"].cumsum()

    abc["ABCClass"] = np.select(
        [
            abc["CumSalesShare"] <= 0.80,
            abc["CumSalesShare"] <= 0.95,
        ],
        ["A", "B"],
        default="C",
    )
    return abc


def build_rolling_features(
    daily_sku_demand: pd.DataFrame,
    windows: tuple[int, ...] = (7, 30, 90),
) -> pd.DataFrame:
    """Crea features rolling de demanda por SKU."""
    features = daily_sku_demand.copy()
    features = features.sort_values(["Date", config.COL_STOCK_CODE]).reset_index(drop=True)
    group_cols = [config.COL_STOCK_CODE, config.COL_DESCRIPTION]

    for window in windows:
        grouped = features.groupby(group_cols, sort=False)["QuantitySold"]
        features[f"demand_mean_{window}d"] = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        features[f"demand_std_{window}d"] = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=1).std()
        ).fillna(0.0)
        features[f"demand_sum_{window}d"] = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=1).sum()
        )

    features["demand_cv_30d"] = np.where(
        features["demand_mean_30d"] > 0,
        features["demand_std_30d"] / features["demand_mean_30d"],
        0.0,
    )
    return features
