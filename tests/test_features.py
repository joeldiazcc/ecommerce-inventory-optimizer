"""Tests de las reglas de limpieza e higiene de demanda."""

import pandas as pd

from inventario_ecommerce import config
from inventario_ecommerce.features import (
    build_daily_sku_demand,
    clean_transactions,
    compute_abc_classification,
    drop_outlier_skus,
    fill_missing_demand_days,
    sales_by_product_last_quarter,
    winsorize_daily_quantity,
)


def _raw_rows(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[
        [
            config.COL_INVOICE,
            config.COL_STOCK_CODE,
            config.COL_DESCRIPTION,
            config.COL_QUANTITY,
            config.COL_INVOICE_DATE,
            config.COL_PRICE,
        ]
    ]


def test_clean_transactions_filters_and_computes_sales():
    raw = _raw_rows(
        [
            {
                config.COL_INVOICE: "1",
                config.COL_STOCK_CODE: "12345",
                config.COL_DESCRIPTION: "WIDGET",
                config.COL_QUANTITY: 2,
                config.COL_INVOICE_DATE: "2011-12-01",
                config.COL_PRICE: 3.0,
            },
            {
                config.COL_INVOICE: "2",
                config.COL_STOCK_CODE: "12346",
                config.COL_DESCRIPTION: None,
                config.COL_QUANTITY: 5,
                config.COL_INVOICE_DATE: "2011-12-01",
                config.COL_PRICE: 1.0,
            },
            {
                config.COL_INVOICE: "C3",
                config.COL_STOCK_CODE: "12345",
                config.COL_DESCRIPTION: "WIDGET",
                config.COL_QUANTITY: -2,
                config.COL_INVOICE_DATE: "2011-12-02",
                config.COL_PRICE: 3.0,
            },
            {
                config.COL_INVOICE: "4",
                config.COL_STOCK_CODE: "POST",
                config.COL_DESCRIPTION: "POSTAGE",
                config.COL_QUANTITY: 1,
                config.COL_INVOICE_DATE: "2011-12-02",
                config.COL_PRICE: 18.0,
            },
        ]
    )

    clean = clean_transactions(raw)

    assert len(clean) == 1
    assert clean.loc[0, "Sales"] == 6.0
    assert clean.loc[0, config.COL_STOCK_CODE] == "12345"


def test_clean_transactions_keeps_products_named_check():
    raw = _raw_rows(
        [
            {
                config.COL_INVOICE: "1",
                config.COL_STOCK_CODE: "22222",
                config.COL_DESCRIPTION: "CHECK HAMMOCK",
                config.COL_QUANTITY: 1,
                config.COL_INVOICE_DATE: "2011-12-01",
                config.COL_PRICE: 4.0,
            },
            {
                config.COL_INVOICE: "2",
                config.COL_STOCK_CODE: "33333",
                config.COL_DESCRIPTION: "check",
                config.COL_QUANTITY: 1,
                config.COL_INVOICE_DATE: "2011-12-01",
                config.COL_PRICE: 4.0,
            },
        ]
    )

    descriptions = set(clean_transactions(raw)[config.COL_DESCRIPTION])

    assert descriptions == {"CHECK HAMMOCK"}


def test_sales_by_product_last_quarter_uses_three_month_window():
    raw = _raw_rows(
        [
            {
                config.COL_INVOICE: "1",
                config.COL_STOCK_CODE: "11111",
                config.COL_DESCRIPTION: "IN WINDOW",
                config.COL_QUANTITY: 1,
                config.COL_INVOICE_DATE: "2011-12-01",
                config.COL_PRICE: 10.0,
            },
            {
                config.COL_INVOICE: "2",
                config.COL_STOCK_CODE: "22222",
                config.COL_DESCRIPTION: "TOO OLD",
                config.COL_QUANTITY: 1,
                config.COL_INVOICE_DATE: "2011-01-15",
                config.COL_PRICE: 10.0,
            },
        ]
    )

    summary = sales_by_product_last_quarter(clean_transactions(raw))

    assert list(summary[config.COL_STOCK_CODE]) == ["11111"]


def test_fill_missing_demand_days_adds_zero_days_per_sku():
    daily = build_daily_sku_demand(
        clean_transactions(
            _raw_rows(
                [
                    {
                        config.COL_INVOICE: "1",
                        config.COL_STOCK_CODE: "11111",
                        config.COL_DESCRIPTION: "WIDGET",
                        config.COL_QUANTITY: 5,
                        config.COL_INVOICE_DATE: "2011-12-01",
                        config.COL_PRICE: 1.0,
                    },
                    {
                        config.COL_INVOICE: "2",
                        config.COL_STOCK_CODE: "11111",
                        config.COL_DESCRIPTION: "WIDGET",
                        config.COL_QUANTITY: 3,
                        config.COL_INVOICE_DATE: "2011-12-04",
                        config.COL_PRICE: 1.0,
                    },
                ]
            )
        )
    )

    filled = fill_missing_demand_days(daily)

    assert len(filled) == 4
    assert list(filled["QuantitySold"]) == [5.0, 0.0, 0.0, 3.0]


def test_winsorize_daily_quantity_caps_at_percentile():
    daily = pd.DataFrame(
        {
            "Date": pd.date_range("2011-12-01", periods=4, freq="D"),
            config.COL_STOCK_CODE: ["11111"] * 4,
            config.COL_DESCRIPTION: ["WIDGET"] * 4,
            "QuantitySold": [1.0, 2.0, 3.0, 100.0],
        }
    )

    capped = winsorize_daily_quantity(daily, percentile=0.5)

    assert capped["QuantitySold"].max() == 2.5


def test_drop_outlier_skus_removes_configured_codes():
    outlier = sorted(config.OUTLIER_STOCK_CODES)[0]
    daily = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2011-12-01", "2011-12-01"]),
            config.COL_STOCK_CODE: [outlier, "11111"],
            config.COL_DESCRIPTION: ["ONE SHOT", "WIDGET"],
            "QuantitySold": [80000.0, 5.0],
        }
    )

    kept = drop_outlier_skus(daily)

    assert list(kept[config.COL_STOCK_CODE]) == ["11111"]


def test_compute_abc_classification_splits_at_80_and_95():
    summary = pd.DataFrame(
        {
            config.COL_STOCK_CODE: ["A1", "B1", "C1"],
            config.COL_DESCRIPTION: ["TOP", "MID", "TAIL"],
            "TotalSales": [80.0, 15.0, 5.0],
        }
    )

    abc = compute_abc_classification(summary)

    assert list(abc["ABCClass"]) == ["A", "B", "C"]
