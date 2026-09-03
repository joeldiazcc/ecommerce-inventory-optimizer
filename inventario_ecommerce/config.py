"""Configuración de rutas y parámetros del proyecto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"  # reservado (forecast / reorden)
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Online Retail II (UCI / Kaggle). Sample de smoke-test: sample_online_retail.csv
DEFAULT_RAW_FILE = RAW_DATA_DIR / "online_retail_II.csv"

# Columnas esperadas (esquema Online Retail II / UCI)
COL_INVOICE = "Invoice"
COL_STOCK_CODE = "StockCode"
COL_DESCRIPTION = "Description"
COL_QUANTITY = "Quantity"
COL_INVOICE_DATE = "InvoiceDate"
COL_PRICE = "Price"
COL_CUSTOMER_ID = "Customer ID"
COL_COUNTRY = "Country"

# Códigos frecuentes que no representan SKUs de inventario físico.
NON_PRODUCT_STOCK_CODES = {
    "POST",
    "DOT",
    "M",
    "C2",
    "BANK CHARGES",
    "PADS",
    "CRUK",
    "D",
}

NON_PRODUCT_DESC_KEYWORDS = (
    "manual",
    "postage",
    "bank charges",
    "adjustment",
    "check",
    "test",
    "amazon fee",
)
