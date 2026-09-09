"""Configuración de rutas y parámetros del proyecto."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Online Retail II (UCI / Kaggle). Sample de smoke-test: sample_online_retail.csv
DEFAULT_RAW_FILE = RAW_DATA_DIR / "online_retail_II.csv"

# Stock físico por SKU, si algún día aparece la fuente. Columnas: StockCode, on_hand, on_order.
STOCK_ON_HAND_FILE = RAW_DATA_DIR / "stock_on_hand.csv"

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
    "TEST001",
    "TEST002",
    "ADJUST",
    "ADJUST2",
}

# Substrings seguros (no matchean productos con "check" / "test" en el nombre).
NON_PRODUCT_DESC_KEYWORDS = (
    "postage",
    "bank charges",
    "amazon fee",
    "this is a test",
    "adjustment by",
)

# One-shots conocidos que distorsionan media móvil / top de recomendaciones.
OUTLIER_STOCK_CODES = {
    "23843",  # PAPER CRAFT LITTLE BIRDIE (~80k ud en un día)
}

# Cap de QuantitySold diaria (percentil sobre días con venta > 0).
DAILY_QTY_WINSOR_PERCENTILE = 0.99

# Un cap global lo fija la cola larga y recorta picos reales del top A, que es
# donde más cuesta fallar. Con caps por clase cada grupo se compara consigo mismo.
WINSOR_BY_ABC_CLASS = True

# ETS (Holt-Winters) sobre top-N clase A; el resto sigue en media móvil 30d.
CLASS_A_ETS_TOP_N = 100
ETS_SEASONAL_PERIODS = 7
ETS_MIN_TRAIN_DAYS = 56

# Política de reposición: revisión periódica con lead time fijo.
LEAD_TIME_DAYS = 14
REVIEW_PERIOD_DAYS = 7
SERVICE_Z_BY_CLASS = {"A": 1.88, "B": 1.65, "C": 1.28}
DEFAULT_SERVICE_Z = 1.28
