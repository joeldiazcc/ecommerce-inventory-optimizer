# Handoff — ecommerce-inventory-optimizer

Documento de contexto para continuar en **otro chat**. Fecha: 3 sep 2026.

Pega este archivo (o su ruta) al nuevo agente y pide: *“Lee `references/HANDOFF.md` y continúa desde Próximos pasos.”*

---

## Identidad del proyecto

- **Nombre GitHub:** `ecommerce-inventory-optimizer` (antes `proyecto-a-inventario-ecommerce`)
- **URL:** https://github.com/joeldiazcc/ecommerce-inventory-optimizer
- **Visibilidad:** PUBLIC
- **Cuenta:** `joeldiazcc`
- **Ruta local:** `c:\code\proyecto-a-inventario-ecommerce`
- **Rama:** `main` → `origin/main`
- **Nivel:** junior / portfolio (MVP de inventario e-commerce)

**Stack:** Python, pandas, numpy, matplotlib, Jupyter. Sin Prophet/sklearn aún.

---

## Qué es (una frase)

Pipeline de transacciones **Online Retail II (UCI/Kaggle)** → limpieza → ventas último trimestre → **ABC** → **features rolling** → **forecast baseline 30d** (media móvil) con backtest temporal → **punto de reorden + stock de seguridad** por SKU.

---

## Git (importante)

| Estado | Detalle |
|--------|---------|
| En GitHub | Solo el **first commit** `efd41d5` — scaffold fase 1 (limpieza + notebook 01 + README + figura top-10) |
| **Sin commit / sin push** | Fase 2 completa: `modeling/train.py`, `modeling/predict.py`, notebook 02, ABC/rolling/forecast, README/Makefile actualizados |
| No versionar | `data/raw/online_retail_II.csv` (~90 MB), CSVs en `data/processed/*` (gitignore) |
| Sí versionar | sample `data/raw/sample_online_retail.csv`, `reports/figures/top_products_last_quarter.png` |

**Primer acción en el siguiente chat si se quiere publicar la fase 2:** commit + push de los archivos de modelado (no el dump de Kaggle).

---

## Dataset

- **Fuente:** [Online Retail II UCI (Kaggle)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
- **Local:** `data/raw/online_retail_II.csv` (~90.5 MB, **gitignored**)
- **Origen copia:** `c:\Users\JoelDiazCarissimi\Downloads\online_retail_II.csv`
- **Esquema:** `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`
- **Rango:** 2009-12-01 → 2011-12-09
- **Filas raw:** 1,067,371
- **Nulos típicos:** Description ~4.4k, Customer ID ~243k
- **Default en código:** `config.DEFAULT_RAW_FILE` = `data/raw/online_retail_II.csv`
- **Smoke-test:** `data/raw/sample_online_retail.csv`

---

## Limpieza (`features.clean_transactions`)

- Rename legado: InvoiceNo→Invoice, UnitPrice→Price, CustomerID→Customer ID
- Drop nulos: StockCode, Description, InvoiceDate, Price
- `Quantity > 0` y `Price >= 0` (quita devoluciones)
- `Sales = Quantity * Price`
- **Filtro no-producto** (`config.NON_PRODUCT_STOCK_CODES` + keywords en description): POST, DOT, M, C2, BANK CHARGES, PADS, CRUK, D; y textos tipo manual, postage, adjustment, check, test, amazon fee

Último trimestre = 3 meses hacia atrás desde `max(InvoiceDate)` (no trimestre calendario).

---

## Módulos clave

| Archivo | Rol |
|---------|-----|
| `inventario_ecommerce/config.py` | Rutas, columnas, filtros no-SKU |
| `inventario_ecommerce/dataset.py` | `load_transactions`, `save_processed` |
| `inventario_ecommerce/features.py` | clean, last-quarter, daily demand, ABC, rolling |
| `inventario_ecommerce/plots.py` | Top N barras horizontales |
| `inventario_ecommerce/modeling/train.py` | ABC + rolling latest + backtest MAE/MAPE |
| `inventario_ecommerce/modeling/predict.py` | forecast 30d + política ROP |
| `notebooks/01_carga_limpieza_retail.ipynb` | Fase 1 |
| `notebooks/02_forecast_reorder_baseline.ipynb` | Fase 2 |

**Ejecutar:**

```bash
cd c:\code\proyecto-a-inventario-ecommerce
pip install -r requirements.txt
python -m inventario_ecommerce.modeling.train
python -m inventario_ecommerce.modeling.predict
```

Makefile: `requirements`, `notebook`, `notebook2`, `train`, `predict`.

---

## Resultados locales (tras filtro no-producto)

Ejecutados en máquina del usuario; CSVs **no** están en GitHub.

| Métrica | Valor |
|---------|-------|
| SKUs ABC último trimestre | 3,386 |
| Latest rolling features | 5,717 filas (historial más largo que el trimestre) |
| Backtest por SKU | 2,961 |
| Recomendaciones (`inner` forecast ∩ latest) | **2,961 SKUs** |
| ABC en recomendaciones | A 678 · B 774 · C 1509 |
| Backtest | cutoff 2011-11-09; eval 2011-11-10 → 2011-12-09 |
| **GlobalMAE** | ~20.19 ud/día |
| **GlobalMAPE** | ~284% (ruido de series intermitentes; MAE es la métrica útil) |
| Horizonte / lookback | 30 / 30 días |

**Tabla de decisión:** `data/processed/inventory_reorder_recommendations.csv`

Columnas: StockCode, Description, ABCClass, TotalSales, forecast_daily, forecast_30d, demand_mean/std/cv_30d, lead_time_demand, safety_stock, reorder_point, target_stock, recommended_order_qty, recommendation.

**Política (baseline junior, explícita):**

- Forecast = media diaria lookback 30d × 30
- Lead time = **14 días** (fijo, no hay stock ni LT real)
- Review period = **7 días**
- z por ABC: A 1.88, B 1.65, C 1.28
- `safety_stock = z * σ_30d * sqrt(LT)`
- `ROP = demanda_LT + SS`
- `target_stock = ROP + demanda_diaria * review`
- `recommended_order_qty = target − ROP` (ciclo de revisión, **no** hay inventario on-hand)

**Outliers conocidos (no “arreglar” sin criterio de negocio):**

- `23843` PAPER CRAFT LITTLE BIRDIE: una venta enorme (~80k ud) distorsiona forecast
- Algunos SKU C con picos (`84826` stickers) y CV alto
- MAPE global alto es esperado con demanda intermitente retail

Otros processed locales: `abc_last_quarter.csv`, `sku_rolling_features_latest.csv`, `forecast_backtest_*.csv`, `sales_last_quarter_by_product.csv`.

---

## Decisiones de producto / portfolio

- Cookiecutter DS (data/, notebooks/, paquete), no carpetas vacías `docs/` ni stubs `NotImplementedError`.
- README orientado a problema + resultados, no a la plantilla.
- LICENSE MIT.
- Diccionario: `references/data_dictionary.md`.
- Figura versionada: `reports/figures/top_products_last_quarter.png`.
- `gh` en Windows: `C:\Program Files\GitHub CLI\gh.exe` (PATH a veces no lo ve).

---

## Qué está “terminado” vs no

**MVP cerrado (código local):** ETL → ABC → rolling → baseline + backtest → ROP por SKU.

**No es un sistema de inventario real:** no hay stock actual, ni lead time por proveedor, ni simulador de fill-rate. Forecast es media móvil, no Prophet/ML.

---

## Próximos pasos (prioridad)

1. **Commit + push fase 2** a GitHub (código + notebook 02 + README; no CSVs grandes).
2. **Winsorizar / excluir outliers** (p. ej. 23843) o capar forecast por percentil para que el top de recomendaciones no sea un one-shot.
3. **Rellenar días sin venta a 0** en la serie diaria (ahora rolling solo sobre días con transacción → sesgo al alza).
4. Forecast por familia o top-N SKUs A (Prophet / ETS / LightGBM) vs baseline.
5. Si hay stock on-hand: `pedido = max(0, target − on_hand − on_order)`.
6. Simulador: quiebre vs sobrestock de la política vs “pedir media”.
7. Dashboard simple o tabla en README con top 10 A (sin outliers).

---

## Prompt sugerido para el siguiente chat

```
Workspace: c:\code\proyecto-a-inventario-ecommerce
Lee references/HANDOFF.md.

Contexto: repo público https://github.com/joeldiazcc/ecommerce-inventory-optimizer
Fase 2 está en disco pero NO está pusheada. Dataset Kaggle en data/raw/online_retail_II.csv (gitignored).

Quiero: [commit+push | mejorar forecast | rellenar días a 0 | dashboard | …]
```

---

## Historial breve de la conversación original

1. Scaffold Cookiecutter + notebook carga/limpieza + requirements.
2. Integración CSV Kaggle; pipeline 1.07M → ~1.04M limpios; Q último trimestre 2011-09-10→2011-12-09.
3. Revisión portfolio: quitar stubs, README, LICENSE, data dictionary, figura.
4. Git init, commit `efd41d5`, `gh auth` (usuario `joeldiazcc`), create repo **privado**, rename a `ecommerce-inventory-optimizer`, visibilidad **pública**.
5. Cierre MVP: ABC, rolling, train/predict, notebook 02, filtro no-producto, inner join recomendaciones 2961 SKUs.
6. Este handoff.
