# Diccionario de datos — Online Retail II (UCI / Kaggle)

Fuente: transacciones de un retailer online de UK, dic. 2009 – dic. 2011 (~1.07M líneas de factura).

## Entrada: `data/raw/online_retail_II.csv`

| Columna | Tipo | Descripción |
|---------|------|-------------|
| Invoice | str | Nº de factura. Prefijo `C` ≈ cancelación/devolución |
| StockCode | str | Código de producto (SKU) |
| Description | str | Nombre del producto |
| Quantity | int | Unidades en la línea (negativo en devoluciones) |
| InvoiceDate | datetime | Fecha/hora de la factura |
| Price | float | Precio unitario |
| Customer ID | float | ID de cliente (nulos ≈ venta sin cliente registrado) |
| Country | str | País del cliente |

`Customer ID` y `Country` no se usan en el pipeline actual.

## Limpieza (`clean_transactions`)

- Drop de filas con `StockCode`, `Description`, `InvoiceDate` o `Price` nulos, y con `Description` vacía tras recortar espacios
- Conservar solo `Quantity > 0` y `Price >= 0` (fuera devoluciones y precios inválidos)
- Excluir líneas que no son inventario físico:
  - por código: `POST`, `DOT`, `M`, `C2`, `BANK CHARGES`, `PADS`, `CRUK`, `D`, `TEST001`, `TEST002`, `ADJUST`, `ADJUST2`
  - por frase contenida en la descripción: `postage`, `bank charges`, `amazon fee`, `this is a test`, `adjustment by`
  - por descripción exacta: `check`, `manual`, `adjustment` (exacta a propósito: no debe tumbar productos como `CHECK HAMMOCK`)
- Columna derivada `Sales = Quantity × Price`

Ventana **último trimestre**: 3 meses hacia atrás desde la fecha máxima del dataset (2011-09-10 → 2011-12-09).

## Demanda diaria (`prepare_daily_demand`)

Serie `QuantitySold` por SKU y día, en este orden:

1. Agregación diaria por `(Date, StockCode, Description)`
2. Relleno de días sin venta a 0, desde la 1.ª venta de cada SKU hasta la fecha máxima global
3. Cap de `QuantitySold` al percentil 99 de los días con venta > 0, calculado **dentro de cada clase ABC**
4. Exclusión de one-shots conocidos: `23843` (PAPER CRAFT LITTLE BIRDIE, ~80k ud en un día)

El cap se calcula sobre el dataset cargado, así que cambia con el sample. Con el CSV completo:

| Cap | Valor | Días-SKU recortados (A / B / C) |
|-----|------:|--------------------------------:|
| Global (`winsorize_daily_quantity`) | 225 ud/día | 3,279 / 1,003 / 622 |
| Por clase (`winsorize_daily_quantity_by_class`) | 315 (A) · 183 (B) · 120 (C) | 1,777 / 1,411 / 1,467 |

Un percentil único lo fija la cola larga de clase C y acaba recortando al top A, que es donde el forecast tiene que acertar. Los SKUs sin clase en el trimestre caen al cap global. Se controla con `config.WINSOR_BY_ABC_CLASS`; `prepare_daily_demand` solo aplica caps por clase si se le pasa el `abc`.

## Features y clasificación

| Concepto | Regla |
|----------|-------|
| ABC | Sobre ventas acumuladas del trimestre: A hasta 80%, B hasta 95%, C el resto |
| Rolling | Media, desviación y suma en ventanas de 7 / 30 / 90 días (`min_periods=1`) |
| `demand_cv_30d` | `demand_std_30d / demand_mean_30d`, o 0 si la media es 0 |

## Salida: `data/processed/inventory_reorder_recommendations.csv`

Una fila por SKU con venta en los últimos 30 días **y** presencia en el ABC del trimestre (2,963 filas con el dataset completo).

| Columna | Unidad | Descripción |
|---------|--------|-------------|
| StockCode / Description | — | Identificación del SKU |
| ABCClass | A/B/C | Clase por contribución a ventas del trimestre |
| TotalSales | moneda | Ventas del SKU en el trimestre |
| forecast_daily | ud/día | Demanda diaria prevista |
| forecast_30d | ud | `forecast_daily × 30` |
| forecast_model | `ma30` / `ets` | Modelo usado en ese SKU |
| demand_mean_30d / demand_std_30d | ud/día | Media y desviación de la demanda diaria (últimos 30 d) |
| demand_cv_30d | ratio | Volatilidad relativa de la demanda |
| lead_time_demand | ud | `forecast_daily × 14` |
| safety_stock | ud | `z × demand_std_30d × √14` |
| reorder_point | ud | `lead_time_demand + safety_stock` |
| target_stock | ud | `reorder_point + forecast_daily × 7` |
| on_hand / on_order | ud | Stock físico y pedidos en curso; vacío si no hay fuente |
| inventory_position | ud | `on_hand + on_order` |
| recommended_order_qty | ud | Cuánto pedir (ver `order_basis`) |
| order_basis | texto | `ciclo_revision` o `target_menos_posicion` |
| recommendation | texto | Cadencia de revisión sugerida según clase |

**Parámetros de la política:** lead time 14 días, ciclo de revisión 7 días, z = 1.88 (A) / 1.65 (B) / 1.28 (C).

## Entrada opcional: `data/raw/stock_on_hand.csv`

El dataset son ventas, no inventario. Si se aporta este fichero, `predict` deja de usar la demanda del ciclo y pide `max(0, target_stock − on_hand − on_order)`, solo cuando la posición ha caído al punto de reorden.

| Columna | Tipo | Descripción |
|---------|------|-------------|
| StockCode | str | SKU; debe casar con el del histórico |
| on_hand | float | Unidades físicas disponibles |
| on_order | float | Unidades ya pedidas y no recibidas |

Basta con una de las dos columnas de cantidad; la que falte se toma como 0, igual que los SKUs ausentes del fichero. Con varias filas por SKU (varios almacenes) se suman.

## Salida: `data/processed/forecast_backtest_class_a.csv`

Comparación MA30 vs ETS en el top 100 clase A, holdout de 30 días (corte 2011-11-09).

| Columna | Descripción |
|---------|-------------|
| MAE_ma30 / MAE_ets | Error absoluto medio de cada modelo, en ud/día |
| MAE_delta | `MAE_ma30 − MAE_ets`; positivo = ETS mejor |
| RealMeanDaily | Demanda media real del SKU en el holdout |
| PredMeanDaily_ma30 / _ets | Predicción media de cada modelo |
| ets_used | `ets` si Holt-Winters ajustó; `ma30_fallback` si no |
| ets_wins | Bool: si ETS tuvo menos MAE que la media móvil |
| DaysEval | Días evaluados |

**Parámetros ETS:** Holt-Winters aditivo, tendencia amortiguada, estacionalidad de 7 días, mínimo 56 días de historia, aplicado al top 100 clase A por ventas.

## Salida: `data/processed/policy_simulation_*.csv`

Simulación de reposición sobre el mismo holdout de 30 días (`python -m inventario_ecommerce.modeling.simulate`). `by_sku` trae una fila por SKU y política; `summary`, una por política; `by_class`, una por política y clase ABC.

| Columna | Unidad | Descripción |
|---------|--------|-------------|
| policy | texto | `media`, `z_clase`, `z_90` / `z_95` / `z_99`, `z_clase_ets` |
| demand_total / units_served / units_lost | ud | Demanda del holdout, lo servido y lo perdido |
| fill_rate | ratio | `units_served / demand_total`, en unidades y no en pedidos |
| stockout_days | días | Días con demanda sin servir (por SKU) |
| avg_on_hand | ud | Stock medio a cierre de día; en el resumen, sumado sobre el catálogo |
| days_of_cover | días | `avg_on_hand` dividido por la demanda diaria media |
| orders_placed / units_ordered | — | Pedidos lanzados y unidades pedidas |

**Reglas de la simulación:** revisión cada 7 días, lead time 14 días, venta perdida (sin backorder) y arranque con `target_stock` en almacén. Cada día llegan las recepciones, se sirve la demanda con lo disponible y, si toca revisión y la posición cayó al punto de reorden, se pide hasta el objetivo.

Los demás CSV de `data/processed/` son intermedios: `sales_last_quarter_by_product.csv`, `abc_last_quarter.csv`, `sku_rolling_features_latest.csv`, `forecast_backtest_by_sku.csv`, `forecast_backtest_global.csv`, `forecast_backtest_class_a_global.csv`.
