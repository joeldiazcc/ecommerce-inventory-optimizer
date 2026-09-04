# Handoff — ecommerce-inventory-optimizer

Documento de contexto para continuar en **otro chat**. Fecha: 4 sep 2026.

Pega este archivo (o su ruta) al nuevo agente y pide: *“Lee `references/HANDOFF.md` y continúa desde Próximos pasos.”*

---

## Identidad del proyecto

- **Nombre GitHub:** `ecommerce-inventory-optimizer`
- **URL:** https://github.com/joeldiazcc/ecommerce-inventory-optimizer
- **Visibilidad:** PUBLIC
- **Cuenta:** `joeldiazcc` (commits: `joeldiazcarissimi@gmail.com`)
- **Ruta local:** `c:\code\proyecto-a-inventario-ecommerce`
- **Rama base:** `main` (fase 2 mergeada vía PR #3)
- **Rama activa fase 3:** `feat/fase-3-demand-hygiene`
- **Nivel:** junior / portfolio (MVP de inventario e-commerce)

**Stack:** Python, pandas, numpy, matplotlib, Jupyter. Sin Prophet/sklearn aún.

---

## Qué es (una frase)

Pipeline Online Retail II → limpieza → ABC → **demanda diaria con ceros + winsor + drop outliers** → features rolling → forecast baseline 30d + ROP por SKU.

---

## Git

| Estado | Detalle |
|--------|---------|
| En `main` | Fase 1 + fase 2 (ABC, train/predict, notebook 02) |
| En curso | Fase 3: higiene de demanda en `feat/fase-3-demand-hygiene` |
| No versionar | `data/raw/online_retail_II.csv`, CSVs en `data/processed/*` |
| Sí versionar | sample, figura top-10, notebooks, código |

---

## Fase 3 (qué cambió)

1. **Filtro no-producto:** quitados substrings `check`/`test`/`manual`/`adjustment` demasiado agresivos; códigos `TEST001/2`, `ADJUST`; frases `this is a test`, `adjustment by`; desc exacta `check`/`manual`/`adjustment`.
2. **`fill_missing_demand_days`:** calendario continuo por SKU (1.ª venta → max fecha) con QuantitySold=0.
3. **`winsorize_daily_quantity`:** cap global P99 sobre días con venta > 0.
4. **`drop_outlier_skus`:** excluye `23843` del modelado.
5. **`prepare_daily_demand`:** orquesta lo anterior; lo usan `train` y `predict`.
6. Forecast solo SKUs con venta > 0 en lookback; policy con **inner** join a ABC (último trimestre).

Notebook: `notebooks/03_demand_hygiene.ipynb`. Makefile: `notebook3`.

---

## Resultados locales (fase 3)

| Métrica | Valor |
|---------|-------|
| SKUs ABC último trimestre | 3,393 |
| Latest rolling | 5,733 |
| Recomendaciones | **2,963** (A 677 · B 774 · C 1512) |
| Backtest | cutoff 2011-11-09 |
| **GlobalMAE** | ~4.06 ud/día (incluye días a 0; fase 2 era ~20.19 sin ceros) |
| **GlobalMAPE** | ~143% |
| Top forecast | `23084` RABBIT NIGHT LIGHT (sin `23843`) |

---

## Módulos clave

| Archivo | Rol |
|---------|-----|
| `config.py` | Rutas, filtros, `OUTLIER_STOCK_CODES`, `DAILY_QTY_WINSOR_PERCENTILE` |
| `features.py` | clean, ABC, daily, fill zeros, winsor, prepare_daily_demand, rolling |
| `modeling/train.py` | ABC + rolling + backtest |
| `modeling/predict.py` | forecast 30d + ROP |
| `notebooks/03_demand_hygiene.ipynb` | Fase 3 |

```bash
cd c:\code\proyecto-a-inventario-ecommerce
python -m inventario_ecommerce.modeling.train
python -m inventario_ecommerce.modeling.predict
```

---

## Próximos pasos (prioridad)

1. Commit + push / PR de `feat/fase-3-demand-hygiene`.
2. Forecast por familia o top-N SKUs A (Prophet / ETS / LightGBM) vs baseline.
3. Stock on-hand: `pedido = max(0, target − on_hand − on_order)`.
4. Simulador quiebre vs sobrestock.
5. Dashboard / tabla README top 10 A.

---

## Prompt sugerido

```
Workspace: c:\code\proyecto-a-inventario-ecommerce
Lee references/HANDOFF.md.
Rama: feat/fase-3-demand-hygiene (o main si ya mergeada).
Quiero: [PR fase 3 | forecast ML | simulador | …]
```
