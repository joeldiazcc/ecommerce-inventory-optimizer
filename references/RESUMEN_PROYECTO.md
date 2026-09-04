# Resumen del proyecto — ecommerce-inventory-optimizer

Fecha: 4 sep 2026. Contexto de este chat + estado del repo para continuar en otra sesión.

---

## Proyecto

| | |
|--|--|
| **Repo** | https://github.com/joeldiazcc/ecommerce-inventory-optimizer (público) |
| **Cuenta** | `joeldiazcc` |
| **Ruta local** | `c:\code\proyecto-a-inventario-ecommerce` |
| **Qué hace** | Transacciones Online Retail II → limpieza → ABC → demanda diaria higienizada → forecast media móvil 30d → punto de reorden / stock de seguridad por SKU |
| **Stack** | Python, pandas, numpy, matplotlib, Jupyter |

**Git / identidad de commits:** `user.email = joeldiazcarissimi@gmail.com`, `user.name = Joel Felipe Díaz Carissimi`. Antes git tomaba el UPN de Azure AD (`joeldiaz@capitole-consulting.com`) y Cursor añadía `Co-authored-by: Cursor` → contributors `joeldiazcapitole` + `cursoragent`. Para commits nuevos: email de Gmail + desactivar co-author de Cursor en Settings. Historial viejo no cambia sin reescribir.

---

## Qué hay construido

1. **Limpieza** — nulos, devoluciones, no-producto, ventas último trimestre.
2. **ABC + rolling + baseline** — `train` / `predict`, notebook 02, ROP con z por clase (A/B/C), LT=14d, review=7d.
3. **Higiene de demanda** — ceros en calendario, winsor P99, drop `23843`, filtros más precisos; notebook 03; `prepare_daily_demand` en el pipeline.

| Artefacto | Uso |
|-----------|-----|
| `inventario_ecommerce/features.py` | clean, ABC, daily, fill zeros, winsor, rolling |
| `inventario_ecommerce/modeling/train.py` | backtest MAE/MAPE |
| `inventario_ecommerce/modeling/predict.py` | forecast + recomendaciones |
| `data/processed/inventory_reorder_recommendations.csv` | tabla de decisión (local, gitignored) |

**No versionar:** CSV Kaggle (~90 MB), CSVs de `data/processed/`.  
**Sí versionar:** sample, figura top-10, notebooks, código.

---

## Resultados locales (tras higiene)

| Métrica | Valor |
|---------|------:|
| SKUs ABC (último trimestre) | 3,393 |
| Recomendaciones (lookback activo ∩ ABC) | 2,963 |
| GlobalMAE (backtest 30d, con días a 0) | ~4.06 ud/día |
| GlobalMAPE | ~143% |
| Top forecast | `23084` RABBIT NIGHT LIGHT (sin `23843`) |

MAE ~20 de la versión sin ceros no es comparable: solo evaluaba días con venta.

---

## Git en este chat

| Rama / PR | Estado |
|-----------|--------|
| `feat/fase-2-forecast-reorder` | Mergeada a `main` (PR #3) |
| `feat/fase-3-demand-hygiene` | Abierta; draft PR #4 — https://github.com/joeldiazcc/ecommerce-inventory-optimizer/pull/4 |
| Limpieza docs (este cambio) | Local en la rama fase 3; commit/push pendiente si aplica |

Ejecutar:

```bash
cd c:\code\proyecto-a-inventario-ecommerce
pip install -r requirements.txt
python -m inventario_ecommerce.modeling.train
python -m inventario_ecommerce.modeling.predict
```

---

## Decisiones útiles

- Portfolio junior: baseline explicable, sin Prophet/ML todavía.
- `recommended_order_qty = forecast_daily × 7` (ciclo de revisión; no hay stock on-hand).
- Recomendaciones = SKUs con venta en lookback 30d ∩ ABC del último trimestre.
- Notebooks y README sin secciones de “próximas mejoras” (el backlog vive solo aquí abajo).

---

## Próximos pasos

1. Mergear PR #4 (fase 3) tras revisión; opcional: commit de la limpieza de textos de este pase.
2. Forecast ML / por familia o top-N clase A (Prophet, ETS o LightGBM) vs baseline.
3. Si hay stock: `pedido = max(0, target − on_hand − on_order)`.
4. Simulador quiebre vs sobrestock vs “pedir la media”.
5. Tabla/dashboard en README con top 10 A (ya sin outliers).
6. (Opcional) Reescribir historial git para dejar un solo contributor `joeldiazcc` (force push; solo si el repo aún es pequeño y lo aceptas).

---

## Prompt para otro chat

```
Workspace: c:\code\proyecto-a-inventario-ecommerce
Lee references/RESUMEN_PROYECTO.md.
Rama actual: feat/fase-3-demand-hygiene (o main si PR #4 ya mergeó).
Quiero: [merge PR | forecast ML | simulador | dashboard | …]
```
