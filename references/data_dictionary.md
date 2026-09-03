# Diccionario de datos — Online Retail II (UCI / Kaggle)

Fuente: transacciones de un retailer online UK (dic. 2009 – dic. 2011).

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

## Reglas de limpieza usadas en este proyecto

- Drop filas con `Description`, `InvoiceDate` o `Price` nulos
- Conservar solo `Quantity > 0` y `Price >= 0`
- `Sales = Quantity * Price`
- Ventana “último trimestre” = 3 meses hacia atrás desde la fecha máxima del dataset
