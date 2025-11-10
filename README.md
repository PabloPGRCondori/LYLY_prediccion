# Predicción de META (Sistema Modular)

Proyecto en Python para descargar datos reales de META (`yfinance`), generar features técnicos ligeros, entrenar un modelo de predicción (Random Forest por defecto, Prophet opcional), visualizar resultados y emitir una recomendación simple (COMPRAR / VENDER / MANTENER) con explicación.

## Características Clave

- Datos 100% reales: la descarga falla con error si no hay datos válidos (sin datos sintéticos).
- Flujo por menú interactivo (1–8) para ejecutar por bloques y reducir carga.
- Features técnicos: retornos, medias móviles (SMA), volatilidad, rango, lags, RSI.
- Modelos: `RandomForestRegressor` (no lineal, robusto) y `Prophet` (opcional, tendencia/estacionalidad).
- Métricas claras: MAE, RMSE, R² sobre el tramo más reciente (split temporal 80/20).
- Visualizaciones: precio y volumen, SMAs y RSI; se guardan en `plots/`.
- Decisión transparente basada en umbrales y calidad del modelo, con razones impresas y ventana gráfica.
- Cache de datos procesados para acelerar iteraciones y persistencia de modelo para reproducibilidad.

## Estructura del Proyecto

```
├── analysis.py                # Estadísticas básicas y comparación reciente vs histórico
├── config.py                  # Parámetros globales (ticker, fechas, horizontes, paths, etc.)
├── data_loader.py             # Descarga y procesamiento de datos; cache; 100% datos reales
├── decision.py                # Regla de decisión y UI de ventana (tkinter), impresión en consola
├── model.py                   # Entrenamiento y predicción (RandomForest / Prophet)
├── prediccion_meta_modular.py # Script principal con menú y flujo modular
├── prediccion_meta.py         # Versión anterior monolítica (referencia)
├── visualization.py           # Gráficas de indicadores y estadísticas
├── plots/                     # Salida de imágenes PNG (se crean al graficar)
├── rf_meta_model.joblib       # Modelo Random Forest entrenado + columnas de features
├── rf_meta_model_prophet.joblib # Modelo Prophet (si se usa)
└── data_cache.joblib          # Cache de datos procesados
```

## Requisitos

- Python 3.13 (recomendado)
- Paquetes:
  - `yfinance`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `joblib`
  - `prophet` (opcional si habilitas `USE_PROPHET=True`)

Instalación rápida:

```bash
pip install yfinance pandas numpy scikit-learn matplotlib joblib
# Opcional
pip install prophet
```

## Configuración

Edita `config.py` para ajustar el comportamiento:

- `TICKER = "META"`: símbolo a descargar.
- `START_DATE`, `END_DATE`: rango temporal (END_DATE=None usa hasta hoy).
- `HORIZON_DAYS = 7`: días hacia adelante a predecir (define el target `Close.shift(-HORIZON_DAYS)`).
- `HORIZON_HOURS = 10`: predicción intradía aproximada (solo con RF; usa una heurística).
- `MAX_ROWS = 1000`: número máximo de filas para reducir carga.
- `RANDOM_FOREST_N_ESTIMATORS = 100`: número de árboles en el bosque.
- `RANDOM_STATE = 42`: semilla para reproducibilidad.
- `USE_PROPHET = False`: alterna entre RF y Prophet.
- `MODEL_PATH = "rf_meta_model.joblib"`: ruta de persistencia del modelo entrenado.
- `PLOT_PATH = "plots"`: carpeta de salida de gráficos.
- `DATA_CACHE_PATH = "data_cache.joblib"`: cache de datos procesados.

## Cómo Ejecutar

- Modo modular (recomendado):

```bash
py prediccion_meta_modular.py
```

Sigue el menú interactivo para correr cada bloque.

- Modo monolítico de referencia:

```bash
py prediccion_meta.py
```

## Flujo por Menú (prediccion_meta_modular.py)

1. Cargar y procesar datos
   - Descarga con `yfinance` y crea features técnicos.
   - Opción de usar cache para acelerar.
2. Análisis de datos
   - Estadísticas básicas (último cierre, promedio, mediana volumen, volatilidad 20 días).
   - Comparación 30 días recientes vs. histórico (medias y volatilidad).
3. Entrenar modelo
   - Entrena `RandomForestRegressor` (por defecto) o `Prophet` si `USE_PROPHET=True`.
   - Muestra métricas en el conjunto de test (último 20%).
4. Generar visualizaciones
   - Gráficas de precio y volumen, SMAs (5 y 10), RSI 14.
   - Se guardan en `plots/basic_stats.png`, `plots/indicators_sma.png`, `plots/indicators_rsi.png`.
5. Hacer predicción y decisión
   - Predicción a `HORIZON_DAYS` y desviación estándar (incertidumbre) del modelo.
   - Con RF: predicción intradía aproximada a `HORIZON_HOURS`.
   - Emite recomendación y razones; puede abrir ventana (tkinter).
6. Ejecutar pipeline completo
   - Corre 1→2→3→4→5 en una secuencia.
7. Limpiar cache de datos
   - Elimina `data_cache.joblib` para recalcular datos procesados.
8. Salir

## Datos y Procesamiento

- Descarga: `data_loader.download_data()` baja columnas estándar (`Date, Open, High, Low, Close, Volume`).
- Garantía de datos reales: si la descarga falla o viene vacía, se lanza excepción con mensaje claro.
- Features (`data_loader.compute_technical_features`):
  - `Return = Close.pct_change()`
  - `SMA_5`, `SMA_10` sobre `Close`
  - `Volatility_5` sobre `Return`
  - `Range = (High - Low) / Open`
  - Lags: `lag_close_1..3`, `lag_vol_1..3`
  - `RSI_14` (implementación simple)
  - `dropna()` para limpiar filas con ventanas y desplazamientos

## Modelos

- Random Forest (por defecto):
  - `train_random_forest`: crea `target = Close.shift(-HORIZON_DAYS)`, excluye `['Date','target','Close','AdjClose']` de `feature_cols`, hace split temporal 80/20.
  - Persiste `(model, feature_cols)` en `MODEL_PATH`.
  - Predicción futura usa las últimas features y estima incertidumbre como std entre árboles.
- Prophet (opcional):
  - Convierte datos a `ds/y`, entrena y evalúa en split temporal 80/20.
  - Predice usando `yhat` y aproxima incertidumbre con bandas (`yhat_upper/lower`).

## Métricas

- `MAE`, `RMSE`, `R²` en el conjunto de test (tramo más reciente).
- La decisión usa además la incertidumbre relativa (`pred_std / predicted_price`) y el error relativo (`RMSE / last_price`).

## Visualizaciones

- `plots/basic_stats.png`: precio de cierre y volumen.
- `plots/indicators_sma.png`: `Close` + `SMA_5` y `SMA_10`.
- `plots/indicators_rsi.png`: `RSI_14` con líneas 30/70.
- En la versión monolítica también se grafica `price_forecast.png` (histórico + test real vs. predicho).

## Decisión

- Regla simple (ajustable):
  - Si `predicted > last * 1.02` ⇒ COMPRAR
  - Si `predicted < last * 0.98` ⇒ VENDER
  - Si no ⇒ MANTENER
- Razones añadidas según:
  - Incertidumbre relativa alta (`pred_std`) ⇒ baja confianza.
  - `R²` bajo ⇒ ajuste pobre del modelo.
  - `RMSE` relativo alto ⇒ volatilidad histórica elevada.

## Cache y Persistencia

- Cache de datos (`DATA_CACHE_PATH`): permite reusar datos procesados sin recalcular.
- Modelo (`MODEL_PATH`): guarda el modelo y columnas de features para asegurar consistencia en inferencia.
- Menú opción 7 limpia la cache de datos.

## Ejemplo de Uso Rápido

```bash
# 1) Ejecuta el script modular
py prediccion_meta_modular.py

# 2) En el menú, corre en orden: 1 → 2 → 3 → 4 → 5
#    (opcional: 6 ejecuta todo el pipeline de una vez)
```

## Resolución de Problemas

- Error de descarga de datos:
  - Verifica conexión a Internet y el símbolo `TICKER`.
  - Ajusta `START_DATE`/`MAX_ROWS` si el rango es demasiado grande.
- Predicción falla por columnas:
  - Si cambiaste `compute_technical_features`, reentrena el modelo (opción 3) para alinear `feature_cols`.
- Resultados variables:
  - Aumenta `RANDOM_FOREST_N_ESTIMATORS` (ej. 200–300) y mantén `RANDOM_STATE` fijo.
- Predicción intradía:
  - Es aproximada; requiere datos intradía reales para mayor fidelidad.

## Mejoras Sugeridas

- Mostrar importancias de features (`model.feature_importances_`) y graficarlas.
- Validación cruzada temporal (`TimeSeriesSplit`) y búsqueda de hiperparámetros (`max_depth`, `min_samples_leaf`).
- Intervalos más fiables: bosques cuantílicos o conformal prediction.
- Features adicionales: MACD, bandas de Bollinger, correlaciones.
- Logging y reintentos de `yfinance` para robustez operativa.

---

Este proyecto es una base sólida y práctica para análisis y predicción diaria de META: rápido de ejecutar, transparente en sus decisiones y fácil de extender.