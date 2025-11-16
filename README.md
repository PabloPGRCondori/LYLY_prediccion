# Predicción de META (Sistema Modular)

Proyecto en Python para descargar datos reales de META (`yfinance`), crear características técnicas y factores externos, entrenar modelos de predicción (regresión o clasificación) con validación temporal y emitir una decisión operativa clara (🟢 COMPRAR / 🔴 VENDER / ⚪ MANTENER) con explicación.

## Características Clave

- Datos 100% reales: la descarga falla si no hay datos válidos (sin datos sintéticos).
- Flujo modular por menú para ejecutar por bloques y reducir carga.
- Features técnicas: retornos, SMA/EMA, MACD, Bandas de Bollinger, Momentum, OBV, volatilidad, lags, RSI.
- Factores externos: `VIX`, `TNX` (10Y), `GSPC` (S&P500) y sus retornos diarios.
- Modelos: Random Forest (regresión/clasificación), XGBoost (opcional), Prophet (opcional, tendencia).
- Validación temporal: `TimeSeriesSplit` con métricas adecuadas a cada tarea.
- Visualizaciones: precio/volumen e indicadores; se guardan en `plots/`.
- Decisión transparente con umbrales y explicación.
- Cache y persistencia para acelerar y reproducir resultados.

## Estructura del Proyecto

```
├── analysis.py                # Estadísticas básicas y comparación reciente vs histórico
├── config.py                  # Parámetros globales (ticker, fechas, horizontes, paths, etc.)
├── data_loader.py             # Descarga y procesamiento de datos; cache; 100% datos reales
├── decision.py                # Regla de decisión y UI de ventana (tkinter), impresión en consola
├── model.py                   # Entrenamiento y predicción (RandomForest / Prophet)
├── prediccion_meta_modular.py # Script principal con menú y flujo modular
├── visualization.py           # Gráficas de indicadores y estadísticas
├── plots/                     # Salida de imágenes PNG (se crean al graficar)
├── rf_meta_model.joblib       # Modelo entrenado + columnas de features (ignorado en git)
├── rf_meta_model_prophet.joblib # Modelo Prophet (si se usa, ignorado en git)
└── data_cache.joblib          # Cache de datos procesados (ignorado en git)
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

```bash
py prediccion_meta_modular.py
```

Sigue el menú interactivo para correr cada bloque.

## Flujo por Menú (prediccion_meta_modular.py)

1. Cargar y procesar datos
   - Descarga con `yfinance` y crea features técnicos.
   - Opción de usar cache para acelerar.
2. Análisis de datos
   - Estadísticas básicas (último cierre, promedio, mediana volumen, volatilidad 20 días).
   - Comparación 30 días recientes vs. histórico (medias y volatilidad).
3. Entrenar modelo
   - Regresión (precio) o Clasificación (dirección 0/1) según configuración en `config.py`.
   - Modelos disponibles: Random Forest (base), XGBoost (opcional), Prophet (tendencia, regresión).
   - Validación temporal con `TimeSeriesSplit` (si se activa) y métricas del último split.
4. Generar visualizaciones
   - Gráficas de precio y volumen, SMAs (5 y 10), RSI 14.
   - Se guardan en `plots/basic_stats.png`, `plots/indicators_sma.png`, `plots/indicators_rsi.png`.
5. Hacer predicción y decisión
   - Regresión: predicción a `HORIZON_DAYS` con incertidumbre.
   - Clasificación: probabilidad de subida y decisión por umbral.
   - Emite recomendación y razones; puede abrir ventana (tkinter).
6. Limpiar cache de datos
   - Elimina `data_cache.joblib` para recalcular datos procesados.
7. Salir

## Datos y Procesamiento

- Descarga: `data_loader.download_data()` baja columnas estándar (`Date, Open, High, Low, Close, Volume`).
- Garantía de datos reales: si la descarga falla o viene vacía, se lanza excepción con mensaje claro.
- Features internas (`data_loader.compute_technical_features`):
  - `Return`, `SMA_5`, `SMA_10`, `SMA_20`, `EMA_12`, `EMA_26`, `MACD`, `MACD_signal`
  - `Volatility_5`, `Volatility_20`, `Range`, lags de `Close` y `Volume`
  - `RSI_14`, Bandas de Bollinger (`BB_upper`, `BB_lower`, `BB_width`), `Momentum_10`, `OBV`
  - Limpieza con `dropna()` tras ventanas y desplazamientos
- Factores externos (`data_loader.augment_with_external_factors`):
  - `VIX`, `TNX`, `SPX` y sus retornos diarios

## Modelos

- Random Forest (base):
  - Regresión y clasificación; opcionalmente con `StandardScaler`/`MinMaxScaler` vía `Pipeline`.
  - Validación temporal (`TimeSeriesSplit`) y búsqueda de hiperparámetros (`GridSearchCV`/`RandomizedSearchCV`).
- XGBoost (opcional):
  - Regresión y clasificación cuando está instalado (`pip install xgboost`).
- Prophet (opcional):
  - Regresión de tendencia/estacionalidad y evaluación en tramo de test.

## Métricas

- Regresión: `MAE`, `RMSE`, `R²` (último split o tramo más reciente).
- Clasificación: `Accuracy`, `Precision`, `Recall`, `F1`, `ROC-AUC`.
- La decisión usa incertidumbre y calidad del modelo.

## Visualizaciones

- `plots/basic_stats.png`: precio de cierre y volumen.
- `plots/indicators_sma.png`: `Close` + `SMA_5` y `SMA_10`.
- `plots/indicators_rsi.png`: `RSI_14` con líneas 30/70.
  

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
#    (ejecuta por bloques; no hay pipeline completo)
```