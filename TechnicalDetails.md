# Detalles Técnicos del Proyecto LYLY_prediccion

Este documento proporciona una guía técnica detallada sobre la estructura, funcionalidad y lógica interna del sistema de predicción de precios para META (Meta Platforms Inc.).

## 1. Visión General
El sistema es una aplicación modular en Python diseñada para predecir el precio futuro de acciones utilizando aprendizaje automático (Machine Learning). Se centra en la descarga de datos reales, la generación de indicadores técnicos, el entrenamiento de modelos predictivos y la emisión de recomendaciones operativas claras (COMPRAR, VENDER, MANTENER).

## 2. Estructura del Proyecto

El proyecto está organizado de manera modular para separar responsabilidades:

| Archivo | Descripción |
| :--- | :--- |
| **`prediccion_meta_modular.py`** | **Script Principal (Entry Point).** Controla el flujo de la aplicación mediante un menú interactivo. Gestiona el estado global (datos cargados, modelos entrenados). |
| **`config.py`** | **Configuración.** Centraliza constantes como el ticker (`META`), fechas, horizonte de predicción, parámetros del modelo y rutas de archivos. |
| **`data_loader.py`** | **Capa de Datos.** Maneja la descarga desde Yahoo Finance (`yfinance`), el sistema de caché (`joblib`) y la ingeniería de características (cálculo de indicadores). |
| **`model.py`** | **Capa de Modelado.** Contiene la lógica para entrenar, evaluar y predecir usando Random Forest, Prophet o XGBoost. Incluye validación temporal (`TimeSeriesSplit`). |
| **`analysis.py`** | **Análisis Estadístico.** Calcula métricas descriptivas y compara el comportamiento reciente vs. histórico. |
| **`visualization.py`** | **Visualización.** Genera gráficos estáticos (PNG) de precios, volumen e indicadores técnicos usando `matplotlib`. |
| **`decision.py`** | **Lógica de Negocio.** Traduce las predicciones numéricas en decisiones operativas y gestiona la interfaz gráfica de resultados (`tkinter`). |
| **`big_data.py`** | **Big Data (Nuevo).** Módulo de procesamiento masivo. Descarga slices de GDELT, descomprime y procesa con DuckDB para extraer sentimiento sobre META. |

## 3. Pipeline de Datos (`data_loader.py`)

### 3.1. Adquisición
- **Fuente:** Yahoo Finance (`yfinance`).
- **Validación:** Se verifica que los datos no estén vacíos y contengan las columnas esenciales.
- **Caché:** Se utiliza `joblib` para guardar el DataFrame descargado en disco (`data_cache.joblib`), evitando descargas repetitivas y acelerando el desarrollo.

### 3.2. Ingeniería de Características (Feature Engineering)
La función `compute_technical_features` enriquece los datos crudos (`Open`, `High`, `Low`, `Close`, `Volume`) con indicadores técnicos clave:

- **Tendencia:**
  - `SMA_5`, `SMA_10`, `SMA_20`: Medias Móviles Simples.
  - `EMA_12`, `EMA_26`: Medias Móviles Exponenciales.
  - `MACD`: Diferencia entre EMAs.
- **Momentum:**
  - `RSI_14`: Índice de Fuerza Relativa (14 periodos).
  - `Momentum_10`: Cambio de precio en 10 días.
- **Volatilidad:**
  - `Volatility_5`, `Volatility_20`: Desviación estándar de los retornos.
  - `BB_upper`, `BB_lower`, `BB_width`: Bandas de Bollinger.
- **Volumen:**
  - `OBV`: On-Balance Volume.
- **Lags (Retardos):**
  - Precios y volúmenes de los 3 días anteriores (`lag_close_1`...`3`, `lag_vol_1`...`3`) para capturar dependencia temporal.

## 4. Modelado Predictivo (`model.py`)

El sistema soporta múltiples enfoques, configurables en `config.py` o seleccionados dinámicamente.

### 4.1. Random Forest (Principal)
- **Tipo:** Regresión (`RandomForestRegressor`) o Clasificación (`RandomForestClassifier`).
- **Target:** Precio de cierre desplazado `HORIZON_DAYS` hacia el futuro (por defecto 7 días).
- **Validación:** Implementa `TimeSeriesSplit` para respetar la cronología de los datos (no mezclar futuro con pasado en entrenamiento).
- **Incertidumbre:** Estima la desviación estándar de las predicciones de los árboles individuales para medir la confianza.

### 4.2. Prophet (Opcional)
- Modelo de series temporales de Facebook, ideal para capturar estacionalidad (diaria, semanal, anual).
- Se activa con `USE_PROPHET = True` en `config.py`.

### 4.3. XGBoost (Opcional)
- Implementación disponible para regresión y clasificación mediante Gradient Boosting.

### 4.4. Predicción Horaria (Heurística)
El sistema incluye una estimación experimental a corto plazo (`HORIZON_HOURS`, por defecto 10 horas).
- **Método:** No es un modelo intradía real. Escala la predicción diaria de Random Forest basándose en la fracción de horas de mercado (asumiendo 6.5 horas/día) y ajusta la incertidumbre.
- **Utilidad:** Proporcionar una guía de tendencia inmediata basada en la estructura diaria.

## 5. Lógica de Decisión (`decision.py`)

La función `decision_rule_and_reason` convierte la predicción numérica en una acción:

1.  **Umbrales:**
    - **COMPRAR:** Si `Precio_Predicho > Precio_Actual * 1.02` (Ganancia esperada > 2%).
    - **VENDER:** Si `Precio_Predicho < Precio_Actual * 0.98` (Pérdida esperada > 2%).
    - **MANTENER:** En cualquier otro caso (movimiento lateral o pequeño).

2.  **Factores de Riesgo (Razones):**
    - **Baja Confianza:** Si la incertidumbre relativa del modelo es > 2%.
    - **Mal Ajuste:** Si el `R2` del modelo en validación es < 0.2.
    - **Alta Volatilidad:** Si el error `RMSE` relativo histórico es > 3%.

Esta lógica asegura que el sistema no solo prediga un número, sino que evalúe la calidad de esa predicción antes de sugerir una acción.

## 6. Visualización e Interfaz

- **Gráficos (`visualization.py`):**
  - `price_forecast.png`: Histórico + Predicción en conjunto de prueba.
  - `indicators_sma.png`: Precio con medias móviles.
  - `indicators_rsi.png`: Oscilador RSI con bandas de sobrecompra/sobreventa.
  - `basic_stats.png`: Precio y Volumen combinados.
- **Interfaz (`decision.py`):**
  - Utiliza `tkinter` para mostrar una ventana emergente con la decisión final, el precio objetivo y la lista de razones/advertencias.

## 8. Big Data y Análisis de Sentimiento (`big_data.py`)

Este módulo implementa una arquitectura moderna de procesamiento de datos para integrar noticias globales (GDELT Project) en la predicción.

### 8.1. Arquitectura
- **Fuente:** GDELT 1.0 Event Database (archivos diarios comprimidos en ZIP).
- **Motor:** DuckDB. Base de datos OLAP embebida que permite ejecutar SQL directo sobre archivos comprimidos y CSV/TSV sin cargarlos completamente en RAM (Out-of-Core Learning).
- **Filtrado:** Se buscan eventos donde:
  - `Actor1Name` o `Actor2Name` contenga "META", "FACEBOOK" o "ZUCKERBERG".
  - Se extrae el `AvgTone` (Tono promedio/Sentimiento) y se agrega por día.

### 8.2. Integración con el Modelo (Fase 3)
Los datos de sentimiento se fusionan con los precios históricos bajo reglas estrictas:
1.  **Lagging (Retardo):** Se crea la variable `Sentiment_Lag1`. El sentimiento de "hoy" se usa para predecir el precio de "mañana", evitando el sesgo de anticipación (look-ahead bias).
2.  **Imputación:** Los días sin noticias relevantes se rellenan con un valor neutral (0).
3.  **Feature Importance:** El modelo Random Forest evalúa automáticamente qué tanto influye el sentimiento en la predicción final.

## 9. Dependencias Clave

El proyecto se basa en el stack científico estándar de Python:
- **`yfinance`**: Descarga de datos financieros.
- **`pandas` / `numpy`**: Manipulación de datos y cálculos vectoriales.
- **`scikit-learn`**: Algoritmos de ML (Random Forest) y métricas.
- **`duckdb`**: Procesamiento SQL de alto rendimiento para Big Data.
- **`requests`**: Descarga de archivos GDELT.
- **`matplotlib`**: Generación de gráficos.
- **`joblib`**: Persistencia de modelos y datos (caché).
- **`prophet` / `xgboost`**: Librerías opcionales para modelos avanzados.
