"""
model.py - Funciones de entrenamiento y predicción del modelo
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from math import sqrt
from config import *

# Import condicional de Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet no está instalado. Ejecuta: pip install prophet")

def train_random_forest(df, horizon_days=HORIZON_DAYS, n_estimators=RANDOM_FOREST_N_ESTIMATORS, save_path=MODEL_PATH):
    """
    Entrena RandomForest para predecir precio 'horizon_days' adelante usando features preparados.
    Retorna: modelo, X_test, y_test, y_pred, metrics
    """
    df = df.copy()
    
    # label: close price horizon days ahead
    df['target'] = df['Close'].shift(-horizon_days)
    df = df.dropna().reset_index(drop=True)
    
    feature_cols = [c for c in df.columns if c not in ['Date', 'target', 'Close', 'AdjClose']]
    X = df[feature_cols]
    y = df['target']

    # split por tiempo: train hasta 80%, test último 20%
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }

    # guardar modelo
    joblib.dump((model, feature_cols), save_path)
    return model, feature_cols, X_test, y_test, y_pred, metrics

def train_model(df):
    """
    Entrena el modelo según USE_PROPHET en config
    Retorna: modelo, metrics, model_type
    """
    print("🔧 Entrenando modelo...")
    
    if USE_PROPHET and PROPHET_AVAILABLE:
        print("📊 Usando Prophet para predicción...")
        model, metrics = train_prophet(df, HORIZON_DAYS)
        model_type = "prophet"
    else:
        print("🌲 Usando Random Forest para predicción...")
        model, feature_cols, X_test, y_test, y_pred, metrics = train_random_forest(df, HORIZON_DAYS)
        model_type = "random_forest"
        # Para Random Forest, incluimos feature_cols en metrics para usarlo después
        metrics['feature_cols'] = feature_cols
    
    return model, metrics, model_type

def predict_future_with_model(model_tuple, last_rows_df, horizon_days=HORIZON_DAYS, model_type="random_forest"):
    """
    Predice el precio futuro usando el modelo entrenado
    Retorna: predicted_price, pred_std
    """
    if model_type == "prophet":
        # Para Prophet, usar función específica
        model = model_tuple[0]  # En Prophet, model_tuple es solo el modelo
        last_date = last_rows_df['Date'].iloc[-1]
        return predict_with_prophet(model, last_date, horizon_days)
    else:
        # Para Random Forest
        model, feature_cols = model_tuple
        
        # Crear datos para predicción
        last_features = last_rows_df[feature_cols].iloc[-1:].values
        
        # Hacer predicción
        predicted_price = model.predict(last_features)[0]
        
        # Calcular incertidumbre (usando std de predicciones de árboles)
        if hasattr(model, 'estimators_'):
            tree_preds = [tree.predict(last_features)[0] for tree in model.estimators_]
            pred_std = np.std(tree_preds)
        else:
            pred_std = 0.1 * predicted_price  # 10% como aproximación
        
        return predicted_price, pred_std

def predict_hours_with_model(model_tuple, last_rows_df, horizon_hours=HORIZON_HOURS):
    """
    Dado (model, feature_cols) y el dataframe con las últimas filas, genera predicción para horizon_hours.
    Retorna predicted_price y una medida simple de 'confidence'.
    """
    model, feature_cols = model_tuple
    X_last = last_rows_df[feature_cols].iloc[-1:].copy()
    
    # RandomForest predict
    preds = np.array([t.predict(X_last) for t in model.estimators_])
    pred_mean = float(np.mean(preds))
    pred_std = float(np.std(preds))
    
    # Ajustar la predicción para horas (asumiendo día de 6.5 horas de trading)
    trading_hours_per_day = 6.5
    hours_fraction = horizon_hours / trading_hours_per_day
    
    # Para predicción a horas, usamos una fracción de la predicción diaria
    # Esto es una aproximación - en un sistema real necesitaríamos datos intraday
    hourly_pred_mean = pred_mean * (1 + (hours_fraction * 0.01))  # Pequeño ajuste
    hourly_pred_std = pred_std * (1 + (hours_fraction * 0.02))     # Mayor incertidumbre
    
    return hourly_pred_mean, hourly_pred_std

def train_prophet(df, horizon_days=HORIZON_DAYS, save_path=MODEL_PATH):
    """
    Entrena modelo Prophet para predecir precio 'horizon_days' adelante.
    Retorna: modelo, metrics
    """
    if not PROPHET_AVAILABLE:
        print("❌ Prophet no está disponible. Instala con: pip install prophet")
        return None, {}
    
    # Preparar datos para Prophet (requiere columnas 'ds' y 'y')
    prophet_df = df[['Date', 'Close']].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Split por tiempo: train hasta 80%, test último 20%
    split_idx = int(len(prophet_df) * 0.8)
    train_df = prophet_df.iloc[:split_idx]
    test_df = prophet_df.iloc[split_idx:]
    
    # Crear y entrenar modelo Prophet
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(train_df)
    
    # Hacer predicciones en test
    future = test_df[['ds']].copy()
    forecast = model.predict(future)
    
    # Calcular métricas
    y_test = test_df['y'].values
    y_pred = forecast['yhat'].values
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    # Guardar modelo
    joblib.dump(model, save_path.replace('.joblib', '_prophet.joblib'))
    return model, metrics

def predict_with_prophet(model, last_date, horizon_days=HORIZON_DAYS):
    """
    Predice con Prophet para horizon_days adelante
    Retorna predicted_price y confidence_interval
    """
    if not PROPHET_AVAILABLE:
        return None, None
    
    import pandas as pd  # Importar pandas aquí para evitar circular imports
    
    # Crear fechas futuras
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon_days,
        freq='D'
    )
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Predecir
    forecast = model.predict(future_df)
    
    # Obtener predicción y intervalo de confianza
    predicted_price = float(forecast['yhat'].iloc[-1])
    pred_std = float((forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) / 4)  # Aproximación
    
    return predicted_price, pred_std

def load_model(model_path=MODEL_PATH):
    """Carga el modelo desde archivo"""
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

def model_exists():
    """Verifica si el modelo ya está entrenado"""
    return os.path.exists(MODEL_PATH)