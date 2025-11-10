"""
prediccion_meta.py
Sistema modular básico para:
- Descargar datos de META con yfinance
- Analizar variables: Close, Volume, High, Low, Open, Adj Close
- Crear features ligeros (lags, MA, volatility, RSI)
- Entrenar RandomForestRegressor (configurable) o Prophet (opcional)
- Comparar métricas y mostrar gráficas claras
- Mostrar ventana final con decisión: "INVERTIR? SI/NO - POR QUÉ"

Ejecutar con: py prediccion_meta.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import timedelta
import joblib
from tkinter import Tk, Label, Button, Toplevel, LEFT, RIGHT, BOTH, X
from math import sqrt

# -----------------------------
# CONFIGURACIÓN (ajustable)
# -----------------------------
TICKER = "META"                      # ticker a analizar
START_DATE = "2021-01-01"            # fecha inicio (puedes acortarla para menos procesamiento)
END_DATE = None                      # None -> hasta hoy
HORIZON_DAYS = 7                     # horizonte de predicción (días futuros que evaluamos)
MAX_ROWS = 1000                      # límite de días descargados (reduce carga)
RANDOM_FOREST_N_ESTIMATORS = 100     # reduce si quieres menor CPU (ej: 50)
RANDOM_STATE = 42
USE_PROPHET = False                  # si True, usa Prophet (requiere 'prophet' instalado)
MODEL_PATH = "rf_meta_model.joblib"  # archivo para guardar modelo
PLOT_PATH = "plots"                  # carpeta donde guardar png
os.makedirs(PLOT_PATH, exist_ok=True)

# -----------------------------
# UTILIDADES / FEATURES
# -----------------------------
def download_data(ticker=TICKER, start=START_DATE, end=END_DATE, max_rows=MAX_ROWS):
    """Descarga datos históricos desde Yahoo Finance y aplica limite de filas."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError("No se obtuvieron datos. Revisa el ticker o la conexión.")
    df = df.reset_index()
    if len(df) > max_rows:
        df = df.tail(max_rows).reset_index(drop=True)
    # keep required columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def compute_technical_features(df):
    """Crea features simples y ligeros para el modelo."""
    df = df.copy()
    # returns
    df['Return'] = df['Close'].pct_change()
    # moving averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    # volatility (std of returns)
    df['Volatility_5'] = df['Return'].rolling(window=5).std()
    # price range
    df['Range'] = (df['High'] - df['Low']) / df['Open']
    # lag features (1..3 days)
    for lag in range(1, 4):
        df[f'lag_close_{lag}'] = df['Close'].shift(lag)
        df[f'lag_vol_{lag}'] = df['Volume'].shift(lag)
    # RSI (simple implementation)
    delta = df['Close'].diff()
    up = delta.where(delta > 0, 0.0)
    down = -delta.where(delta < 0, 0.0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI_14'] = 100.0 - (100.0 / (1.0 + rs))
    # dropna
    df = df.dropna().reset_index(drop=True)
    return df

# -----------------------------
# ANÁLISIS Y COMPARACIÓN
# -----------------------------
def analyze_basic_stats(df):
    """Devuelve estadísticas básicas y gráficos simples."""
    stats = {
        'last_close': float(df['Close'].iloc[-1]),
        'mean_close': float(df['Close'].mean()),
        'median_volume': float(df['Volume'].median()),
        'volatility_20': float(df['Close'].pct_change().rolling(20).std().dropna().iloc[-1])
    }
    return stats

def compare_recent_vs_historical(df, window=30):
    """Compara últimas n días vs histórico (promedios)."""
    recent = df.tail(window)
    comparison = {
        'recent_mean_close': float(recent['Close'].mean()),
        'overall_mean_close': float(df['Close'].mean()),
        'recent_volatility': float(recent['Close'].pct_change().std()),
        'overall_volatility': float(df['Close'].pct_change().std())
    }
    return comparison

# -----------------------------
# MODELO: RandomForest (ligero)
# -----------------------------
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

# -----------------------------
# PREDICCIÓN FUTURA (últimos datos)
# -----------------------------
def predict_future_with_model(model_tuple, last_rows_df, horizon_days=HORIZON_DAYS):
    """
    Dado (model, feature_cols) y el dataframe con las últimas filas, genera predicción para horizon_days.
    Retorna predicted_price y una medida simple de 'confidence' (std de predictions de árboles).
    """
    model, feature_cols = model_tuple
    X_last = last_rows_df[feature_cols].iloc[-1:].copy()
    # RandomForest predict
    preds = np.array([t.predict(X_last) for t in model.estimators_])
    pred_mean = float(np.mean(preds))
    pred_std = float(np.std(preds))
    return pred_mean, pred_std

# -----------------------------
# GRAFICAS
# -----------------------------
def plot_price_and_forecast(df, y_test_indexed, y_test, y_pred, save_prefix=os.path.join(PLOT_PATH, "price_forecast")):
    """Grafica histórico close y predicciones sobre el conjunto de test."""
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Close'], label='Histórico Close', linewidth=1)
    # plot test true vs pred (align by Date index)
    test_dates = df['Date'].iloc[y_test_indexed.index]
    plt.plot(test_dates, y_test.values, label='Test - Real', linestyle='--')
    plt.plot(test_dates, y_pred, label='Test - Pred (RF)', linestyle=':')
    plt.title(f'{TICKER} - Histórico y Predicción (RandomForest)')
    plt.xlabel('Fecha')
    plt.ylabel('Precio (USD)')
    plt.legend()
    out_png = f"{save_prefix}.png"
    plt.tight_layout()
    plt.savefig(out_png)
    plt.show()

def plot_indicators(df, save_prefix=os.path.join(PLOT_PATH, "indicators")):
    """Grafica indicadores simples: SMA y RSI."""
    plt.figure(figsize=(12,6))
    plt.plot(df['Date'], df['Close'], label='Close')
    if 'SMA_5' in df.columns and 'SMA_10' in df.columns:
        plt.plot(df['Date'], df['SMA_5'], label='SMA 5')
        plt.plot(df['Date'], df['SMA_10'], label='SMA 10')
    plt.title(f'{TICKER} - Precio y SMAs')
    plt.legend()
    plt.xlabel('Fecha')
    plt.ylabel('USD')
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_sma.png")
    plt.show()

    if 'RSI_14' in df.columns:
        plt.figure(figsize=(12,3))
        plt.plot(df['Date'], df['RSI_14'], label='RSI 14')
        plt.axhline(70, color='red', linestyle='--', linewidth=0.7)
        plt.axhline(30, color='green', linestyle='--', linewidth=0.7)
        plt.title(f'{TICKER} - RSI (14)')
        plt.xlabel('Fecha')
        plt.ylabel('RSI')
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_rsi.png")
        plt.show()

# -----------------------------
# DECISIÓN Y VENTANA
# -----------------------------
def decision_rule_and_reason(last_price, predicted_price, pred_std, metrics):
    """
    Regla simple:
    - Si predicho > last * 1.02 => Comprar
    - Si predicho < last * 0.98 => Vender
    - else => Mantener

    Explicación adicional basada en:
    - confidence low (pred_std grande) -> agregar: "confianza baja"
    - R2 bajo -> "modelo con bajo ajuste"
    - volatilidad alta -> "alto riesgo"
    """
    change_pct = (predicted_price - last_price) / last_price
    if predicted_price > last_price * 1.02:
        decision = "COMPRAR"
    elif predicted_price < last_price * 0.98:
        decision = "VENDER"
    else:
        decision = "MANTENER"

    reasons = []
    # confianza aproximada: si desviación relativa grande
    rel_uncertainty = pred_std / (predicted_price + 1e-9)
    if rel_uncertainty > 0.02:  # umbral arbitrario ajustable
        reasons.append(f"Confianza baja (incertidumbre relativa ~ {rel_uncertainty:.2%}).")
    if metrics.get('R2', 0) < 0.2:
        reasons.append(f"Modelo con bajo ajuste (R2={metrics.get('R2'):.2f}).")
    # volatilidad
    if metrics.get('RMSE', 0) / max(last_price, 1) > 0.03:
        reasons.append("Alta volatilidad histórica (error relativo RMSE > 3%).")
    # añadir observación de la dirección
    if change_pct > 0:
        reasons.append(f"Predicción positiva: cambio esperado ~ {change_pct*100:.2f}%.")
    else:
        reasons.append(f"Predicción negativa: cambio esperado ~ {change_pct*100:.2f}%.")

    if decision == "MANTENER" and not reasons:
        reasons.append("Cambio pequeño esperado, no hay señal clara.")

    return decision, reasons, change_pct

def show_decision_window(decision, reasons, last_price, predicted_price, change_pct):
    """Muestra una ventana con la decisión y razones (tkinter)."""
    root = Tk()
    root.title(f"Decisión - {TICKER}")
    # center window small
    root.geometry("520x260")
    Label(root, text=f"Ticker: {TICKER}", font=("Arial", 12, "bold")).pack(pady=4)
    Label(root, text=f"Precio actual: {last_price:.2f} USD", font=("Arial", 11)).pack()
    Label(root, text=f"Predicho a {HORIZON_DAYS} días: {predicted_price:.2f} USD ({change_pct*100:.2f}%)", font=("Arial", 11)).pack(pady=6)
    Label(root, text=f"RECOMENDACIÓN: {decision}", font=("Arial", 14, "bold")).pack(pady=8)

    # razones
    frame = Toplevel(root)
    frame.title("Razones")
    frame.geometry("520x260")
    Label(frame, text="Razones / Explicación:", font=("Arial", 12, "bold")).pack(pady=6)
    for r in reasons:
        Label(frame, text=f"• {r}", anchor="w", justify=LEFT, wraplength=480).pack(fill=X, padx=10, pady=2)

    # botón cerrar
    Button(root, text="Cerrar", command=root.destroy, width=12).pack(pady=12)
    root.mainloop()

# -----------------------------
# FLUJO PRINCIPAL
# -----------------------------
def main():
    print("1) Descargando datos...")
    df0 = download_data()
    print(f"Datos descargados: {len(df0)} filas (desde {df0['Date'].iloc[0].date()} hasta {df0['Date'].iloc[-1].date()})")

    print("2) Calculando features...")
    df = compute_technical_features(df0)

    print("3) Análisis básico y comparación reciente vs histórico...")
    stats = analyze_basic_stats(df)
    comp = compare_recent_vs_historical(df)
    print("Estadísticas básicas:")
    for k,v in stats.items():
        print(f"  - {k}: {v}")
    print("Comparación (últimos 30 días vs general):")
    for k,v in comp.items():
        print(f"  - {k}: {v}")

    # Entrenar modelo ligero (RandomForest)
    print("4) Entrenando RandomForest (ligero)...")
    model, feature_cols, X_test, y_test, y_pred, metrics = train_random_forest(df, horizon_days=HORIZON_DAYS, n_estimators=RANDOM_FOREST_N_ESTIMATORS)
    print("Métricas en test:")
    for k,v in metrics.items():
        print(f"  - {k}: {v:.4f}")

    # graficar
    print("5) Graficando resultados...")
    # Para alinear fechas del test con df, construimos índice de test asociado
    df_for_plot = df.copy()
    # y_test corresponde a las últimas filas del df tras shift; recuperamos indices:
    test_start_idx = int(len(df) * 0.8)
    y_test_indexed = df_for_plot.iloc[test_start_idx: test_start_idx + len(y_test)].reset_index(drop=True)
    # mostrar gráficas (también guardadas en png)
    plot_price_and_forecast(df_for_plot, y_test_indexed, y_test, y_pred)
    plot_indicators(df_for_plot)

    # predicción para últimos datos
    print("6) Predicción futura (usando el modelo entrenado)...")
    model_loaded = joblib.load(MODEL_PATH)  # (model, feature_cols)
    # Predecimos a partir de la última fila disponible (ya con features)
    predicted_price, pred_std = predict_future_with_model(model_loaded, df_for_plot)
    last_price = float(df_for_plot['Close'].iloc[-1])
    print(f"Precio actual: {last_price:.2f} USD")
    print(f"Predicción a {HORIZON_DAYS} días: {predicted_price:.2f} USD (std aprox: {pred_std:.4f})")

    # regla de decisión
    decision, reasons, change_pct = decision_rule_and_reason(last_price, predicted_price, pred_std, metrics)
    print(f"DECISIÓN: {decision}")
    print("Razones:")
    for r in reasons:
        print(" -", r)

    # mostrar ventana con decisión
    show_decision_window(decision, reasons, last_price, predicted_price, change_pct)
    print("Fin.")

if __name__ == "__main__":
    main()
