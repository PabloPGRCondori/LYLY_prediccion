"""
pattern_search.py - Módulo de búsqueda de patrones históricos (KNN)
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from config import HORIZON_DAYS

def find_similar_days(df, n_neighbors=5, features=None):
    """
    Busca en el historial los días más parecidos al estado actual del mercado (última fila).
    
    Args:
        df (pd.DataFrame): DataFrame con datos históricos e indicadores técnicos.
        n_neighbors (int): Cantidad de días similares a buscar.
        features (list): Lista de columnas a usar para la comparación. Si es None, usa un set default.
        
    Returns:
        list: Lista de diccionarios con info de los días similares (fecha, distancia, retorno futuro).
    """
    df = df.copy()
    
    # 1. Definir features de comparación (Estado del mercado)
    # No usamos precios absolutos, sino indicadores normalizados o relativos.
    if features is None:
        features = [
            'RSI_14', 
            'MACD', 
            'Volatility_20', 
            'BB_width', 
            'SMA_20', # Nota: SMA_20 es absoluto, mejor usar distancia al SMA
            'Return'
        ]
        
    # Crear features relativas si es necesario para mejorar la comparación
    # Distancia porcentual al SMA_20 (Normalizada por precio)
    if 'SMA_20' in df.columns and 'Close' in df.columns:
        df['Dist_SMA_20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        if 'SMA_20' in features: features.remove('SMA_20')
        features.append('Dist_SMA_20')
        
    # Asegurar que features existan
    valid_features = [f for f in features if f in df.columns]
    
    # Limpiar NaNs en features (KNN no soporta NaNs)
    df_clean = df.dropna(subset=valid_features).reset_index(drop=True)
    
    if len(df_clean) < n_neighbors + 1:
        return []
        
    # 2. Escalar los datos (Crucial para KNN)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[valid_features])
    
    # 3. Entrenar KNN
    # Usamos metric='euclidean' por defecto
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    knn.fit(X)
    
    # 4. Buscar vecinos del día actual (última fila)
    current_state = X[-1].reshape(1, -1)
    distances, indices = knn.kneighbors(current_state)
    
    # 5. Recopilar resultados
    results = []
    
    # indices[0] contiene los índices de los vecinos. 
    # OJO: El vecino más cercano suele ser el mismo día (distancia 0) si está incluido.
    # Lo saltamos si es idéntico (distancia ~ 0) y es el último día.
    
    last_date = df_clean.iloc[-1]['Date']
    
    for i in range(len(indices[0])):
        idx = indices[0][i]
        dist = distances[0][i]
        
        neighbor_date = df_clean.iloc[idx]['Date']
        
        # Saltar si es el mismo día actual
        if neighbor_date == last_date:
            continue
            
        # Calcular qué pasó después (Outcome)
        # Buscamos el precio HORIZON_DAYS después de ese día vecino
        # Necesitamos volver al DF original para asegurar continuidad si df_clean saltó filas
        # Pero para simplicidad, usaremos df_clean asumiendo continuidad razonable o buscaremos por fecha
        
        try:
            # Buscar el índice en el df original para poder hacer shift/lookup seguro
            original_idx = df[df['Date'] == neighbor_date].index[0]
            future_idx = original_idx + HORIZON_DAYS
            
            outcome = None
            if future_idx < len(df):
                price_then = df.iloc[original_idx]['Close']
                price_future = df.iloc[future_idx]['Close']
                change_pct = (price_future - price_then) / price_then
                outcome = change_pct
            
            results.append({
                'date': neighbor_date,
                'distance': dist,
                'similarity': 1 / (1 + dist), # Score de similitud 0-1
                'outcome_pct': outcome,
                'price_then': df.iloc[original_idx]['Close']
            })
            
        except IndexError:
            continue
            
    # Ordenar por similitud (mayor es mejor)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    return results

def interpret_patterns(results):
    """
    Genera un resumen textual de los patrones encontrados.
    """
    if not results:
        return "No hay suficientes datos históricos para encontrar patrones similares."
        
    valid_outcomes = [r['outcome_pct'] for r in results if r['outcome_pct'] is not None]
    
    if not valid_outcomes:
        return "Se encontraron días similares, pero son muy recientes para saber el resultado futuro."
        
    avg_outcome = sum(valid_outcomes) / len(valid_outcomes)
    positive_outcomes = sum(1 for x in valid_outcomes if x > 0)
    probability_up = positive_outcomes / len(valid_outcomes)
    
    direction = "ALCISTA 📈" if avg_outcome > 0 else "BAJISTA 📉"
    
    summary = (
        f"Basado en los {len(valid_outcomes)} días históricos más parecidos a hoy:\n"
        f"- Tendencia Histórica: **{direction}**\n"
        f"- Retorno Promedio (a {HORIZON_DAYS} días): **{avg_outcome*100:.2f}%**\n"
        f"- Probabilidad de Subida: **{probability_up*100:.0f}%**"
    )
    
    return summary
