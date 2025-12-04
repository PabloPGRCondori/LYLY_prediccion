
"""
signal_aggregator.py - Módulo para unificar todas las señales en una puntuación final.
"""

def calculate_confidence_score(technical_signals, ml_prediction, knn_pattern, fundamentals):
    """
    Calcula una puntuación de confianza (0-100) y una señal unificada (COMPRA/VENTA/NEUTRO).
    
    Args:
        technical_signals (dict): {'rsi': val, 'macd_bullish': bool, 'trend': 'up/down'}
        ml_prediction (dict): {'direction': 'up/down', 'confidence': 0.0-1.0}
        knn_pattern (dict): {'direction': 'up/down', 'probability': 0.0-1.0}
        fundamentals (dict): {'recommendation': 'buy/sell', 'upside_potential': float}
        
    Returns:
        dict: {
            'score': 0-100, 
            'signal': 'STRONG BUY' | 'BUY' | 'NEUTRAL' | 'SELL' | 'STRONG SELL',
            'breakdown': list of reasons
        }
    """
    score = 50.0  # Base neutral
    reasons = []
    
    # --- 1. ANÁLISIS TÉCNICO (Peso: 30%) ---
    rsi = technical_signals.get('rsi', 50)
    if rsi < 30:
        score += 15
        reasons.append("RSI indica Sobreventa (Positivo)")
    elif rsi > 70:
        score -= 15
        reasons.append("RSI indica Sobrecompra (Negativo)")
        
    if technical_signals.get('macd_bullish'):
        score += 10
        reasons.append("MACD Cruce Alcista")
    else:
        score -= 10
        reasons.append("MACD Cruce Bajista")
        
    # --- 2. MACHINE LEARNING (Peso: 30%) ---
    # Asumimos que ml_prediction['direction'] es 'up' o 'down'
    if ml_prediction:
        ml_dir = ml_prediction.get('direction')
        ml_conf = ml_prediction.get('confidence', 0.5)
        
        impact = 30 * ml_conf
        if ml_dir == 'up':
            score += impact
            reasons.append(f"IA predice subida (Confianza: {ml_conf:.0%})")
        else:
            score -= impact
            reasons.append(f"IA predice bajada (Confianza: {ml_conf:.0%})")
            
    # --- 3. PATRONES HISTÓRICOS KNN (Peso: 20%) ---
    if knn_pattern:
        knn_dir = knn_pattern.get('direction') # 'ALCISTA' / 'BAJISTA'
        knn_prob = knn_pattern.get('probability', 0.5)
        
        impact = 20 * knn_prob
        if 'ALCISTA' in knn_dir:
            score += impact
            reasons.append(f"Patrones históricos alcistas ({knn_prob:.0%})")
        elif 'BAJISTA' in knn_dir:
            score -= impact
            reasons.append(f"Patrones históricos bajistas ({knn_prob:.0%})")
            
    # --- 4. FUNDAMENTALES (Peso: 20%) ---
    rec = fundamentals.get('recommendation', '').lower()
    if 'buy' in rec:
        score += 15
        if 'strong' in rec: score += 5
        reasons.append(f"Analistas recomiendan: {rec.upper()}")
    elif 'sell' in rec:
        score -= 15
        if 'strong' in rec: score -= 5
        reasons.append(f"Analistas recomiendan: {rec.upper()}")
        
    upside = fundamentals.get('upside_potential', 0)
    if upside > 0.10: # >10% potencial
        score += 10
        reasons.append(f"Potencial alcista vs Target: +{upside:.1%}")
    elif upside < -0.10:
        score -= 10
        reasons.append(f"Precio por encima del Target: {upside:.1%}")
        
    # --- RESULTADO FINAL ---
    score = max(0, min(100, score))
    
    if score >= 80: signal = "COMPRA FUERTE 🚀"
    elif score >= 60: signal = "COMPRA 🟢"
    elif score <= 20: signal = "VENTA FUERTE 🩸"
    elif score <= 40: signal = "VENTA 🔴"
    else: signal = "NEUTRAL ⚪"
    
    return {
        'score': score,
        'signal': signal,
        'reasons': reasons
    }
