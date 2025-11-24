"""
decision.py - Funciones de decisión e interfaz
MODIFICADO: Mayor sensibilidad para mostrar siempre el impacto de Big Data.
"""

from tkinter import Tk, Label, Button, Toplevel, LEFT, X, Frame
from config import *

def decision_rule_and_reason(last_price, predicted_price, pred_std, metrics, 
                             sentiment_score=0, external_factors=None, horizon_name="días"):
    """
    Regla Avanzada Sensible:
    - Umbrales reducidos para detectar sentimiento leve.
    - Siempre reporta el estado Macro (VIX/Cobre).
    """
    # 1. Calcular cambio esperado por el modelo numérico
    change_pct = (predicted_price - last_price) / last_price
    
    # 2. Definir umbrales base
    umbral_compra = 1.02  # +2%
    umbral_venta = 0.98   # -2%
    
    reasons = []

    # --- ANÁLISIS DE SENTIMIENTO (UMBRALES MÁS SENSIBLES) ---
    # Antes era 0.15, ahora bajamos a 0.05 para que detecte tu -0.13
    if sentiment_score > 0.05:
        umbral_compra = 1.01 
        reasons.append(f"📰 Sentimiento POSITIVO ({sentiment_score:.2f}): Noticias impulsan la compra.")
    elif sentiment_score < -0.05:
        umbral_venta = 0.99 
        # Aquí caerá tu -0.13 porque es menor que -0.05
        reasons.append(f"📰 Sentimiento NEGATIVO ({sentiment_score:.2f}): Noticias generan precaución.")
    else:
        # Si es neutral (ej. 0.00), AHORA SÍ lo decimos
        reasons.append(f"📰 Sentimiento NEUTRAL ({sentiment_score:.2f}): Noticias sin impacto direccional.")

    # --- ANÁLISIS DE FACTORES EXTERNOS ---
    risk_lock = False
    
    if external_factors:
        vix = external_factors.get('VIX', 0)
        cobre = external_factors.get('Cobre', 0)
        
        # VIX (Miedo)
        if vix > 25:
            risk_lock = True
            reasons.append(f"😨 ALTO RIESGO (VIX {vix:.2f}): Mercado muy volátil.")
        else:
            # AHORA SÍ confirmamos que es seguro
            reasons.append(f"✅ Volatilidad Global Controlada (VIX {vix:.2f}).")
        
        # Cobre
        if cobre > 0 and cobre < 3.5:
            reasons.append(f"📉 Cobre Débil (${cobre:.2f}): Señal de riesgo económico.")
        elif cobre > 4.5:
             # AHORA SÍ presumimos el precio alto
             reasons.append(f"🏗️ Cobre Fuerte (${cobre:.2f}): Soporte macroeconómico.")

    # 3. Toma de Decisión Final
    if risk_lock:
        decision = "MANTENER" # Bloqueo por pánico
    elif predicted_price > last_price * umbral_compra:
        decision = "COMPRAR"
    elif predicted_price < last_price * umbral_venta:
        decision = "VENDER"
    else:
        decision = "MANTENER"

    # 4. Razones Técnicas
    rel_uncertainty = pred_std / (predicted_price + 1e-9)
    if rel_uncertainty > 0.02:
        reasons.append(f"⚠️ Incertidumbre técnica moderada (~{rel_uncertainty:.2%}).")
    
    direction = "ALZA" if change_pct > 0 else "BAJA"
    reasons.append(f"📊 Modelo numérico predice {direction} de {change_pct*100:.2f}% a {horizon_name}.")

    return decision, reasons, change_pct

def show_decision_window(decision, reasons, last_price, predicted_price, change_pct, sentiment_score=0, external_factors=None):
    """Ventana gráfica con todos los indicadores."""
    root = Tk()
    root.title(f"Sistema de Decisión - {TICKER}")
    root.geometry("650x500") # Un poco más alta para que quepan las razones
    
    # Título
    Label(root, text=f"Análisis Inteligente: {TICKER}", font=("Arial", 16, "bold"), fg="#003366").pack(pady=10)
    
    # Sección Precios
    frame_price = Frame(root, relief="groove", borderwidth=2, bg="#f0f0f0")
    frame_price.pack(fill=X, padx=20, pady=5)
    Label(frame_price, text="--- Modelo Predictivo ---", font=("Arial", 10, "bold"), bg="#f0f0f0").pack()
    Label(frame_price, text=f"Precio Actual: {last_price:.2f} USD", font=("Arial", 11), bg="#f0f0f0").pack()
    
    pct_color = "green" if change_pct > 0 else "red"
    Label(frame_price, text=f"Predicción ({HORIZON_DAYS}d): {predicted_price:.2f} USD ({change_pct*100:+.2f}%)", 
          font=("Arial", 12, "bold"), fg=pct_color, bg="#f0f0f0").pack(pady=5)

    # Sección Big Data
    frame_data = Frame(root, relief="groove", borderwidth=2, bg="#fff8e1") # Color suave
    frame_data.pack(fill=X, padx=20, pady=5)
    Label(frame_data, text="--- Big Data & Sentimiento ---", font=("Arial", 10, "bold"), bg="#fff8e1").pack()
    
    sent_text = "POSITIVO" if sentiment_score > 0.05 else ("NEGATIVO" if sentiment_score < -0.05 else "NEUTRAL")
    sent_color = "green" if sentiment_score > 0.05 else ("red" if sentiment_score < -0.05 else "gray")
    
    Label(frame_data, text=f"Score de Noticias: {sentiment_score:.4f} ({sent_text})", 
          font=("Arial", 11, "bold"), fg=sent_color, bg="#fff8e1").pack()
    
    if external_factors:
        info_ext = f"VIX: {external_factors.get('VIX',0):.2f} | Cobre: ${external_factors.get('Cobre',0):.2f} | PEN: S/{external_factors.get('Dolar_Peru',0):.2f}"
        Label(frame_data, text=info_ext, font=("Arial", 10), bg="#fff8e1").pack(pady=2)

    # Decisión Grande
    color_dec = "#008000" if decision == "COMPRAR" else ("#CC0000" if decision == "VENDER" else "#FF8C00")
    Label(root, text=f"RECOMENDACIÓN: {decision}", font=("Arial", 20, "bold"), fg=color_dec).pack(pady=15)

    # Botón Ver Razones
    def open_reasons():
        top = Toplevel(root)
        top.title("Explicación Detallada")
        top.geometry("600x350")
        Label(top, text="Factores de Decisión (Fusión de Datos):", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Íconos simples con texto
        for r in reasons:
            bullet = "•"
            if "Sentimiento" in r: bullet = "📰"
            elif "VIX" in r: bullet = "📊"
            elif "Cobre" in r: bullet = "🏗️"
            elif "Modelo" in r: bullet = "📈"
            
            Label(top, text=f"{bullet} {r}", anchor="w", justify=LEFT, wraplength=550, font=("Arial", 10)).pack(fill=X, padx=15, pady=2)
            
        Button(top, text="Cerrar", command=top.destroy).pack(pady=10)

    Button(root, text="Ver Razones Detalladas", command=open_reasons, bg="#dddddd", width=25, font=("Arial", 10)).pack(pady=2)
    Button(root, text="Cerrar Sistema", command=root.destroy, width=20, fg="red").pack(pady=10)
    
    root.mainloop()

def print_decision(decision, reasons, last_price, predicted_price, change_pct, horizon_name="días"):
    """Imprime la decisión en consola"""
    print(f"\n" + "="*40)
    print(f"=== INFORME DE DECISIÓN: {TICKER} ===")
    print(f"="*40)
    print(f"Precio Actual:   {last_price:.2f} USD")
    print(f"Objetivo ({horizon_name}): {predicted_price:.2f} USD")
    print(f"Retorno Esp.:    {change_pct*100:+.2f}%")
    print(f"-"*40)
    print(f"RECOMENDACIÓN:   >> {decision} <<")
    print(f"-"*40)
    print("Justificación (Fusión de Fuentes):")
    for r in reasons:
        print(f"  • {r}")
    print("="*40 + "\n")

def print_hourly_prediction(last_price, hourly_predicted_price, hourly_change_pct, hourly_pred_std):
    """Imprime predicción horaria"""
    pass