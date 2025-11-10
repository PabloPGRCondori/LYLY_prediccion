"""
decision.py - Funciones de decisión y interfaz de usuario
"""

from tkinter import Tk, Label, Button, Toplevel, LEFT, RIGHT, BOTH, X
from config import *

def decision_rule_and_reason(last_price, predicted_price, pred_std, metrics, horizon_name="días"):
    """
    Regla simple:
    - Si predicho > last * 1.02 => Comprar
    - Si predicho < last * 0.98 => Vender
    - else => Mantener
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
        reasons.append(f"Predicción positiva a {horizon_name}: cambio esperado ~ {change_pct*100:.2f}%.")
    else:
        reasons.append(f"Predicción negativa a {horizon_name}: cambio esperado ~ {change_pct*100:.2f}%.")

    if decision == "MANTENER" and not reasons:
        reasons.append(f"Cambio pequeño esperado a {horizon_name}, no hay señal clara.")

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

def print_decision(decision, reasons, last_price, predicted_price, change_pct, horizon_name="días"):
    """Imprime la decisión en consola"""
    print(f"\n=== DECISIÓN DE INVERSIÓN ===")
    print(f"Precio actual: {last_price:.2f} USD")
    print(f"Predicción a {horizon_name}: {predicted_price:.2f} USD ({change_pct*100:+.2f}%)")
    print(f"RECOMENDACIÓN: {decision}")
    print("\nRazones:")
    for r in reasons:
        print(f"  • {r}")

def print_hourly_prediction(last_price, hourly_predicted_price, hourly_change_pct, hourly_pred_std):
    """Imprime la predicción horaria en consola"""
    print(f"\n📈 PREDICCIÓN A {HORIZON_HOURS} HORAS:")
    print(f"   Precio actual: {last_price:.2f} USD")
    print(f"   Predicción: {hourly_predicted_price:.2f} USD ({hourly_change_pct*100:+.2f}%)")
    print(f"   Incertidumbre: ±{hourly_pred_std:.2f} USD")
    
    # Decisión simple para horizonte corto
    if hourly_change_pct > 0.005:  # +0.5%
        print(f"   📊 Tendencia: ALZA esperada")
    elif hourly_change_pct < -0.005:  # -0.5%
        print(f"   📊 Tendencia: BAJA esperada")
    else:
        print(f"   📊 Tendencia: LATERAL esperada")