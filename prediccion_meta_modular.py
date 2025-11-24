"""
prediccion_meta_modular.py
Sistema modular para predicción de META con ejecución por bloques
Versión PRO: Incluye Inteligencia de Negocios 360° y Big Data
"""

import os
import warnings
warnings.filterwarnings("ignore")

# Importar módulos
from config import *
from data_loader import download_data, compute_technical_features, clear_data_cache
from analysis import analyze_basic_stats, compare_recent_vs_historical, get_data_summary

# --- CORRECCIÓN CRÍTICA AQUÍ ---
# Importamos 'train_model' y lo renombramos a 'train_model_func' para mantener compatibilidad
from model import train_model as train_model_func, predict_future_with_model, predict_hours_with_model, load_model, model_exists

from visualization import plot_price_and_forecast, plot_indicators, plot_basic_stats, plot_big_data_dashboard
from decision import decision_rule_and_reason, show_decision_window, print_decision
from market_intelligence import get_sentiment_analysis, get_external_factors

# Variables globales para mantener estado entre bloques
global_df = None
global_df_processed = None
global_model = None
global_metrics = None

def load_and_process_data(use_cache=True):
    """Carga y procesa los datos"""
    global global_df, global_df_processed
    
    print("\n=== CARGA Y PROCESAMIENTO DE DATOS ===")
    
    # Descargar datos
    df = download_data(use_cache=use_cache)
    print(f"Datos descargados: {len(df)} filas")
    print(f"Período: {df['Date'].iloc[0].date()} hasta {df['Date'].iloc[-1].date()}")
    
    # Procesar features
    df_processed = compute_technical_features(df)
    print(f"Datos procesados: {len(df_processed)} filas con features técnicos")
    
    global_df = df
    global_df_processed = df_processed
    
    return df, df_processed

def analyze_data():
    """Análisis básico de los datos"""
    global global_df_processed
    
    if global_df_processed is None:
        print("Primero debe cargar los datos (opción 1)")
        return
    
    print("\n=== ANÁLISIS DE DATOS ===")
    
    # Estadísticas básicas
    stats = analyze_basic_stats(global_df_processed)
    print("\nEstadísticas básicas:")
    for k, v in stats.items():
        print(f"  - {k}: {v:.2f}")
    
    # Comparación reciente vs histórico
    comp = compare_recent_vs_historical(global_df_processed)
    print("\nComparación (últimos 30 días vs general):")
    for k, v in comp.items():
        print(f"  - {k}: {v:.2f}")
    
    # Resumen de datos
    summary = get_data_summary(global_df_processed)
    print(f"\nResumen de datos:")
    print(f"  - Período: {summary['start_date']} a {summary['end_date']}")
    print(f"  - Total de filas: {summary['total_rows']}")
    print(f"  - Columnas: {len(summary['columns'])}")

def train_model():
    """Entrenar el modelo"""
    global global_df_processed, global_model, global_metrics
    
    if global_df_processed is None:
        print("Primero debe cargar los datos (opción 1)")
        return
    
    print("\n=== ENTRENAMIENTO DEL MODELO ===")
    
    # Usar la función train_model importada correctamente
    model, metrics, model_type = train_model_func(global_df_processed)
    
    if model_type == "prophet":
        print("✅ Modelo Prophet entrenado exitosamente")
        global_model = (model,)  # Para Prophet solo guardamos el modelo
    else:
        print("✅ Modelo Random Forest entrenado exitosamente")
        global_model = (model, metrics.get('feature_cols', []))
    
    print("\nMétricas en test:")
    for k, v in metrics.items():
        if k not in ['feature_cols', 'X_test', 'y_test', 'y_pred']:
            print(f"  - {k}: {v:.4f}")
    
    global_metrics = metrics
    
    return model, metrics, model_type

def generate_visualizations():
    """Generar visualizaciones"""
    global global_df_processed, global_metrics
    
    if global_df_processed is None:
        print("Primero debe cargar los datos (opción 1)")
        return
    
    if global_metrics is None:
        print("Primero debe entrenar el modelo (opción 3)")
        return
    
    print("\n=== GENERACIÓN DE VISUALIZACIONES ===")
    
    # Gráficas
    plot_basic_stats(global_df_processed)
    plot_indicators(global_df_processed)
    
    print("Visualizaciones generadas y guardadas en carpeta 'plots'")

def make_prediction():
    """Hacer predicción, análisis 360° y decisión final"""
    global global_df_processed, global_model, global_metrics
    
    if global_df_processed is None:
        print("❌ Primero debe cargar los datos (opción 1)")
        return
    
    if global_model is None:
        print("❌ Primero debe entrenar el modelo (opción 3)")
        return
    
    print("\n" + "="*60)
    print("🤖 INICIANDO SISTEMA DE INTELIGENCIA DE NEGOCIOS (BI)")
    print("="*60)
    
    # ---------------------------------------------------------
    # 1. MODELO MATEMÁTICO (TÉCNICO)
    # ---------------------------------------------------------
    if global_model is None and model_exists():
        global_model = load_model()
    
    model_type = "prophet" if len(global_model) == 1 else "random_forest"
    
    # Predicción a 7 días
    predicted_price, pred_std = predict_future_with_model(
        global_model, 
        global_df_processed, 
        horizon_days=HORIZON_DAYS,
        model_type=model_type
    )
    
    last_price = float(global_df_processed['Close'].iloc[-1])
    
    # Predicción Intradía (solo RF)
    if model_type == "random_forest":
        hourly_predicted_price, hourly_pred_std = predict_hours_with_model(
            global_model, global_df_processed
        )
        hourly_change_pct = (hourly_predicted_price - last_price) / last_price
    else:
        hourly_predicted_price, hourly_pred_std, hourly_change_pct = 0, 0, 0
    
    print(f"\n📊 ANÁLISIS TÉCNICO (MODELO PREDICTIVO):")
    print(f"   Precio actual:   {last_price:.2f} USD")
    print(f"   Objetivo (7d):   {predicted_price:.2f} USD")
    print(f"   Riesgo/Desv.Std: {pred_std:.4f}")

    if model_type == "random_forest":
        print(f"   Predicción Intradía (20h): {hourly_predicted_price:.2f} USD ({hourly_change_pct:.2%})")

    # ---------------------------------------------------------
    # 2. BIG DATA & INTELIGENCIA 360° (MEJORADO)
    # ---------------------------------------------------------
    print("\n📡 ESCANEANDO BIG DATA EN TIEMPO REAL (ANÁLISIS 360°)...")
    
    # A) Análisis de Narrativa y Tópicos (NLP)
    sentiment, headlines, keywords = get_sentiment_analysis(TICKER)
    
    # Determinar etiqueta de texto
    if sentiment > 0.05: sent_label = "POSITIVO (Optimismo)"
    elif sentiment < -0.05: sent_label = "NEGATIVO (Miedo)"
    else: sent_label = "NEUTRAL (Indecisión)"
    
    print(f"   📰 Sentiment Score: {sentiment:.4f} => {sent_label}")
    
    if keywords:
        print("   🔑 Tópicos Clave Detectados (Trending Topics):")
        topicos_str = ", ".join([f"{k[0].upper()}" for k in keywords])
        print(f"      👉 {topicos_str}")
    else:
        print("      (No se detectaron tópicos claros)")
        
    print("   Titulares Recientes:")
    for h in headlines[:3]:
        print(f"      - {h}")

    # B) Contexto Macroeconómico
    factors = get_external_factors()
    print(f"\n🌍 CONTEXTO MACROECONÓMICO GLOBAL:")
    print(f"   - VIX (Psicología de Masas): {factors.get('VIX', 0):.2f}")
    print(f"   - Cobre (Termómetro Económico): ${factors.get('Cobre', 0):.2f}")
    print(f"   - Oro (Refugio): ${factors.get('Oro', 0):.2f}")
    print(f"   - Dólar (PEN): S/ {factors.get('Dolar_Peru', 0):.2f}")

    # =========================================================
    # GENERACIÓN DE GRÁFICOS BIG DATA (DASHBOARD 360°)
    # =========================================================
    print("\n🎨 Renderizando Dashboard Visual de Big Data...")
    try:
        plot_path = plot_big_data_dashboard(sentiment, headlines, keywords, factors)
        
        # Intentar abrir la imagen automáticamente (Windows)
        import os
        os.startfile(plot_path) 
        print("   ✅ Dashboard abierto exitosamente.")
    except Exception as e:
        print(f"   ⚠️ Aviso: La imagen se guardó en 'plots/' pero no se pudo abrir sola ({e})")
    # =========================================================

    # ---------------------------------------------------------
    # 3. TOMA DE DECISIÓN FUSIONADA
    # ---------------------------------------------------------
    decision, reasons, change_pct = decision_rule_and_reason(
        last_price, predicted_price, pred_std, global_metrics, 
        sentiment_score=sentiment,
        external_factors=factors,
        horizon_name="7 días"
    )
    
    # Mostrar informe final
    print_decision(decision, reasons, last_price, predicted_price, change_pct, horizon_name="7 días")
    
    # ---------------------------------------------------------
    # 4. INTERFAZ GRÁFICA DETALLADA
    # ---------------------------------------------------------
    respuesta = input("¿Desea mostrar la ventana de decisión detallada? (s/n): ")
    if respuesta.lower() == 's':
        show_decision_window(decision, reasons, last_price, predicted_price, change_pct, 
                             sentiment_score=sentiment,
                             external_factors=factors)

def show_menu():
    """Mostrar menú interactivo"""
    while True:
        print("\n" + "="*60)
        print("SISTEMA DE INTELIGENCIA DE NEGOCIOS - META (PRO)")
        print("="*60)
        print("1. Cargar y procesar datos")
        print("2. Análisis de datos")
        print("3. Entrenar modelo")
        print("4. Generar visualizaciones")
        print("5. Hacer predicción y decisión (BI Completo)")
        print("6. Limpiar cache de datos")
        print("7. Salir")
        print("-"*60)
        
        opcion = input("Seleccione una opción (1-7): ")
        
        if opcion == '1':
            use_cache = input("¿Usar cache? (s/n): ").lower() == 's'
            load_and_process_data(use_cache=use_cache)
            
        elif opcion == '2':
            analyze_data()
            
        elif opcion == '3':
            train_model()
            
        elif opcion == '4':
            generate_visualizations()
            
        elif opcion == '5':
            make_prediction()
            
        elif opcion == '6':
            clear_data_cache()
            
        elif opcion == '7':
            print("Saliendo del sistema...")
            break
            
        else:
            print("Opción no válida. Intente nuevamente.")
        
        input("\nPresione Enter para continuar...")

if __name__ == "__main__":
    # Crear carpeta para plots si no existe
    os.makedirs(PLOT_PATH, exist_ok=True)
    
    print("Iniciando Sistema de Predicción META...")
    print("Módulos cargados: Big Data, NLP, Macroeconomía")
    
    show_menu()