"""
prediccion_meta_modular.py
Sistema modular para predicción de META con ejecución por bloques

Ejecutar con: py prediccion_meta_modular.py
"""

import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

# Importar módulos
from config import *
from data_loader import download_data, compute_technical_features, clear_data_cache
from analysis import analyze_basic_stats, compare_recent_vs_historical, get_data_summary
from model import train_model, load_model, predict_future_with_model, predict_future_direction, predict_hours_with_model, model_exists
from visualization import plot_price_and_forecast, plot_indicators, plot_basic_stats, plot_sentiment_vs_price
from decision import decision_rule_and_reason, show_decision_window, print_decision
from big_data import run_big_data_pipeline

# Variables globales para compartir datos entre funciones
global_df = None
global_df_processed = None
global_model = None
global_metrics = None
global_big_data_df = None

def run_big_data():
    """Ejecuta el pipeline de Big Data"""
    global global_big_data_df, global_df
    
    print("\n=== PIPELINE DE BIG DATA (GDELT + DuckDB) ===")
    print("Este módulo descarga datos masivos de noticias globales y los procesa en memoria eficiente.")
    
    try:
        df = run_big_data_pipeline()
        if not df.empty:
            global_big_data_df = df
            print(f"\n✅ Big Data procesada exitosamente. {len(df)} registros listos para fusión.")
            
            # --- FASE 2: Visualización Automática ---
            print("\n=== GENERANDO VISUALIZACIÓN DE IMPACTO (Precios vs Noticias) ===")
            
            # Verificar si tenemos datos de precios cargados
            if global_df is None:
                print("Datos de precios no encontrados en memoria. Descargando historial reciente de META...")
                # Descargamos datos usando el cargador existente
                # No usamos cache para asegurar datos recientes alineados con GDELT
                global_df = download_data(use_cache=True) 
            
            if global_df is not None and not global_df.empty:
                plot_sentiment_vs_price(global_big_data_df, global_df)
                print("✅ Gráfico 'impacto_noticias_meta.png' generado en carpeta 'plots'.")
            else:
                print("⚠️ No se pudieron obtener datos de precios para generar el gráfico.")
            
            # Recargar y procesar datos para incluir Big Data en el set de entrenamiento
            print("\n🔄 Actualizando dataset de entrenamiento con Big Data...")
            load_and_process_data(use_cache=True)
                
        else:
            print("\n⚠️ No se obtuvieron datos de Big Data.")
    except Exception as e:
        print(f"\n❌ Error en pipeline de Big Data: {e}")

def load_and_process_data(use_cache=True):
    """Carga y procesa los datos"""
    global global_df, global_df_processed, global_big_data_df
    
    print("\n=== CARGA Y PROCESAMIENTO DE DATOS ===")
    
    # Descargar datos
    df = download_data(use_cache=use_cache)
    print(f"Datos descargados: {len(df)} filas")
    print(f"Período: {df['Date'].iloc[0].date()} hasta {df['Date'].iloc[-1].date()}")
    
    # Procesar features
    df_processed = compute_technical_features(df)
    print(f"Datos procesados: {len(df_processed)} filas con features técnicos")
    
    # INTEGRACIÓN BIG DATA (Fase 3)
    if global_big_data_df is not None:
        print("\n🔄 Integrando Big Data (GDELT) al dataset principal...")
        
        # Asegurar formato fecha
        df_processed['Date'] = pd.to_datetime(df_processed['Date'])
        global_big_data_df['Date'] = pd.to_datetime(global_big_data_df['Date'])
        
        # Merge (Left join para mantener todos los días de trading)
        df_merged = pd.merge(df_processed, global_big_data_df[['Date', 'Avg_Sentiment', 'News_Volume']], on='Date', how='left')
        
        # 1. Feature Engineering: Lagging (Sentiment_Lag1)
        # Regla: NO usar sentimiento de hoy para predecir hoy/mañana (evitar look-ahead bias de publicación)
        # Usamos el sentimiento de ayer (shift 1)
        df_merged['Sentiment_Lag1'] = df_merged['Avg_Sentiment'].shift(1)
        
        # 2. Imputation (Manejo de vacíos)
        # Rellenar con 0 (Neutral)
        df_merged['Avg_Sentiment'] = df_merged['Avg_Sentiment'].fillna(0)
        df_merged['News_Volume'] = df_merged['News_Volume'].fillna(0)
        df_merged['Sentiment_Lag1'] = df_merged['Sentiment_Lag1'].fillna(0)
        
        # Eliminar 'Avg_Sentiment' actual para asegurar que el modelo NO lo use
        # (Solo permitimos Sentiment_Lag1 y News_Volume)
        df_merged.drop(columns=['Avg_Sentiment'], inplace=True)
        
        df_processed = df_merged
        print(f"✅ Big Data integrado. Nuevos features: Sentiment_Lag1, News_Volume.")
        print("   (Avg_Sentiment eliminado del set de entrenamiento para evitar bias)")

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
    
    # Usar la nueva función train_model que decide entre Prophet y Random Forest
    model, metrics, model_type = train_model(global_df_processed)
    
    if model_type == "prophet":
        print("✅ Modelo Prophet entrenado exitosamente")
        global_model = (model,)  # Para Prophet solo guardamos el modelo
    else:
        print("✅ Modelo Random Forest entrenado exitosamente")
        global_model = (model, metrics.get('feature_cols', []))
        
        # --- FASE 3: Feature Importance ---
        try:
            feature_cols = metrics.get('feature_cols', [])
            
            # Obtener importancias
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'named_steps') and hasattr(model.named_steps['rf'], 'feature_importances_'):
                importances = model.named_steps['rf'].feature_importances_
            else:
                importances = None
                
            if importances is not None:
                # Crear DataFrame y ordenar
                feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importances})
                feat_imp = feat_imp.sort_values('importance', ascending=False).reset_index(drop=True)
                
                print("\n📊 Importancia de Variables (Top 10):")
                print(feat_imp.head(10).to_string(index=False))
                
                # Verificar Sentiment_Lag1
                if 'Sentiment_Lag1' in feat_imp['feature'].values:
                    rank = feat_imp[feat_imp['feature'] == 'Sentiment_Lag1'].index[0] + 1
                    imp_val = feat_imp[feat_imp['feature'] == 'Sentiment_Lag1']['importance'].values[0]
                    print(f"\n🔍 Sentiment_Lag1: Puesto #{rank} (Importancia: {imp_val:.4f})")
                else:
                    print("\n⚠️ Sentiment_Lag1 no fue usado en el modelo.")
        except Exception as e:
            print(f"No se pudo calcular importancia de variables: {e}")
    
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
    
    # Para alinear fechas del test
    test_start_idx = int(len(global_df_processed) * 0.8)
    y_test_indexed = global_df_processed.iloc[test_start_idx: test_start_idx + len(global_metrics.get('y_test', []))].reset_index(drop=True)
    
    # Gráficas
    plot_basic_stats(global_df_processed)
    plot_indicators(global_df_processed)
    
    print("Visualizaciones generadas y guardadas en carpeta 'plots'")

def make_prediction():
    """Hacer predicción y mostrar decisión"""
    global global_df_processed, global_model, global_metrics
    
    if global_df_processed is None:
        print("Primero debe cargar los datos (opción 1)")
        return
    
    if global_model is None:
        print("Primero debe entrenar el modelo (opción 3)")
        return
    
    print("\n=== PREDICCIÓN Y DECISIÓN ===")
    
    # Cargar modelo si no está en memoria
    if global_model is None and model_exists():
        global_model = load_model()
    
    # Determinar tipo de modelo
    model_type = "prophet" if len(global_model) == 1 else "random_forest"
    
    # Hacer predicción a 7 días
    predicted_price, pred_std = predict_future_with_model(
        global_model, 
        global_df_processed, 
        horizon_days=HORIZON_DAYS,
        model_type=model_type
    )
    
    last_price = float(global_df_processed['Close'].iloc[-1])
    
    # Hacer predicción a 20 horas (solo para Random Forest)
    if model_type == "random_forest":
        hourly_predicted_price, hourly_pred_std = predict_hours_with_model(
            global_model,
            global_df_processed
        )
        hourly_change_pct = (hourly_predicted_price - last_price) / last_price
    else:
        hourly_predicted_price, hourly_pred_std = None, None
        hourly_change_pct = 0
    
    print(f"Precio actual: {last_price:.2f} USD")
    print(f"Predicción a {HORIZON_DAYS} días: {predicted_price:.2f} USD")
    print(f"Desviación estándar: {pred_std:.4f}")
    
    # Mostrar predicción a 20 horas (solo si es Random Forest)
    if model_type == "random_forest":
        print(f"\nPredicción a 20 horas: {hourly_predicted_price:.2f} USD")
        print(f"Cambio porcentual: {hourly_change_pct:.2%}")
        print(f"Desviación estándar (20h): {hourly_pred_std:.4f}")
    else:
        print(f"\n⚠️  Predicción a 20 horas no disponible para Prophet")
    
    # Tomar decisión para 7 días
    decision, reasons, change_pct = decision_rule_and_reason(
        last_price, predicted_price, pred_std, global_metrics, horizon_name="7 días"
    )
    
    # Mostrar en consola
    print_decision(decision, reasons, last_price, predicted_price, change_pct, horizon_name="7 días")
    
    # Preguntar si mostrar ventana
    respuesta = input("\n¿Desea mostrar la ventana gráfica? (s/n): ")
    if respuesta.lower() == 's':
        show_decision_window(decision, reasons, last_price, predicted_price, change_pct)

def show_menu():
    """Mostrar menú interactivo"""
    while True:
        print("\n" + "="*60)
        print("SISTEMA MODULAR DE PREDICCIÓN - META")
        print("="*60)
        print("1. Cargar y procesar datos")
        print("2. Análisis de datos")
        print("3. Entrenar modelo")
        print("4. Generar visualizaciones")
        print("5. Hacer predicción y decisión")
        print("6. Limpiar cache de datos")
        print("7. Pipeline Big Data (Experimental)")
        print("8. Salir")
        print("-"*60)
        
        opcion = input("Seleccione una opción (1-8): ")
        
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
            run_big_data()
            
        elif opcion == '8':
            print("Saliendo del sistema...")
            break
            
        else:
            print("Opción no válida. Intente nuevamente.")
        
        input("\nPresione Enter para continuar...")

if __name__ == "__main__":
    # Crear carpeta para plots si no existe
    os.makedirs(PLOT_PATH, exist_ok=True)
    
    print("Sistema modular de predicción para META")
    print("Este sistema permite ejecutar por bloques para reducir carga del sistema")
    
    show_menu()