from big_data import run_big_data_pipeline
import pandas as pd
import sys

# Configurar pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("=== INICIANDO PRUEBA DE INTEGRACIÓN BIG DATA (GDELT + DUCKDB) ===")
print("Ejecutando pipeline...")

try:
    # Ejecutar la función principal
    df = run_big_data_pipeline()
    
    print("\n" + "="*50)
    print("REPORTE DE EJECUCIÓN")
    print("="*50)
    
    if df is not None and not df.empty:
        print(f"✅ ÉXITO: Se obtuvo un DataFrame con {len(df)} filas.")
        print("\n--- Muestra de Datos ---")
        print(df)
        print("\n--- Tipos de Datos ---")
        print(df.dtypes)
        print("\n--- Estadísticas Básicas ---")
        print(df.describe())
    else:
        print("⚠️ AVISO: El pipeline finalizó sin errores pero el DataFrame está vacío.")
        print("Posibles causas: No hay noticias de META en los días descargados o fallo en filtrado.")

except Exception as e:
    print(f"\n❌ ERROR CRÍTICO DURANTE LA EJECUCIÓN: {e}")
    import traceback
    traceback.print_exc()

print("\n=== FIN DE LA PRUEBA ===")
