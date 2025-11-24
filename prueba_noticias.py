"""
prueba_noticias.py
Script para diagnosticar y extraer correctamente las noticias de Yahoo Finance.
"""
import yfinance as yf
import json

def probar_noticias():
    print("📡 Conectando a Yahoo Finance para descargar noticias de META...")
    
    try:
        # Descargar noticias
        ticker = yf.Ticker("META")
        news = ticker.news
        
        if not news:
            print("❌ No se encontraron noticias (lista vacía).")
            return

        print(f"✅ Se encontraron {len(news)} noticias.")
        
        # Tomamos la primera noticia para examinarla
        first_item = news[0]
        
        print("\n🔎 --- INSPECCIÓN DE LA ESTRUCTURA ---")
        print(f"Llaves principales: {list(first_item.keys())}")
        
        # INTENTO DE EXTRACCIÓN
        title = ""
        
        # Caso 1: Estructura antigua (directa)
        if 'title' in first_item:
            print("👉 Estructura detectada: TIPO A (Directa)")
            title = first_item['title']
            
        # Caso 2: Estructura nueva (anidada en 'content')
        elif 'content' in first_item:
            print("👉 Estructura detectada: TIPO B (Anidada en 'content')")
            content_data = first_item['content']
            
            # A veces 'content' es un diccionario, a veces es otra cosa. Verificamos.
            if isinstance(content_data, dict):
                print(f"   Llaves dentro de 'content': {list(content_data.keys())}")
                title = content_data.get('title', 'NO TITULO EN CONTENT')
            else:
                print(f"   ⚠️ 'content' no es diccionario, es: {type(content_data)}")
        
        print(f"\n📰 TITULO EXTRAÍDO: '{title}'")
        
        if title:
            print("\n🎉 ¡ÉXITO! El script ya sabe dónde buscar.")
        else:
            print("\n⚠️ Aún no encontramos el título. Mándame el 'Raw Data' de abajo:")
            print(json.dumps(first_item, indent=2, default=str))

    except Exception as e:
        print(f"❌ Error crítico: {e}")

if __name__ == "__main__":
    probar_noticias()