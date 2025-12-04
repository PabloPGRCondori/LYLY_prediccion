"""
research.py - Módulo de investigación usando Perplexity AI
"""

import os
from config import PERPLEXITY_API_KEY

def get_financial_news(ticker, max_results=3):
    """
    Busca noticias financieras recientes y análisis para un ticker específico usando Perplexity AI.
    """
    
    # Verificar API Key
    api_key = PERPLEXITY_API_KEY or os.getenv("PERPLEXITY_API_KEY")
    
    if not api_key:
        return [
            {
                "title": "⚠️ API Key no configurada",
                "url": "#",
                "snippet": "Configura la variable de entorno PERPLEXITY_API_KEY para obtener noticias en tiempo real."
            }
        ]

    try:
        # Intentar importar la librería instalada
        try:
            from perplexity import Perplexity
        except ImportError:
            return [{
                "title": "⚠️ Librería faltante",
                "url": "#",
                "snippet": "La librería 'perplexityai' no está instalada correctamente."
            }]

        # Inicializar cliente
        client = Perplexity(api_key=api_key)

        # Crear query específico
        query = f"Latest important financial news, market sentiment, and analyst ratings for {ticker} stock today. Summarize key factors influencing the price."

        # Ejecutar búsqueda
        search = client.search.create(
            query=query,
            max_results=max_results,
            search_domain_filter=["-reddit.com", "-twitter.com"], # Filtro profesional
            max_tokens_per_page=512
        )
        
        results = []
        for result in search.results:
            results.append({
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet
            })
            
        return results

    except Exception as e:
        print(f"❌ Error en Perplexity Research: {e}")
        return [
            {
                "title": "Error en la búsqueda",
                "url": "#",
                "snippet": f"No se pudieron obtener noticias: {str(e)}"
            }
        ]

if __name__ == "__main__":
    # Prueba rápida si se ejecuta directamente
    print("Probando investigación para META...")
    news = get_financial_news("META")
    for n in news:
        print(f"- {n['title']}: {n['snippet'][:100]}...")
