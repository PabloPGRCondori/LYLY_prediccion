"""
market_intelligence.py - Módulo Avanzado de Inteligencia de Mercado
Incluye: NLP, Extracción de Tópicos y Factores Macro
"""
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re

# Inicializar analizador
analyzer = SentimentIntensityAnalyzer()

# Lista de palabras vacías (stopwords) en inglés para limpiar los temas
STOPWORDS = set([
    'the', 'to', 'in', 'of', 'and', 'a', 'for', 'is', 'on', 'that', 'with', 
    'as', 'at', 'by', 'this', 'are', 'be', 'from', 'it', 'or', 'an', 'its', 
    'has', 'stock', 'market', 'stocks', 'why', 'how', 'what', 'after', 'shares'
])

def extract_top_keywords(headlines):
    """
    Analiza de qué se está hablando en las noticias.
    Retorna: Lista de (Palabra, Frecuencia)
    """
    all_text = " ".join(headlines).lower()
    # Limpiar caracteres no alfanuméricos
    words = re.findall(r'\b[a-z]{3,}\b', all_text)
    # Filtrar stopwords
    meaningful_words = [w for w in words if w not in STOPWORDS]
    # Contar
    counter = Counter(meaningful_words)
    return counter.most_common(5)

def get_sentiment_analysis(ticker_symbol):
    """
    Realiza un análisis profundo de noticias: Score + Tópicos.
    """
    print(f"   🔎 Escaneando narrativa de mercado para {ticker_symbol}...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        news_list = ticker.news
        
        if not news_list:
            return 0.0, [], []
        
        total_score = 0
        headlines_text = []
        scored_headlines = []
        count = 0

        for item in news_list:
            # Estrategia de extracción robusta (Content > Title > Headline)
            title = item.get('title')
            if not title and 'content' in item and isinstance(item['content'], dict):
                title = item['content'].get('title') or item['content'].get('summary')
            if not title:
                continue 

            # Análisis VADER
            vs = analyzer.polarity_scores(title)
            compound = vs['compound']
            
            total_score += compound
            headlines_text.append(title)
            
            # Formatear para mostrar
            clean_title = title.replace("\n", " ").strip()
            scored_headlines.append(f"({compound:+.2f}) {clean_title[:80]}...")
            count += 1
            
        if count == 0:
            return 0.0, [], []

        avg_score = total_score / count
        
        # --- NUEVO: Extraer Tópicos ---
        top_keywords = extract_top_keywords(headlines_text)
        
        return avg_score, scored_headlines[:5], top_keywords
        
    except Exception as e:
        print(f"⚠️ Error en análisis de inteligencia: {e}")
        return 0.0, ["Error de conexión"], []

def get_external_factors():
    """
    Obtiene Commodities (Oro, Cobre), Moneda (PEN) y Riesgo (VIX).
    """
    print("   🌍 Analizando Factores Macro Globales...")
    factors = {'Dolar_Peru': 0, 'Oro': 0, 'Cobre': 0, 'VIX': 0}
    try:
        tickers = ["PEN=X", "GC=F", "HG=F", "^VIX"]
        data = yf.download(tickers, period="1d", progress=False)
        
        if not data.empty:
            closes = data['Close'].iloc[-1]
            factors = {
                'Dolar_Peru': float(closes.get('PEN=X', 0)),
                'Oro': float(closes.get('GC=F', 0)),
                'Cobre': float(closes.get('HG=F', 0)),
                'VIX': float(closes.get('^VIX', 0))
            }
    except Exception as e:
        print(f"⚠️ Error obteniendo macro-datos: {e}")
    
    return factors