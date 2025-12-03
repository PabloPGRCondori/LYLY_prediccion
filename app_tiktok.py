import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Importar lógica del backend
from data_loader import download_data, compute_technical_features
from model import train_model, load_model
from big_data import run_big_data_pipeline
from config import TICKERS

# Configuración de la página
st.set_page_config(
    page_title="LYLY - TikTok Financiero",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Estilos CSS para simular feed de TikTok (Vertical, centrado, tarjetas oscuras)
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max_width: 600px;
    }
    .stCard {
        background-color: #1E1E1E;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 30px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1, h2, h3 {
        color: #FFFFFF !important;
    }
    p, div {
        color: #E0E0E0;
    }
    .metric-label {
        color: #888 !important;
        font-size: 0.9rem;
    }
    .metric-value {
        font-size: 2rem !important;
        font-weight: bold;
    }
    .highlight-green {
        color: #4CAF50;
        font-weight: bold;
    }
    .highlight-red {
        color: #FF5252;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- FUNCIONES CACHEADAS PARA PERFORMANCE ---
@st.cache_data(ttl=3600)
def get_ticker_data(ticker):
    try:
        df = download_data(ticker=ticker, use_cache=True)
        if df is None or df.empty: return None
        df_processed = compute_technical_features(df)
        return df_processed
    except Exception as e:
        print(f"Error getting data for {ticker}: {e}")
        return None

@st.cache_resource
def get_model_for_ticker(df_processed):
    # Entrenar o cargar modelo
    model, metrics, model_type = train_model(df_processed)
    return model, metrics, model_type

# --- ESTRUCTURA DEL FEED ---

st.title("📱 LYLY Feed")
st.caption("Desliza para ver el mercado")

# --- LOOP PRINCIPAL (FEED INFINITO SIMULADO) ---
for ticker in TICKERS:
    with st.container():
        # Cargar datos
        df = get_ticker_data(ticker)
        
        if df is None:
            continue # Saltar si falla la descarga
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calcular cambio porcentual diario
        daily_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
        color = "#4CAF50" if daily_change >= 0 else "#FF5252"
        
        # --- TARJETA DE TICKER ---
        st.markdown(f'<div class="stCard" style="border-left: 5px solid {color};">', unsafe_allow_html=True)
        
        # Encabezado: Ticker + Precio
        col1, col2 = st.columns([2, 1])
        with col1:
             st.markdown(f"## {ticker}")
             st.markdown(f"<span class='metric-value'>${latest['Close']:.2f}</span> <span style='color:{color}'>({daily_change:+.2f}%)</span>", unsafe_allow_html=True)
             st.caption(f"Vol: {latest['Volume']:,}")
        with col2:
             # Gráfico sparkline
             spark_df = df.tail(30)
             fig_spark = px.line(spark_df, x='Date', y='Close', template="plotly_dark")
             fig_spark.update_layout(
                 margin=dict(l=0, r=0, t=0, b=0),
                 xaxis=dict(visible=False),
                 yaxis=dict(visible=False),
                 height=60,
                 paper_bgcolor='rgba(0,0,0,0)',
                 plot_bgcolor='rgba(0,0,0,0)'
             )
             fig_spark.update_traces(line_color=color, line_width=2)
             st.plotly_chart(fig_spark, use_container_width=True, config={'displayModeBar': False})
        
        # Acción Recomendada
        rsi = latest['RSI_14']
        macd = latest['MACD']
        signal = latest['MACD_signal']
        
        recommendation = "MANTENER ⚪"
        reason = "Mercado lateral"
        
        if rsi < 30: 
            recommendation = "COMPRA FUERTE 🟢"
            reason = "Sobreventa (RSI < 30)"
        elif rsi > 70: 
            recommendation = "VENTA FUERTE 🔴"
            reason = "Sobrecompra (RSI > 70)"
        elif macd > signal: 
            recommendation = "COMPRAR 🟢"
            reason = "Tendencia alcista (MACD)"
        elif macd < signal: 
            recommendation = "VENDER 🔴"
            reason = "Tendencia bajista (MACD)"
            
        st.markdown(f"### {recommendation}")
        st.caption(f"Razón: {reason}")
        
        # Expander para detalles y predicción
        with st.expander(f"📊 Ver análisis detallado de {ticker}"):
            
            # Entrenar modelo rápido para métricas
            with st.spinner(f"Analizando {ticker} con IA..."):
                model, metrics, m_type = get_model_for_ticker(df)
            
            accuracy = metrics.get('R2', 0) * 100
            st.progress(min(max(int(accuracy), 0), 100))
            st.caption(f"Confianza del Modelo ({m_type}): {accuracy:.1f}% (R²)")
            
            # Gráfico de Bandas de Bollinger
            chart_data = df.tail(100)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=chart_data['Date'], y=chart_data['Close'], mode='lines', name='Precio', line=dict(color='white')))
            fig.add_trace(go.Scatter(x=chart_data['Date'], y=chart_data['BB_upper'], mode='lines', name='Upper BB', line=dict(color='gray', dash='dot')))
            fig.add_trace(go.Scatter(x=chart_data['Date'], y=chart_data['BB_lower'], mode='lines', name='Lower BB', line=dict(color='gray', dash='dot'), fill='tonexty'))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0),
                height=300,
                showlegend=False,
                title="Bandas de Bollinger"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **Indicadores:**
            - RSI: {rsi:.2f}
            - Volatilidad: {latest['Volatility_20']:.4f}
            """)

        st.markdown('</div>', unsafe_allow_html=True)

# --- BOTÓN FLOTANTE O FINAL ---
st.markdown("---")
if st.button("🔄 Actualizar Todo"):
    st.cache_data.clear()
    st.rerun()

st.caption("LYLY Prediction System v2.1 - Multi-Asset Feed")