"""
Dashboard de Suivi et Monitoring Temps Réel pour Bot de Trading TimesFM (ETH 1h).
Construit avec Streamlit & Plotly pour une expérience visuelle haut de gamme.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime

from config.settings import config
from src.screening.momentum_screener import MomentumScreener
from src.models.timesfm_engine import TimesFMEngine
from src.risk.risk_manager import RiskManager

# Configuration de la page Streamlit
st.set_page_config(
    page_title="TimesFM Quant Trading Terminal | ETH 1h",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personnalisé pour un rendu Terminal Quant Premium
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #1E222D; padding: 15px; border-radius: 10px; border: 1px solid #2A2E39; }
    .status-buy { color: #00E676; font-weight: bold; font-size: 1.2rem; }
    .status-sell { color: #FF5252; font-weight: bold; font-size: 1.2rem; }
    .card { background-color: #1E222D; padding: 20px; border-radius: 12px; border: 1px solid #2A2E39; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

from src.data_loader import get_large_eth_data

@st.cache_data(ttl=300)
def load_market_data(symbol: str, days: int = 90):
    return get_large_eth_data(symbol="ETH/USDT", timeframe="1h", days_back=days, force_refresh=False)


def main():
    st.title("⚡ Terminal Quantitatif TimesFM — Ethereum (ETH 1h)")
    st.caption("Modèle de fondation IA de séries temporelles & Sizing Dynamique de Kelly")
    
    # Sidebar
    st.sidebar.header("⚙️ Paramètres du Bot")
    symbol = st.sidebar.selectbox("Paire de Trading", ["ETH-USD", "ETH-EUR", "BTC-USD"], index=0)
    lookback_days = st.sidebar.slider("Période Historique (Jours)", 7, 60, 30)
    capital_input = st.sidebar.number_input("Capital Total (€/$)", min_value=100.0, value=10000.0, step=500.0)
    risk_pct_input = st.sidebar.slider("Risque Fixe par Trade (%)", 0.5, 5.0, 2.0) / 100.0
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Garde-fous Quantitatifs")
    st.sidebar.write(f"• **Seuil RVOL** : `{config.rvol_threshold}x`")
    st.sidebar.write(f"• **Fraction Kelly** : `{config.max_kelly_fraction * 100}%` (Quarter Kelly)")
    st.sidebar.write(f"• **Plafond Enveloppe** : `{config.max_portfolio_allocation * 100}%` du capital")
    
    # Bouton de rafraîchissement
    if st.sidebar.button("🔄 Lancer Inférence Temps Réel"):
        st.cache_data.clear()

    # Ingestion & Calculs
    with st.spinner("Inférence TimesFM & Ingestion des données en cours..."):
        df = load_market_data(symbol, days=lookback_days)
        
        screener = MomentumScreener(rvol_threshold=config.rvol_threshold)
        df_screened = screener.compute_indicators(df)
        latest_screen = screener.evaluate_latest(df_screened)
        
        timesfm_engine = TimesFMEngine(context_len=config.context_len, horizon_len=config.horizon_len, backend=config.backend)
        signal = timesfm_engine.generate_signal(df_screened, screener_passed=latest_screen['passed'])
        
        risk_mgr = RiskManager(default_risk_pct=risk_pct_input)
        position_info = risk_mgr.compute_position_size(
            total_capital=capital_input,
            entry_price=signal['current_price'],
            df_ohlcv=df_screened,
            signal_dict=signal
        )

    # 1. Rangée des KPI Principaux
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label="Prix Actuel ETH", value=f"${signal['current_price']:,.2f}")
        
    with col2:
        pred_delta = signal['predicted_return_pct']
        st.metric(
            label="Prédiction TimesFM (H+1)",
            value=f"${signal['predicted_price']:,.2f}",
            delta=f"{pred_delta:+.2f}%"
        )
        
    with col3:
        sig_label = signal['signal_label']
        st.metric(
            label="Signal Binaire IA",
            value=sig_label,
            delta="ACHAT (LONG)" if signal['signal_binary'] == 1 else "NEUTRE / CONSERVATION"
        )
        
    with col4:
        rvol_val = latest_screen['rvol']
        st.metric(
            label="Volume Relatif (RVOL)",
            value=f"{rvol_val:.2f}x",
            delta="Volume d'activité validé" if latest_screen['rvol_ok'] else "Volume standard"
        )
        
    with col5:
        allocated = position_info['capital_allocated']
        st.metric(
            label="Sizing Capital Alloué",
            value=f"{allocated:,.2f} €",
            delta=f"{allocated/capital_input*100:.1f}% du Portefeuille"
        )

    # 2. Graphiques d'Analyse Interactive
    tab1, tab2, tab3 = st.tabs(["📈 Graphique Prix & Inférence TimesFM", "🧮 Détail Risk Management & Kelly", "📊 Backtest & Historique"])
    
    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
        
        # Chandeliers OHLC
        fig.add_trace(
            go.Candlestick(
                x=df_screened.index,
                open=df_screened['Open'], high=df_screened['High'],
                low=df_screened['Low'], close=df_screened['Close'],
                name="ETH/USD 1h"
            ), row=1, col=1
        )
        
        # SMAs
        fig.add_trace(go.Scatter(x=df_screened.index, y=df_screened['SMA_50'], name="SMA 50h", line=dict(color='orange', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_screened.index, y=df_screened['SMA_200'], name="SMA 200h", line=dict(color='red', width=1.5)), row=1, col=1)
        
        # Point de prédiction H+1
        next_time = df_screened.index[-1] + pd.Timedelta(hours=1)
        fig.add_trace(
            go.Scatter(
                x=[next_time],
                y=[signal['predicted_price']],
                mode='markers+text',
                name="TimesFM Target (H+1)",
                marker=dict(size=12, color='cyan', symbol='star'),
                text=[f"Target: ${signal['predicted_price']:.1f}"],
                textposition="top center"
            ), row=1, col=1
        )
        
        # Sous-Graphique Volume & RVOL
        colors = ['green' if r > config.rvol_threshold else 'gray' for r in df_screened['RVOL']]
        fig.add_trace(go.Bar(x=df_screened.index, y=df_screened['Volume'], name="Volume 1h", marker_color=colors), row=2, col=1)
        
        fig.update_layout(
            template="plotly_dark",
            height=600,
            title="Analyse Chartiste ETH 1h avec Inférence TimesFM H+1",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Détails du Calcul de Position Sizing & Protection du Capital")
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("### 📐 Niveaux de Trading Identifiés")
            st.write(f"• **Prix d'Entrée** : `${position_info['entry_price']:,.2f}`")
            st.write(f"• **Stop-Loss Dynamique (2x ATR)** : `${position_info['stop_loss_price']:,.2f}`")
            st.write(f"• **Take-Profit Dynamique (3.5x ATR)** : `${position_info['take_profit_price']:,.2f}`")
            st.write(f"• **Ratio Risque / Rendement (R/R)** : `{position_info.get('risk_reward_ratio', 1.75):.2f}`")
            st.write(f"• **Volatilité ATR(14)** : `${position_info.get('atr', 0):.2f}`")

        with c2:
            st.write("### 🛡️ Dynamic Kelly Allocation & Safeties")
            st.write(f"• **Quantité d'ETH à acheter** : `{position_info['quantity_units']:.4f} ETH`")
            st.write(f"• **Capital Total Engagé** : `{position_info['capital_allocated']:,.2f} €`")
            st.write(f"• **Perte Maximale autorisée (Risk Amount)** : `{position_info['max_risk_amount']:,.2f} €` (`{position_info['risk_pct_used']:.2f}%` du capital)")
            st.write(f"• **Plafond Enveloppe Respecté** : `{'✅ OUI' if position_info['envelope_safety_passed'] else '❌ NON'}`")

    with tab3:
        st.subheader("Historique des Signaux & Test Directionnel")
        st.write("Évaluation des 10 dernières bougies 1h :")
        
        recent_records = []
        for i in range(-10, 0):
            sub = df_screened.iloc[:i]
            if len(sub) > 50:
                p_curr = sub['Close'].iloc[-1]
                p_next = df_screened['Close'].iloc[i+1] if i+1 < 0 else df_screened['Close'].iloc[-1]
                direction = "HAUSSE" if p_next > p_curr else "BAISSE"
                recent_records.append({
                    "Timestamp": sub.index[-1],
                    "Prix ETH": f"${p_curr:,.2f}",
                    "Prix Suivant H+1": f"${p_next:,.2f}",
                    "Mouvement": direction,
                    "RVOL": f"{sub['RVOL'].iloc[-1]:.2f}x",
                    "Trend OK": sub['Trend_OK'].iloc[-1]
                })
        st.dataframe(pd.DataFrame(recent_records), use_container_width=True)

if __name__ == "__main__":
    main()
