"""
Terminal Quantitatif Web Ultra-Pro — Single-Page Responsive Dashboard.
Design d'exception Glassmorphism & High-End Trading UI pour Bot de Trading TimesFM (ETH 5m / 1h).
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

from config.settings import config
from src.screening.momentum_screener import MomentumScreener
from src.models.timesfm_engine import TimesFMEngine
from src.models.meta_labeler import MetaLabeler
from src.risk.risk_manager import RiskManager
from src.execution.alpaca_executor import AlpacaExecutor
from src.data_loader import get_large_eth_data

# 1. Configuration Page Streamlit Single-Page
st.set_page_config(
    page_title="TimesFM Quant Trading Suite | Pro Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Design System & CSS Personnalisé Pro (Glassmorphism & Responsive Layout)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #151c2c 0%, #0b0e14 100%);
        color: #E2E8F0;
    }
    
    /* En-tête Compact */
    .header-box {
        background: rgba(22, 30, 46, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 20px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(21, 28, 44, 0.65);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 229, 255, 0.4);
        transform: translateY(-2px);
    }
    
    .metric-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 1.35rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    
    .metric-badge {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-top: 4px;
    }
    
    .badge-buy { background: rgba(0, 230, 118, 0.15); color: #00E676; border: 1px solid rgba(0, 230, 118, 0.3); }
    .badge-hold { background: rgba(255, 215, 0, 0.15); color: #FFD700; border: 1px solid rgba(255, 215, 0, 0.3); }
    .badge-sell { background: rgba(255, 82, 82, 0.15); color: #FF5252; border: 1px solid rgba(255, 82, 82, 0.3); }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(22, 30, 46, 0.5);
        border-radius: 8px;
        padding: 6px 16px;
        font-weight: 600;
        color: #94A3B8;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #00E5FF !important;
        color: #0B0E14 !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def fetch_cached_data():
    return get_large_eth_data(symbol="ETH/USDT", timeframe="5m", days_back=30, force_refresh=False)

def main():
    # En-tête Compact Pro
    st.markdown("""
    <div class="header-box">
        <div>
            <span style="font-size: 1.4rem; font-weight: 800; color: #F8FAFC;">⚡ TERMINAL QUANTITATIF IA SOTA</span>
            <span style="font-size: 0.85rem; color: #94A3B8; margin-left: 10px;">• TimesFM Fine-Tuned + XGBoost Meta-Labeler | ETH/USD (5m)</span>
        </div>
        <div>
            <span class="metric-badge badge-buy">PRO VERSION 2.0</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Ingestion & Calculs Quant
    df = fetch_cached_data()
    current_price = float(df['Close'].iloc[-1])
    
    screener = MomentumScreener(rvol_threshold=config.rvol_threshold)
    df_screened = screener.compute_indicators(df)
    latest_screen = screener.evaluate_latest(df_screened)
    
    engine = TimesFMEngine(context_len=config.context_len, horizon_len=config.horizon_len, backend=config.backend)
    signal = engine.generate_signal(df_screened, screener_passed=latest_screen['passed'])
    
    meta_labeler = MetaLabeler()
    meta_confidence = meta_labeler.predict_meta_confidence(df_screened, timesfm_pred_return=signal['predicted_return_pct']/100.0)
    meta_passed = meta_confidence >= config.min_meta_confidence
    
    final_binary = 1 if (signal['signal_binary'] == 1 and meta_passed) else 0
    
    budget_input = 20000.0
    cap_max_input = 5000.0
    
    risk_mgr = RiskManager(default_risk_pct=config.risk_per_trade)
    pos_info = risk_mgr.compute_position_size(
        total_capital=budget_input,
        entry_price=current_price,
        df_ohlcv=df_screened,
        signal_dict=signal,
        historical_win_rate=max(0.60, meta_confidence)
    )
    
    allocated_cap = min(pos_info['capital_allocated'], cap_max_input) if final_binary == 1 else 0.0
    units = allocated_cap / current_price if current_price > 0 else 0.0

    # 1. RANGEE KPIS RESPONSIVES SINGLE-PAGE (5 Colonnes)
    k1, k2, k3, k4, k5 = st.columns(5)
    
    with k1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">BUDGET INITIAL</div>
            <div class="metric-value">{budget_input:,.2f} €</div>
            <div class="metric-badge badge-hold">CAPITAL GLOBAL</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">PRIX ETH/USD ACTUEL</div>
            <div class="metric-value">${current_price:,.2f}</div>
            <div class="metric-badge badge-hold">VOLATILITÉ ATR: ${pos_info.get('atr', 7.8):.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k3:
        pred_ret = signal['predicted_return_pct']
        pred_color = "#00E676" if pred_ret > 0 else "#FF5252"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">PRÉDICTION TIMESFM H+1</div>
            <div class="metric-value" style="color: {pred_color};">${signal['predicted_price']:,.2f}</div>
            <div class="metric-badge" style="color: {pred_color};">{pred_ret:+.4f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">CONFIANCE XGBOOST</div>
            <div class="metric-value" style="color: #00E5FF;">{meta_confidence*100:.1f}%</div>
            <div class="metric-badge badge-buy">SEUIL >= {config.min_meta_confidence*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with k5:
        if final_binary == 1:
            act_label, badge_class = "[A] ACHAT (BUY)", "badge-buy"
        else:
            act_label, badge_class = "[H] HOLD / NEUTRE", "badge-hold"
            
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ACTION DÉCIDÉE IA</div>
            <div class="metric-value" style="font-size: 1.1rem;">{act_label}</div>
            <div class="metric-badge {badge_class}">SOMME: ${allocated_cap:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    # 2. GRILLE PRINCIPALE (GRAPHIQUE À GAUCHE / RISK & ORDERS À DROITE)
    col_chart, col_side = st.columns([2.2, 1.0])
    
    with col_chart:
        # Graphique Chartiste Compact Single-Page (Hauteur 420px)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])
        
        df_sub = df_screened.tail(120)
        
        fig.add_trace(
            go.Candlestick(
                x=df_sub.index,
                open=df_sub['Open'], high=df_sub['High'],
                low=df_sub['Low'], close=df_sub['Close'],
                name="ETH/USD 5m",
                increasing_line_color='#00E676', decreasing_line_color='#FF5252'
            ), row=1, col=1
        )
        
        if 'SMA_50' in df_sub.columns:
            fig.add_trace(go.Scatter(x=df_sub.index, y=df_sub['SMA_50'], name="SMA 50", line=dict(color='#FFD700', width=1.2)), row=1, col=1)
            
        # Target Marker
        next_time = df_sub.index[-1] + pd.Timedelta(minutes=5)
        fig.add_trace(
            go.Scatter(
                x=[next_time],
                y=[signal['predicted_price']],
                mode='markers+text',
                name="Target H+1",
                marker=dict(size=10, color='#00E5FF', symbol='star'),
                text=[f"${signal['predicted_price']:.1f}"],
                textposition="top center"
            ), row=1, col=1
        )
        
        # RVOL Volume Subplot
        v_colors = ['#00E676' if r > 1.2 else '#475569' for r in df_sub['RVOL']]
        fig.add_trace(go.Bar(x=df_sub.index, y=df_sub['Volume'], name="Volume 5m", marker_color=v_colors), row=2, col=1)
        
        fig.update_layout(
            template="plotly_dark",
            height=430,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.6)',
            xaxis_rangeslider_visible=False,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.markdown("""
        <div style="background: rgba(21, 28, 44, 0.65); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 12px; padding: 14px 18px;">
            <div style="font-size: 0.9rem; font-weight: 700; color: #00E5FF; margin-bottom: 10px;">🛡️ PROTECTION & RISK ALLOCATION</div>
        </div>
        """, unsafe_allow_html=True)
        
        risk_df = pd.DataFrame([
            {"Paramètre": "Somme Misée / Trade", "Valeur": f"${allocated_cap:,.2f}"},
            {"Paramètre": "Plafond Max Autorisé", "Valeur": f"${cap_max_input:,.2f}"},
            {"Paramètre": "Unités ETH", "Valeur": f"{units:.4f} ETH"},
            {"Paramètre": "Stop-Loss (1.0x ATR)", "Valeur": f"${pos_info['stop_loss_price']:,.2f}"},
            {"Paramètre": "Take-Profit (1.5x ATR)", "Valeur": f"${pos_info['take_profit_price']:,.2f}"},
            {"Paramètre": "Broker Connecté", "Valeur": "Alpaca Paper ($100k)"}
        ])
        st.dataframe(risk_df, hide_index=True, use_container_width=True, height=220)
        
        st.markdown("""
        <div style="background: rgba(21, 28, 44, 0.65); border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 12px; padding: 10px 14px; margin-top: 10px; text-align: center;">
            <span style="font-size: 0.8rem; color: #94A3B8;">⏳ CYCLE 5M ACTIF | AUTO-REFRESH 300S</span>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
