"""
Configuration centrale du Bot de Trading Quantitatif TimesFM (ETH 1h).
"""
import os
from pydantic import BaseModel

class TradingConfig(BaseModel):
    # Actif & Timeframe
    symbol: str = "ETH/USDT"
    yf_symbol: str = "ETH-USD"
    timeframe: str = "1h"
    
    # Paramètres TimesFM
    context_len: int = 512       # Fenêtre de contexte historique (512 heures ~ 21 jours)
    horizon_len: int = 1         # Horizon de prédiction (1h à venir)
    backend: str = "cpu"         # 'cpu', 'gpu', 'cuda'
    
    # Règle de Pré-Screening (Volume & Momentum)
    rvol_threshold: float = 1.2  # Volume relatif > 1.2x
    use_sma_filter: bool = True  # Prix > SMA50 et Prix > SMA200
    rsi_period: int = 14
    rsi_min: float = 50.0
    rsi_max: float = 72.0
    
    # Gestion du Risque & Kelly Sizing
    risk_per_trade: float = 0.02 # 2% de risque fixe par défaut
    max_kelly_fraction: float = 0.25 # Fraction de Kelly conservatrice (Quarter-Kelly)
    max_portfolio_allocation: float = 0.50 # Enveloppe de sécurité (max 50% du capital)
    
    # Exchange CCXT
    exchange_id: str = "binance"  # ou 'kraken', 'coinbase'
    sandbox_mode: bool = True    # Mode Paper Trading par défaut

config = TradingConfig()
