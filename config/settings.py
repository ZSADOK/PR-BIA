"""
Configuration centrale du Bot de Trading Quantitatif TimesFM + Meta-Labeler (ETH 5m).
"""
import os
from pydantic import BaseModel

class TradingConfig(BaseModel):
    # Actif & Timeframe 5 Minutes (5m)
    symbol: str = "ETH/USDT"
    yf_symbol: str = "ETH-USD"
    timeframe: str = "5m"         # Timeframe ultra-rapide 5 minutes
    
    # Paramètres TimesFM
    context_len: int = 512        # Fenêtre de mémoire historique (512 * 5 min = ~42.6 heures)
    horizon_len: int = 1          # Horizon de prédiction (5 min à venir)
    backend: str = "cpu"          # 'cpu', 'gpu', 'cuda'
    
    # Règle de Pré-Screening (Volume & Momentum 5m)
    rvol_threshold: float = 1.2   # Volume relatif > 1.2x sur 5m
    use_sma_filter: bool = True   # Prix > SMA50 (5m) et Prix > SMA200 (5m)
    rsi_period: int = 14
    rsi_min: float = 50.0
    rsi_max: float = 72.0
    
    # Méta-Labeling & Seuil de Conviction (Win Rate > 72%)
    min_meta_confidence: float = 0.65  # Seuil de probabilité minimale du Méta-Model pour valider un trade
    min_predicted_return: float = 0.0010 # 0.10% de rendement prédit minimum sur 5m
    
    # Gestion du Risque & Kelly Sizing
    risk_per_trade: float = 0.015 # 1.5% de risque fixe par trade sur 5m
    max_kelly_fraction: float = 0.25 # Fraction de Kelly (Quarter-Kelly)
    max_portfolio_allocation: float = 0.50 # Enveloppe de sécurité (max 50% du capital)
    
    # Exchange CCXT
    exchange_id: str = "binance"
    sandbox_mode: bool = True     # Mode Paper Trading

config = TradingConfig()
