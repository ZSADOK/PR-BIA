"""
Configuration centrale du Bot de Trading Quantitatif TimesFM + Meta-Labeler (ETH 5m / Alpaca).
Mode Actif & Haute Fréquence (Seuil de Conviction = 50%).
"""
import os
from pydantic import BaseModel

class TradingConfig(BaseModel):
    # Actif & Timeframe 5 Minutes (5m)
    symbol: str = "ETH/USD"       # Paire Alpaca / Exchange
    yf_symbol: str = "ETH-USD"
    timeframe: str = "5m"         # Timeframe 5 minutes
    
    # Paramètres TimesFM
    context_len: int = 512        # Fenêtre de mémoire historique (512 * 5 min = ~42.6 heures)
    horizon_len: int = 1          # Horizon de prédiction (5 min à venir)
    backend: str = "cpu"          # 'cpu', 'gpu', 'cuda'
    
    # Règle de Pré-Screening (Volume & Momentum 5m)
    rvol_threshold: float = 1.0   # Volume relatif neutre (>= 1.0x) pour trading dynamique
    use_sma_filter: bool = False  # Désactivé pour capturer les rebonds sous les SMAs
    rsi_period: int = 14
    rsi_min: float = 40.0
    rsi_max: float = 80.0
    
    # Méta-Labeling & Seuil de Conviction (Mode Actif = 50%)
    min_meta_confidence: float = 0.50  # Seuil de probabilité à 50% pour trading très actif
    min_predicted_return: float = 0.0001 # 0.01% de rendement prédit minimum sur 5m
    
    # Gestion du Risque & Kelly Sizing
    risk_per_trade: float = 0.015 # 1.5% de risque fixe par trade sur 5m
    max_kelly_fraction: float = 0.25 # Fraction de Kelly (Quarter-Kelly)
    max_portfolio_allocation: float = 0.50 # Enveloppe de sécurité (max 50% du capital)
    
    # Exchange Selection (Alpaca Paper Trading 24/7)
    exchange_id: str = "alpaca"    # 'alpaca' ou 'binance'
    sandbox_mode: bool = True     # Mode Paper Trading

config = TradingConfig()
