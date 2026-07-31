"""
Script de test automatisé pour l'Étape 2 : Chargement données ETH 1h, Screener & TimesFM Inférence.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
from src.screening.momentum_screener import MomentumScreener
from src.models.timesfm_engine import TimesFMEngine
from config.settings import config

def main():
    print("=== TEST STEP 2 : ETH 1h Data & TimesFM Inference ===")
    print(f"Téléchargement des données pour {config.yf_symbol} (TF: {config.timeframe})...")
    
    ticker = yf.Ticker(config.yf_symbol)
    df_raw = ticker.history(period="30d", interval="1h")
    df = df_raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()
    print(f"Chargé: {len(df)} bougies de 1 heure.")
    
    print("\n--- 1. Pré-Screening Volume & Momentum ---")
    screener = MomentumScreener(rvol_threshold=config.rvol_threshold)
    df_screened = screener.compute_indicators(df)
    latest_screen = screener.evaluate_latest(df_screened)
    print(f"Dernière bougie {df_screened.index[-1]}: Screener Passed = {latest_screen['passed']}")
    print(f" - RVOL: {latest_screen['rvol']:.2f} (Seuil > {config.rvol_threshold})")
    print(f" - Trend OK (Close > SMA50 & SMA200): {latest_screen['trend_ok']}")
    print(f" - RSI(14): {latest_screen['rsi']:.2f} (RSI OK: {latest_screen['rsi_ok']})")
    
    print("\n--- 2. Modélisation TimesFM (Prédiction H+1) ---")
    engine = TimesFMEngine(context_len=config.context_len, horizon_len=config.horizon_len, backend=config.backend)
    signal = engine.generate_signal(df_screened, screener_passed=latest_screen['passed'])
    
    print(f" - Prix actuel ETH: {signal['current_price']:.2f} $")
    print(f" - Prix prédit H+1: {signal['predicted_price']:.2f} $")
    print(f" - Variation prédite: {signal['predicted_return_pct']:+.4f} %")
    print(f" - Signal binaire final: {signal['signal_binary']} ({signal['signal_label']})")
    print("\n[SUCCESS] Test Étape 2 validé avec succès !")

if __name__ == "__main__":
    main()
