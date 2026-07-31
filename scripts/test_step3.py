"""
Script de test automatisé pour l'Étape 3 : Module de Gestion du Risque et Position Sizing.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from src.risk.risk_manager import RiskManager
from src.screening.momentum_screener import MomentumScreener
from src.models.timesfm_engine import TimesFMEngine
from config.settings import config

def main():
    print("=== TEST STEP 3 : Risk Management & Position Sizing ===")
    
    # Capital de départ pour le test
    total_capital = 10000.0  # 10 000 € / $
    print(f"Capital Total de départ : {total_capital:.2f} €")
    
    # Chargement de données ETH 1h récentes
    print("Chargement des données ETH pour calcul du sizing...")
    ticker = yf.Ticker(config.yf_symbol)
    df_raw = ticker.history(period="15d", interval="1h")
    df = df_raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()
    
    screener = MomentumScreener(rvol_threshold=config.rvol_threshold)
    df_screened = screener.compute_indicators(df)
    latest_screen = screener.evaluate_latest(df_screened)
    
    engine = TimesFMEngine(context_len=config.context_len, horizon_len=config.horizon_len, backend=config.backend)
    signal = engine.generate_signal(df_screened, screener_passed=latest_screen['passed'])
    
    print(f"Prix d'entrée ETH : {signal['current_price']:.2f} $")
    print(f"Signal binaire généré : {signal['signal_binary']} ({signal['signal_label']})")
    
    # Test A : Simulation d'un signal BUY forcé pour valider les formules de sizing
    simulated_buy_signal = signal.copy()
    simulated_buy_signal['signal_binary'] = 1
    simulated_buy_signal['confidence'] = 0.75
    
    risk_mgr = RiskManager(
        default_risk_pct=config.risk_per_trade,
        max_kelly_fraction=config.max_kelly_fraction,
        max_portfolio_cap=config.max_portfolio_allocation
    )
    
    position_info = risk_mgr.compute_position_size(
        total_capital=total_capital,
        entry_price=signal['current_price'],
        df_ohlcv=df_screened,
        signal_dict=simulated_buy_signal
    )
    
    print("\n--- RÉSULTATS DU POSITION SIZING (Signal Achat) ---")
    print(f" - Capital Alloué (€/$) : {position_info['capital_allocated']:.2f} € (soit {position_info['capital_allocated']/total_capital*100:.2f}% du portefeuille)")
    print(f" - Quantité d'ETH à Acheter : {position_info['quantity_units']:.4f} ETH")
    print(f" - Prix d'Entrée : {position_info['entry_price']:.2f} $")
    print(f" - Stop-Loss Dynamique (2x ATR) : {position_info['stop_loss_price']:.2f} $")
    print(f" - Take-Profit Dynamique (3.5x ATR) : {position_info['take_profit_price']:.2f} $")
    print(f" - Ratio Risque/Rendement (R/R) : {position_info['risk_reward_ratio']:.2f}")
    print(f" - Montant Maximal Risqué en € : {position_info['max_risk_amount']:.2f} € ({position_info['risk_pct_used']:.2f}% du Capital Total)")
    print(f" - Dynamic Kelly Fraction : {position_info['kelly_fraction']*100:.2f}%")
    print(f" - Envelope Safety Compliance : {position_info['envelope_safety_passed']}")
    
    assert position_info['capital_allocated'] <= total_capital * config.max_portfolio_allocation, "Erreur: Plafond enveloppe dépassé"
    assert position_info['max_risk_amount'] <= total_capital * config.risk_per_trade * 1.5, "Erreur: Risque trop élevé"
    
    print("\n[SUCCESS] Test Étape 3 (Risk Manager & Dynamic Kelly Sizing) validé !")

if __name__ == "__main__":
    main()
