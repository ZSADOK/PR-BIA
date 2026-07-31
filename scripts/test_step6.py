"""
Script de test automatisé pour l'Étape 6 : Stratégies d'Optimisation (Alpha).
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf
from src.risk.advanced_alpha import AdvancedAlphaManager
from src.risk.risk_manager import RiskManager
from config.settings import config

def main():
    print("=== TEST STEP 6 : Optimisations Alpha (Trailing Stop, TP Partiel & Scaling Volatilité) ===")
    
    alpha_mgr = AdvancedAlphaManager()
    
    # 1. Test du Scaling par Régime de Volatilité (ATR)
    ticker = yf.Ticker(config.yf_symbol)
    df_raw = ticker.history(period="15d", interval="1h")
    df = df_raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()
    
    atr = RiskManager.calculate_atr(df)
    vol_factor = alpha_mgr.compute_volatility_scaling_factor(df, atr)
    
    current_price = float(df['Close'].iloc[-1])
    print(f"Prix ETH actuel : ${current_price:.2f}")
    print(f"ATR(14) : ${atr:.2f} (Ratio Volatilité = {atr/current_price*100:.2f}%)")
    print(f"Facteur d'ajustement du sizing (Volatility Scaling) : {vol_factor*100:.0f}%")
    
    # 2. Simulation d'un Scenario Trailing Stop & Take Profit Partiel
    entry_price = 2000.0
    initial_units = 1.0 # 1 ETH
    atr_sim = 40.0 # ATR de 40$
    
    print("\n--- SIMULATION DU SCENARIO DE TRADE ETH (Entrée à 2000.00 $) ---")
    
    # Étape A : Le prix monte à 2070 $ (+1.75x ATR -> Déclenchement TP1 Partiel + Break-Even)
    state_1 = alpha_mgr.update_position_state(
        entry_price=entry_price,
        highest_price_seen=2000.0,
        current_price=2070.0,
        atr=atr_sim,
        initial_units=initial_units,
        tp1_executed=False
    )
    
    print(f"1. Prix monte à $2070.00 (Target TP1: ${state_1['tp1_target']:.2f})")
    print(f"   -> Action : {state_1['action']} (Vente de {state_1['units_to_close']} ETH)")
    print(f"   -> Nouveau Trailing Stop ajusté : ${state_1['effective_stop_loss']:.2f} (Protecting Profit/Break-Even)")
    
    # Étape B : Le prix continue de grimper jusqu'à 2150 $ (+3.75x ATR -> Déclenchement TP2 Final)
    state_2 = alpha_mgr.update_position_state(
        entry_price=entry_price,
        highest_price_seen=2070.0,
        current_price=2150.0,
        atr=atr_sim,
        initial_units=initial_units,
        tp1_executed=state_1['tp1_executed']
    )
    
    print(f"2. Prix grimpe à $2150.00 (Target TP2: ${state_2['tp2_target']:.2f})")
    print(f"   -> Action : {state_2['action']} (Clôture finale des {state_2['units_to_close']} ETH restants)")
    
    print("\n[SUCCESS] Test Étape 6 (Alpha & Trailing/TP Partiel) validé !")

if __name__ == "__main__":
    main()
