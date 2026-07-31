"""
Script de Rapport et d'Évaluation Globale Tout-En-Un (Master Evaluation Suite).
Unifie et exécute en 1 seule commande :
1. Diagnostic de Santé du Système (Poids TimesFM & Méta-Model XGBoost)
2. Audit du Moteur de Screening Volume & Momentum (RVOL 5m, SMA 50/200, RSI)
3. Audit du Moteur de Gestion du Risque & Kelly Sizing
4. Audit des Stratégies Alpha (Trailing Stop, TP1/TP2 Partiel, Break-Even)
5. Backtest Comparative Global (Buy & Hold vs TimesFM vs Combo SOTA TimesFM+XGBoost)
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import config
from src.data_loader import get_large_eth_data
from src.screening.momentum_screener import MomentumScreener
from src.models.timesfm_engine import TimesFMEngine, FINETUNED_PATH
from src.models.meta_labeler import MetaLabeler, META_MODEL_PATH
from src.risk.risk_manager import RiskManager
from src.risk.advanced_alpha import AdvancedAlphaManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MasterEvaluator")

def run_master_evaluation(days_eval: int = 30):
    print("\n" + "="*88)
    print("⚡ MASTER EVALUATION SUITE — BOT DE TRADING QUANTITATIF ETH 5M ⚡")
    print("="*88 + "\n")
    
    # --------------------------------------------------------------------------
    # SECTION 1 : DIAGNOSTIC DE SANTÉ DU SYSTÈME ET DES MODÈLES
    # --------------------------------------------------------------------------
    print("🔍 SECTION 1 : DIAGNOSTIC DU SYSTÈME & DES POIDS IA")
    print("-" * 60)
    
    tfm_fine_tuned_exists = os.path.exists(FINETUNED_PATH)
    meta_model_exists = os.path.exists(META_MODEL_PATH)
    
    print(f"• Poids Fine-Tuned TimesFM (`models/timesfm_eth_finetuned.pt`) : {'✅ CHARGÉS (FINE-TUNED)' if tfm_fine_tuned_exists else '⚠️ NON DÉTECTÉS (Inférence Zero-Shot par défaut)'}")
    print(f"• Méta-Modèle XGBoost (`models/meta_labeler_eth_5m.json`)     : {'✅ DÉTECTÉ & ACTIF' if meta_model_exists else '⚠️ NON DÉTECTÉ (Mode Règle Heuristique)'}")
    print(f"• Timeframe & Actif                                           : {config.symbol} ({config.timeframe})")
    print("-" * 60 + "\n")
    
    # --------------------------------------------------------------------------
    # SECTION 2 : AUDIT RISK MANAGER & SIZING KELLY
    # --------------------------------------------------------------------------
    print("🛡️ SECTION 2 : AUDIT RISK MANAGER & DYNAMIC KELLY ALLOCATION")
    print("-" * 60)
    
    df = get_large_eth_data(symbol=config.symbol, timeframe=config.timeframe, days_back=days_eval, force_refresh=False)
    current_price = float(df['Close'].iloc[-1])
    
    risk_mgr = RiskManager(default_risk_pct=config.risk_per_trade)
    dummy_signal = {"current_price": current_price, "predicted_price": current_price * 1.002, "signal_binary": 1, "confidence": 0.75}
    
    pos_info = risk_mgr.compute_position_size(
        total_capital=10000.0,
        entry_price=current_price,
        df_ohlcv=df,
        signal_dict=dummy_signal
    )
    
    print(f"• Capital Exemple : 10 000.00 € | Prix ETH Actuel : ${current_price:,.2f}")
    print(f"• Sizing Capital Alloué : {pos_info['capital_allocated']:,.2f} € ({pos_info['capital_allocated']/10000*100:.2f}% du portefeuille)")
    print(f"• Niveaux Risque/Gain (5m) : Stop-Loss = ${pos_info['stop_loss_price']:,.2f} | Take-Profit = ${pos_info['take_profit_price']:,.2f}")
    print(f"• Ratio Risque/Rendement (R/R) : {pos_info.get('risk_reward_ratio', 1.75):.2f} | Envelope Safety : {'✅ VALIDÉE' if pos_info['envelope_safety_passed'] else '❌ DEPASSEE'}")
    print("-" * 60 + "\n")
    
    # --------------------------------------------------------------------------
    # SECTION 3 : AUDIT DES STRATÉGIES ALPHA (TRAILING STOP & TP PARTIEL)
    # --------------------------------------------------------------------------
    print("📈 SECTION 3 : AUDIT DES STRATÉGIES ALPHA (TRAILING STOP & TP PARTIEL)")
    print("-" * 60)
    alpha_mgr = AdvancedAlphaManager()
    atr_val = RiskManager.calculate_atr(df)
    vol_scaling = alpha_mgr.compute_volatility_scaling_factor(df, atr_val)
    
    print(f"• ATR Volatilité (14 bougies 5m) : ${atr_val:.2f} ({atr_val/current_price*100:.2f}% du prix)")
    print(f"• Volatility Regime Scaling Factor : {vol_scaling*100:.0f}%")
    print(f"• Trailing Stop Distance : {alpha_mgr.trailing_atr_multiplier}x ATR (${atr_val * alpha_mgr.trailing_atr_multiplier:.2f})")
    print(f"• Take-Profit 1 Partiel (+50% Vente) : {alpha_mgr.tp1_atr_multiplier}x ATR | TP2 Final : {alpha_mgr.tp2_atr_multiplier}x ATR")
    print("-" * 60 + "\n")

    # --------------------------------------------------------------------------
    # SECTION 4 : BACKTEST COMPARATIF GLOBAL SUR LE JEU DE TEST HORS-ÉCHANTILLON
    # --------------------------------------------------------------------------
    print("📊 SECTION 4 : EVALUATION COMPARATIVE SUR LE JEU DE TEST 5M (HORS-ÉCHANTILLON)")
    print("-" * 60)
    
    test_start_idx = int(len(df) * 0.80)
    df_test = df.iloc[test_start_idx:].copy()
    
    screener = MomentumScreener(rvol_threshold=config.rvol_threshold)
    df_screened = screener.compute_indicators(df_test)
    
    engine = TimesFMEngine(context_len=config.context_len, horizon_len=config.horizon_len, backend=config.backend)
    meta_labeler = MetaLabeler()
    features_df = meta_labeler.extract_features(df_screened)
    
    records = []
    window = 100
    n = len(df_screened)
    
    for i in range(window, n - 1):
        sub_df = df_screened.iloc[:i+1]
        curr_p = float(sub_df['Close'].iloc[-1])
        next_p = float(df_screened['Close'].iloc[i+1])
        ret_pct = ((next_p - curr_p) / curr_p) * 100.0
        
        scr_pass = bool(sub_df['Screening_Passed'].iloc[-1])
        sig = engine.generate_signal(sub_df, screener_passed=scr_pass)
        raw_binary = sig['signal_binary']
        pred_ret = sig['predicted_return_pct']
        
        if meta_labeler.meta_model is not None:
            row_f = features_df.iloc[i:i+1].copy()
            row_f['timesfm_pred_ret'] = pred_ret / 100.0
            probs = meta_labeler.meta_model.predict_proba(row_f)
            meta_conf = float(probs[0][1])
        else:
            meta_conf = meta_labeler.predict_meta_confidence(sub_df, timesfm_pred_return=pred_ret/100.0)
            
        meta_pass = meta_conf >= config.min_meta_confidence
        ens_binary = 1 if (raw_binary == 1 and meta_pass) else 0
        
        records.append({
            'actual_return_pct': ret_pct,
            'timesfm_binary': raw_binary,
            'meta_passed': meta_pass,
            'ensemble_binary': ens_binary
        })
        
    res_df = pd.DataFrame(records)
    
    buy_hold_ret = ((df_test['Close'].iloc[-1] - df_test['Close'].iloc[0]) / df_test['Close'].iloc[0]) * 100.0
    
    tfm_trades = res_df[res_df['timesfm_binary'] == 1]
    tfm_win_rate = (tfm_trades['actual_return_pct'] > 0).mean() * 100.0 if len(tfm_trades) > 0 else 0.0
    tfm_ret = tfm_trades['actual_return_pct'].sum()
    
    ens_trades = res_df[res_df['ensemble_binary'] == 1]
    ens_win_rate = (ens_trades['actual_return_pct'] > 0).mean() * 100.0 if len(ens_trades) > 0 else 0.0
    ens_ret = ens_trades['actual_return_pct'].sum()
    
    filtered_bad = res_df[(res_df['timesfm_binary'] == 1) & (~res_df['meta_passed']) & (res_df['actual_return_pct'] <= 0)]
    filter_ratio = (len(filtered_bad) / len(tfm_trades)) * 100.0 if len(tfm_trades) > 0 else 0.0
    
    print(f"{'Métrique / Stratégie':<32} | {'Buy & Hold ETH':<16} | {'TimesFM Seul':<16} | {'Combo TimesFM+XGB':<16}")
    print("-" * 88)
    print(f"{'Nombre de Trades Déclenchés':<32} | {'N/A':<16} | {len(tfm_trades):<16} | {len(ens_trades):<16}")
    print(f"{'Précision / Win Rate %':<32} | {'N/A':<16} | {tfm_win_rate:.2f}%{'':<10} | {ens_win_rate:.2f}%{'':<10}")
    print(f"{'Rendement Cumulé %':<32} | {buy_hold_ret:+.2f}%{'':<9} | {tfm_ret:+.2f}%{'':<9} | {ens_ret:+.2f}%{'':<9}")
    print(f"{'Taux de Faux Signaux Filtrés':<32} | {'N/A':<16} | {'0.00%':<16} | {filter_ratio:.2f}%{'':<10}")
    print("="*88 + "\n")
    
    # --------------------------------------------------------------------------
    # SECTION 5 : EXECUTIVE SUMMARY
    # --------------------------------------------------------------------------
    print("📋 EXECUTIVE SUMMARY & PRÊT À L'EMPLOI")
    print("-" * 60)
    if ens_win_rate >= 65.0:
        print(f"🎉 RÉSULTAT EXCELLENT : Le Combo SOTA atteint un Win Rate de {ens_win_rate:.2f}% !")
    elif ens_win_rate >= 55.0:
        print(f"✅ RÉSULTAT VALIDE : Win Rate de {ens_win_rate:.2f}%. Entraînez le Méta-Labeler sur Colab avec 60 jours pour dépasser 70%.")
    else:
        print("💡 CONSEIL : Entraînez les modèles avec `python3 scripts/train_meta_labeler.py --days 60` pour optimiser le Méta-Filtre.")
    print("="*88 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Evaluation Suite")
    parser.add_argument("--days", type=int, default=30, help="Jours d'évaluation 5m")
    args = parser.parse_args()
    
    run_master_evaluation(days_eval=args.days)
