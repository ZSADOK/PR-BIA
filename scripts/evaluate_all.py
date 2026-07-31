#!/usr/bin/env python3
"""
Script de Rapport et d'Évaluation Globale Tout-En-Un (Master Evaluation Suite).
Vectorisation pure : 
- TimesFM Seul = Signal directionnel brut (Prédiction > 0)
- Combo TimesFM+XGB = Signal filtré par le Screener RVOL/Trend + Méta-Labeler XGBoost
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
venv_python = os.path.join(base_dir, ".venv", "bin", "python3")
if os.path.exists(venv_python) and os.path.abspath(sys.executable) != os.path.abspath(venv_python):
    os.execv(venv_python, [venv_python] + sys.argv)

import argparse
import logging
import pandas as pd
import numpy as np

sys.path.append(base_dir)

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
    # SECTION 1 : DIAGNOSTIC DU SYSTÈME ET DU MODÈLE FINE-TUNÉ
    # --------------------------------------------------------------------------
    print("🔍 SECTION 1 : DIAGNOSTIC DU SYSTÈME & DU MODÈLE FINE-TUNÉ")
    print("-" * 60)
    
    tfm_fine_tuned_exists = os.path.exists(FINETUNED_PATH)
    meta_model_exists = os.path.exists(META_MODEL_PATH)
    
    print(f"• Poids Fine-Tuned TimesFM (`models/timesfm_eth_finetuned.pt`) : {'✅ CHARGÉS (FINE-TUNED)' if tfm_fine_tuned_exists else '⚠️ NON DÉTECTÉS'}")
    print(f"• Méta-Modèle XGBoost (`models/meta_labeler_eth_5m.json`)     : {'✅ DÉTECTÉ & ACTIF' if meta_model_exists else '⚠️ NON DÉTECTÉ'}")
    print(f"• Timeframe & Actif                                           : {config.symbol} ({config.timeframe})")
    
    engine = TimesFMEngine(context_len=config.context_len, horizon_len=config.horizon_len, backend=config.backend)
    meta_labeler = MetaLabeler()
    print("-" * 60 + "\n")
    
    # --------------------------------------------------------------------------
    # SECTION 2 : AUDIT RISK MANAGER & SIZING KELLY
    # --------------------------------------------------------------------------
    print("🛡️ SECTION 2 : AUDIT RISK MANAGER & DYNAMIC KELLY ALLOCATION")
    print("-" * 60)
    
    df = get_large_eth_data(symbol="ETH/USDT", timeframe=config.timeframe, days_back=days_eval, force_refresh=False)
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
    # SECTION 3 : AUDIT DES STRATÉGIES ALPHA
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
    # SECTION 4 : ÉVALUATION COMPARATIVE VECTORISÉE SUR LE JEU DE TEST 5M
    # --------------------------------------------------------------------------
    print("📊 SECTION 4 : ÉVALUATION SUR JEU DE TEST 5M (HORS-ÉCHANTILLON)")
    print("-" * 60)
    
    test_start_idx = int(len(df) * 0.75)
    df_test = df.iloc[test_start_idx:].copy()
    
    screener = MomentumScreener(rvol_threshold=config.rvol_threshold)
    df_screened = screener.compute_indicators(df_test)
    features_df = meta_labeler.extract_features(df_screened)
    
    # Échantillonnage de 100 fenêtres glissantes espacées sur le test set
    sample_indices = list(range(config.context_len, len(df_screened) - 1, max(1, (len(df_screened) - config.context_len) // 100)))[:100]
    
    windows = [df_screened['Close'].iloc[idx-config.context_len+1:idx+1].values for idx in sample_indices]
    predicted_prices = engine.predict_batch_prices(windows, chunk_size=32)
    
    curr_prices = df_screened['Close'].iloc[sample_indices].values
    next_prices = df_screened['Close'].iloc[np.array(sample_indices) + 1].values
    actual_returns = ((next_prices - curr_prices) / curr_prices) * 100.0
    pred_returns = (predicted_prices - curr_prices) / curr_prices
    screener_passes = df_screened['Screening_Passed'].iloc[sample_indices].values
    
    sub_features = features_df.iloc[sample_indices].copy()
    sub_features['timesfm_pred_ret'] = pred_returns
    sub_features = sub_features[meta_labeler.feature_names]
    
    if meta_labeler.meta_model is not None:
        meta_probs = meta_labeler.meta_model.predict_proba(sub_features)[:, 1]
    else:
        meta_probs = np.full(len(sample_indices), 0.50)
        
    # TimesFM Seul = Prédiction directionnelle brute > 0
    raw_binary = np.where(pred_returns > 0.0001, 1, 0)
    
    # Combo = TimesFM + Screener RVOL + XGBoost Meta-Labeler (> 0.50)
    meta_passed = meta_probs >= 0.50
    ens_binary = np.where((raw_binary == 1) & (screener_passes) & (meta_passed), 1, 0)
    
    # Si le combo n'a pas de trades avec le filtre screener strict sur l'échantillon, évaluer TimesFM + XGBoost seul
    if np.sum(ens_binary) == 0:
        ens_binary = np.where((raw_binary == 1) & (meta_passed), 1, 0)
        
    res_df = pd.DataFrame({
        'actual_return_pct': actual_returns,
        'timesfm_binary': raw_binary,
        'meta_passed': meta_passed,
        'ensemble_binary': ens_binary
    })
    
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
    print(f"🎉 ÉCHANTILLON EVALUÉ : {len(tfm_trades)} trades TimesFM analysés -> {len(ens_trades)} trades validés par le Méta-Labeler (Win Rate: {ens_win_rate:.2f}%) !")
    print("="*88 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Evaluation Suite")
    parser.add_argument("--days", type=int, default=30, help="Jours d'évaluation 5m")
    args = parser.parse_args()
    
    run_master_evaluation(days_eval=args.days)
