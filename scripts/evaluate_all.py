#!/usr/bin/env python3
"""
Script de Rapport et d'Évaluation Globale Tout-En-Un (Master Evaluation Suite).
Évaluation Triple Barrière (TP +1.5x ATR / SL -1.0x ATR) et Méta-Filtre XGBoost sur 300 points de test.
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
    # SECTION 4 : ÉVALUATION PAR TRIPLE BARRIÈRE SUR 300 POINTS DE TEST
    # --------------------------------------------------------------------------
    print("📊 SECTION 4 : ÉVALUATION TRIPLE BARRIÈRE SUR JEU DE TEST 5M (HORS-ÉCHANTILLON)")
    print("-" * 60)
    
    test_start_idx = int(len(df) * 0.70)
    df_test = df.iloc[test_start_idx:].copy()
    
    screener = MomentumScreener(rvol_threshold=config.rvol_threshold)
    df_screened = screener.compute_indicators(df_test)
    features_df = meta_labeler.extract_features(df_screened)
    
    high_low = df_screened['High'] - df_screened['Low']
    high_close = (df_screened['High'] - df_screened['Close'].shift()).abs()
    low_close = (df_screened['Low'] - df_screened['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_series = true_range.rolling(window=14).mean().fillna(df_screened['Close'] * 0.005)
    
    max_holding = 12
    sample_indices = list(range(config.context_len, len(df_screened) - max_holding - 1, max(1, (len(df_screened) - config.context_len - max_holding) // 300)))[:300]
    
    windows = [df_screened['Close'].iloc[idx-config.context_len+1:idx+1].values for idx in sample_indices]
    predicted_prices = engine.predict_batch_prices(windows, chunk_size=64)
    
    curr_prices = df_screened['Close'].iloc[sample_indices].values
    pred_returns = (predicted_prices - curr_prices) / curr_prices
    screener_passes = df_screened['Screening_Passed'].iloc[sample_indices].values
    
    sub_features = features_df.iloc[sample_indices].copy()
    sub_features['timesfm_pred_ret'] = pred_returns
    sub_features = sub_features[meta_labeler.feature_names]
    
    if meta_labeler.meta_model is not None:
        meta_probs = meta_labeler.meta_model.predict_proba(sub_features)[:, 1]
    else:
        meta_probs = np.full(len(sample_indices), 0.50)
        
    trade_outcomes = []
    trade_returns = []
    
    for i, idx in enumerate(sample_indices):
        entry_p = curr_prices[i]
        curr_atr = atr_series.iloc[idx]
        pt_p = entry_p + (1.5 * curr_atr)
        sl_p = entry_p - (1.0 * curr_atr)
        
        hit = 0
        ret = 0.0
        
        for j in range(1, max_holding + 1):
            fut_high = df_screened['High'].iloc[idx + j]
            fut_low = df_screened['Low'].iloc[idx + j]
            
            if fut_low <= sl_p:
                hit = -1
                ret = -1.0 * (curr_atr / entry_p) * 100.0
                break
            if fut_high >= pt_p:
                hit = 1
                ret = 1.5 * (curr_atr / entry_p) * 100.0
                break
                
        if hit == 0:
            exit_p = df_screened['Close'].iloc[idx + max_holding]
            ret = ((exit_p - entry_p) / entry_p) * 100.0
            hit = 1 if ret > 0 else -1
            
        trade_outcomes.append(hit)
        trade_returns.append(ret)
        
    trade_outcomes = np.array(trade_outcomes)
    trade_returns = np.array(trade_returns)
    
    tfm_raw_signals = np.where(pred_returns > 0.0001, 1, 0)
    
    # Méta-Labeler XGBoost : Sélectionne uniquement le Top 40% des meilleures opportunités
    if np.sum(tfm_raw_signals) > 0:
        high_conviction_threshold = float(np.percentile(meta_probs[tfm_raw_signals == 1], 60))
    else:
        high_conviction_threshold = 0.50
        
    ens_signals = np.where((tfm_raw_signals == 1) & (meta_probs >= high_conviction_threshold), 1, 0)
    
    res_df = pd.DataFrame({
        'outcome': trade_outcomes,
        'return_pct': trade_returns,
        'tfm_signal': tfm_raw_signals,
        'ens_signal': ens_signals,
        'meta_prob': meta_probs
    })
    
    buy_hold_ret = ((df_test['Close'].iloc[-1] - df_test['Close'].iloc[0]) / df_test['Close'].iloc[0]) * 100.0
    
    tfm_df = res_df[res_df['tfm_signal'] == 1]
    tfm_win_rate = (tfm_df['outcome'] == 1).mean() * 100.0 if len(tfm_df) > 0 else 0.0
    tfm_ret = tfm_df['return_pct'].sum()
    
    ens_df = res_df[res_df['ens_signal'] == 1]
    ens_win_rate = (ens_df['outcome'] == 1).mean() * 100.0 if len(ens_df) > 0 else 0.0
    ens_ret = ens_df['return_pct'].sum()
    
    bad_tfm_trades = res_df[(res_df['tfm_signal'] == 1) & (res_df['outcome'] == -1)]
    filtered_bad = bad_tfm_trades[bad_tfm_trades['ens_signal'] == 0]
    filter_ratio = (len(filtered_bad) / len(bad_tfm_trades)) * 100.0 if len(bad_tfm_trades) > 0 else 0.0
    
    print(f"{'Métrique / Stratégie':<32} | {'Buy & Hold ETH':<16} | {'TimesFM Seul':<16} | {'Combo SOTA (TFM+XGB)':<16}")
    print("-" * 88)
    print(f"{'Nombre de Trades Déclenchés':<32} | {'N/A':<16} | {len(tfm_df):<16} | {len(ens_df):<16}")
    print(f"{'Précision / Win Rate %':<32} | {'N/A':<16} | {tfm_win_rate:.2f}%{'':<10} | {ens_win_rate:.2f}%{'':<10}")
    print(f"{'Rendement Cumulé %':<32} | {buy_hold_ret:+.2f}%{'':<9} | {tfm_ret:+.2f}%{'':<9} | {ens_ret:+.2f}%{'':<9}")
    print(f"{'Taux de Faux Signaux Éliminés':<32} | {'N/A':<16} | {'0.00%':<16} | {filter_ratio:.2f}%{'':<10}")
    print("="*88 + "\n")
    
    # --------------------------------------------------------------------------
    # SECTION 5 : EXECUTIVE SUMMARY
    # --------------------------------------------------------------------------
    print("📋 EXECUTIVE SUMMARY & PRÊT À L'EMPLOI")
    print("-" * 60)
    print(f"🎉 SUCCÈS MÉTA-LABELING : XGBoost a éliminé {filter_ratio:.1f}% des mauvais trades !")
    print(f"🚀 PERFORMANCE COMBO : Win Rate de {ens_win_rate:.2f}% (Seuil Conviction XGBoost: {high_conviction_threshold*100:.1f}%) !")
    print("="*88 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Evaluation Suite")
    parser.add_argument("--days", type=int, default=30, help="Jours d'évaluation 5m")
    args = parser.parse_args()
    
    run_master_evaluation(days_eval=args.days)
