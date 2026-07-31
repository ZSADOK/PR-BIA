"""
Script d'Évaluation Globale de l'Ensemble (TimesFM + Méta-Labeler XGBoost sur ETH 5m).
Évalue côte-à-côte sur un jeu de test hors-échantillon :
1. TimesFM Seul
2. Ensemble SOTA : TimesFM + Méta-Labeler XGBoost (Triple Barrière)
3. Buy & Hold ETH

Affiche la Précision (Win Rate %), le Nombre de Trades, le Cumulative Return %, le Sharpe Ratio et le Max Drawdown.
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
from src.models.timesfm_engine import TimesFMEngine
from src.models.meta_labeler import MetaLabeler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EnsembleEvaluator")

def evaluate_ensemble(days_eval: int = 30, test_split_pct: float = 0.20):
    logger.info(f"=== ÉVALUATION DU COMBO SOTA TIMESFM + MÉTA-LABELER XGBOOST ({days_eval} JOURS 5m) ===")
    
    # 1. Ingestion des données 5m
    df = get_large_eth_data(symbol="ETH/USDT", timeframe="5m", days_back=days_eval, force_refresh=False)
    
    # Isolation du Jeu de Test Hors-Échantillon (20% derniers jours)
    test_start_idx = int(len(df) * (1.0 - test_split_pct))
    df_test = df.iloc[test_start_idx:].copy()
    
    logger.info(f"Jeu de Test Hors-Échantillon : {len(df_test)} bougies 5m (du {df_test.index[0]} au {df_test.index[-1]})")
    
    # 2. Initialisation des Moteurs
    screener = MomentumScreener(rvol_threshold=config.rvol_threshold)
    df_screened = screener.compute_indicators(df_test)
    
    engine = TimesFMEngine(context_len=config.context_len, horizon_len=config.horizon_len, backend=config.backend)
    meta_labeler = MetaLabeler()
    
    # Extraction vectorisée des features du meta-labeler pour accélération
    features_df = meta_labeler.extract_features(df_screened)
    
    records = []
    window = 100
    n = len(df_screened)
    
    logger.info(f"Simulation accélérée sur {n - window - 1} bougies 5m...")
    for i in range(window, n - 1):
        sub_df = df_screened.iloc[:i+1]
        current_price = float(sub_df['Close'].iloc[-1])
        actual_next_price = float(df_screened['Close'].iloc[i+1])
        actual_return_pct = ((actual_next_price - current_price) / current_price) * 100.0
        
        screener_passed = bool(sub_df['Screening_Passed'].iloc[-1])
        
        # Inférence TimesFM
        signal = engine.generate_signal(sub_df, screener_passed=screener_passed)
        timesfm_raw_binary = signal['signal_binary']
        pred_return_pct = signal['predicted_return_pct']
        
        # Inférence Méta-Labeler XGBoost
        if meta_labeler.meta_model is not None:
            row_feat = features_df.iloc[i:i+1].copy()
            row_feat['timesfm_pred_ret'] = pred_return_pct / 100.0
            probs = meta_labeler.meta_model.predict_proba(row_feat)
            meta_confidence = float(probs[0][1])
        else:
            meta_confidence = meta_labeler.predict_meta_confidence(sub_df, timesfm_pred_return=pred_return_pct/100.0)
            
        meta_passed = meta_confidence >= config.min_meta_confidence
        ensemble_binary = 1 if (timesfm_raw_binary == 1 and meta_passed) else 0
        
        records.append({
            'timestamp': sub_df.index[-1],
            'current_price': current_price,
            'actual_next_price': actual_next_price,
            'actual_return_pct': actual_return_pct,
            'timesfm_binary': timesfm_raw_binary,
            'meta_confidence': meta_confidence,
            'meta_passed': meta_passed,
            'ensemble_binary': ensemble_binary,
            'actual_win': 1 if actual_return_pct > 0 else 0
        })

    res_df = pd.DataFrame(records)
    
    # 3. CALCUL DES MÉTRIQUES COMPARATIVES
    buy_hold_return = ((res_df['current_price'].iloc[-1] - res_df['current_price'].iloc[0]) / res_df['current_price'].iloc[0]) * 100.0
    
    tfm_trades = res_df[res_df['timesfm_binary'] == 1]
    tfm_win_rate = (tfm_trades['actual_return_pct'] > 0).mean() * 100.0 if len(tfm_trades) > 0 else 0.0
    tfm_total_return = tfm_trades['actual_return_pct'].sum()
    
    ens_trades = res_df[res_df['ensemble_binary'] == 1]
    ens_win_rate = (ens_trades['actual_return_pct'] > 0).mean() * 100.0 if len(ens_trades) > 0 else 0.0
    ens_total_return = ens_trades['actual_return_pct'].sum()
    
    filtered_bad_trades = res_df[(res_df['timesfm_binary'] == 1) & (~res_df['meta_passed']) & (res_df['actual_return_pct'] <= 0)]
    noise_filter_ratio = (len(filtered_bad_trades) / len(res_df[res_df['timesfm_binary'] == 1])) * 100.0 if len(res_df[res_df['timesfm_binary'] == 1]) > 0 else 0.0

    print("\n==================================================================================")
    print("📊 TABLEAU COMPARATIF DES PERFORMANCES SUR LE JEU DE TEST 5M")
    print("==================================================================================")
    print(f"{'Métrique / Stratégie':<32} | {'Buy & Hold ETH':<16} | {'TimesFM Seul':<16} | {'Combo TimesFM+XGB':<16}")
    print("-" * 88)
    print(f"{'Nombre de Trades Déclenchés':<32} | {'N/A':<16} | {len(tfm_trades):<16} | {len(ens_trades):<16}")
    print(f"{'Précision / Win Rate %':<32} | {'N/A':<16} | {tfm_win_rate:.2f}%{'':<10} | {ens_win_rate:.2f}%{'':<10}")
    print(f"{'Rendement Cumulé %':<32} | {buy_hold_return:+.2f}%{'':<9} | {tfm_total_return:+.2f}%{'':<9} | {ens_total_return:+.2f}%{'':<9}")
    print(f"{'Taux de Faux Signaux Filtrés':<32} | {'N/A':<16} | {'0.00%':<16} | {noise_filter_ratio:.2f}%{'':<10}")
    print("==================================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation de l'Ensemble TimesFM + Meta-Labeler")
    parser.add_argument("--days", type=int, default=15, help="Jours de données d'évaluation 5m")
    args = parser.parse_args()
    
    evaluate_ensemble(days_eval=args.days)
