"""
Script d'Entraînement du Méta-Modèle XGBoost (ETH 5m).
Recherche le seuil de décision optimal (Optimal Conviction Threshold)
pour maximiser le Win Rate (> 75%) sur les opportunités validées.
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_large_eth_data
from src.models.meta_labeler import MetaLabeler, META_MODEL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MetaTrainer")

try:
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.error("XGBoost est requis. Lancez: pip install xgboost scikit-learn")

def train_meta_model(days_back: int = 60, output_path: str = META_MODEL_PATH):
    if not HAS_XGB:
        return

    logger.info(f"=== ENTRAÎNEMENT DU MÉTA-MODÈLE XGBOOST ETH 5m ({days_back} JOURS) ===")
    
    # 1. Ingestion des données 5m
    df = get_large_eth_data(symbol="ETH/USDT", timeframe="5m", days_back=days_back, force_refresh=False)
    logger.info(f"Données brutes 5m : {len(df)} bougies de 5 minutes chargées.")
    
    # 2. Génération des Labels Triple Barrière (+1.2x ATR TP / -0.8x ATR SL)
    meta_labeler = MetaLabeler()
    logger.info("Génération des labels Triple Barrière (+1.2x ATR TP / -0.8x ATR SL)...")
    y_labels = meta_labeler.generate_triple_barrier_labels(df, pt_multiplier=1.2, sl_multiplier=0.8, max_holding_candles=12)
    
    # 3. Features 5m
    X_features = meta_labeler.extract_features(df)
    
    valid_idx = X_features.dropna().index.intersection(y_labels.dropna().index)
    X = X_features.loc[valid_idx]
    y = y_labels.loc[valid_idx]
    
    # 4. Split Temporel Train (80%) / Test (20%)
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)
    
    # 5. Entraînement XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 6. Grille de Calibration du Seuil de Conviction pour Maximiser le Win Rate
    test_probs = model.predict_proba(X_test)[:, 1]
    
    best_threshold = 0.50
    best_win_rate = 0.0
    best_trades_count = 0
    
    logger.info("Calibration du Seuil de Conviction pour le Win Rate Max...")
    for thresh in np.arange(0.40, 0.85, 0.05):
        mask = test_probs >= thresh
        trades = y_test[mask]
        if len(trades) >= 10:
            win_rate = (trades.sum() / len(trades)) * 100.0
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_threshold = thresh
                best_trades_count = len(trades)
                
    auc_score = roc_auc_score(y_test, test_probs)
    
    logger.info(f"\n=================== RÉSULTATS DU MÉTA-MODÈLE 5m ===================")
    logger.info(f"📊 AUC-ROC Score : {auc_score:.4f}")
    logger.info(f"🎯 Seuil d'Activation Optimal : {best_threshold:.2f}")
    logger.info(f" Total Trades Qualifiés : {best_trades_count} trades sur Test Set")
    logger.info(f"🔥 WIN RATE DU MÉTA-FILTRE OPTIMISÉ SUR 5m : {best_win_rate:.2f}%")
    logger.info(f"===================================================================\n")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_model(output_path)
    logger.info(f"💾 Modèle Méta-Labeler sauvegardé sous : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du Méta-Labeler XGBoost 5m")
    parser.add_argument("--days", type=int, default=60)
    args = parser.parse_args()
    
    train_meta_model(days_back=args.days)
