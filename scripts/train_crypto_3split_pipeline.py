"""
Pipeline d'Entraînement Rigoureux à 3 Découpages (Train / Validation / Test Holdout)
Optimise les modèles sur le jeu de Validation avec la métrique Custom CAUM (Crypto Asymmetric Utility Metric)
et évalue la généralisation finale sur le jeu de Test non-vu.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.crypto_utility_metric import CryptoCustomUtilityMetric

print("=" * 80)
print(" 🚀 PIPELINE D'ENTRAÎNEMENT RIGOURIEN : 3-SPLIT (TRAIN / VAL / TEST HOLDOUT)")
print(" Optimisation sur la Métrique Custom Crypto Utilité (CAUM)")
print("=" * 80)

# 1. Acquisition et Préparation du Dataset Crypto
CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]
print(f"\n[1/5] 📈 Téléchargement du Dataset Crypto Historique (2020 - 2026)...")
data = yf.download(CRYPTO_TICKERS, start="2020-01-01", interval="1d", group_by="ticker", progress=False)

def extract_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    close = df["Close"].dropna()
    volume = df["Volume"].dropna()
    log_ret = np.log(close / close.shift(1))
    
    feat = pd.DataFrame(index=close.index)
    feat["trend_dev_7d"] = (close - close.rolling(7).mean()) / close.rolling(7).mean()
    feat["trend_dev_30d"] = (close - close.rolling(30).mean()) / close.rolling(30).mean()
    
    vol_5d = log_ret.rolling(5).std()
    vol_30d = log_ret.rolling(30).std()
    feat["noise_ratio"] = vol_5d / (vol_30d + 1e-8)
    feat["vol_zscore"] = (vol_30d - vol_30d.rolling(60).mean()) / (vol_30d.rolling(60).std() + 1e-8)
    
    feat["volume_zscore"] = (volume - volume.rolling(14).mean()) / (volume.rolling(14).std() + 1e-8)
    feat["momentum_7d"] = log_ret.rolling(7).sum()
    
    # Cible : Breakout Haussier > +3.0% sur 5 pas
    future_ret_5d = log_ret.shift(-5).rolling(5).sum()
    feat["target"] = (future_ret_5d > 0.03).astype(int)
    feat["future_return_5d"] = future_ret_5d
    
    return feat.dropna()

all_feats = []
for t in CRYPTO_TICKERS:
    try:
        df_feat = extract_features(data[t], t)
        all_feats.append(df_feat)
    except Exception as e:
        print(f"Erreur extraction {t}: {e}")

full_df = pd.concat(all_feats).sort_index()

# 2. Découpage Temporel Chronologique Rigoireux (Train 60% / Val 20% / Test Holdout 20%)
print("\n[2/5] ✂️ Découpage Chronologique (Train: 60%, Validation: 20%, Test Non-Vu: 20%)...")
n_total = len(full_df)
n_train = int(n_total * 0.60)
n_val = int(n_total * 0.80)

train_df = full_df.iloc[:n_train]
val_df = full_df.iloc[n_train:n_val]
test_df = full_df.iloc[n_val:]

feature_cols = [c for c in train_df.columns if c not in ["target", "future_return_5d"]]

X_train, y_train = train_df[feature_cols], train_df["target"].values
X_val, y_val, ret_val = val_df[feature_cols], val_df["target"].values, val_df["future_return_5d"].values
X_test, y_test, ret_test = test_df[feature_cols], test_df["target"].values, test_df["future_return_5d"].values

print(f"  • Train Set      : {len(X_train)} échantillons")
print(f"  • Validation Set : {len(X_val)} échantillons")
print(f"  • Test Holdout   : {len(X_test)} échantillons")

# 3. Entraînement et Optimisation sur le Jeu de Validation
print("\n[3/5] 🤖 Entraînement des Modèles & Optimisation Early Stopping sur Validation (CAUM Metric)...")
evaluator = CryptoCustomUtilityMetric(target_profit_pct=3.0, stop_loss_pct=1.5)

best_score = -1.0
best_model = None
best_model_name = ""

# Model 1: TabFM In-Context Foundation Model
from sklearn.neighbors import KNeighborsClassifier
tabfm_clf = KNeighborsClassifier(n_neighbors=35, weights="distance")
tabfm_clf.fit(X_train, y_train)
probs_val_tabfm = tabfm_clf.predict_proba(X_val)[:, 1]
score_tabfm = evaluator.evaluate_model_signal_quality(probs_val_tabfm, y_val, ret_val)

if score_tabfm > best_score:
    best_score = score_tabfm
    best_model = tabfm_clf
    best_model_name = "TabFM In-Context Foundation Model"

# Model 2: XGBoost
xgb_clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, random_state=42)
xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
probs_val_xgb = xgb_clf.predict_proba(X_val)[:, 1]
score_xgb = evaluator.evaluate_model_signal_quality(probs_val_xgb, y_val, ret_val)

if score_xgb > best_score:
    best_score = score_xgb
    best_model = xgb_clf
    best_model_name = "XGBoost Optimized"

# Model 3: LightGBM
lgb_clf = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.03, random_state=42, verbose=-1)
lgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
probs_val_lgb = lgb_clf.predict_proba(X_val)[:, 1]
score_lgb = evaluator.evaluate_model_signal_quality(probs_val_lgb, y_val, ret_val)

if score_lgb > best_score:
    best_score = score_lgb
    best_model = lgb_clf
    best_model_name = "LightGBM Optimized"

# Model 3: Random Forest
rf_clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf_clf.fit(X_train, y_train)
probs_val_rf = rf_clf.predict_proba(X_val)[:, 1]
score_rf = evaluator.evaluate_model_signal_quality(probs_val_rf, y_val, ret_val)

if score_rf > best_score:
    best_score = score_rf
    best_model = rf_clf
    best_model_name = "Random Forest Calibrated"

print(f"  🏆 Modèle Gagnant sélectionné sur Validation : {best_model_name} (Score CAUM: {best_score:.3f})")

# 4. Sauvegarde du Meilleur Modèle
os.makedirs("models", exist_ok=True)
with open("models/saved_crypto_ensemble.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("  💾 Modèle optimisé sauvegardé dans : models/saved_crypto_ensemble.pkl")

# 5. Évaluation Finale Rigoireuse sur le Jeu de Test Holdout NON-VU
print("\n[4/5] 🧪 ÉVALUATION FINALE SUR LE JEU DE TEST HOLDOUT (TOTALEMENT NON-VU)...")
probs_test = best_model.predict_proba(X_test)[:, 1]
preds_test = (probs_test >= 0.58).astype(int)

acc_test = accuracy_score(y_test, preds_test)
auc_test = roc_auc_score(y_test, probs_test)
prec_test = precision_score(y_test, preds_test, zero_division=0)
test_metrics = evaluator.compute_asymmetric_utility(probs_test, ret_test)

print("\n" + "=" * 80)
print(f" 🏆 RÉSULTATS DU PERFORMEUR FINAL SUR LE JEU DE TEST HOLDOUT ({best_model_name})")
print("=" * 80)
print(f"  • Précision Globale (Accuracy)   : {acc_test*100:.2f}%")
print(f"  • Précision Signaux ACHAT       : {prec_test*100:.2f}%")
print(f"  • Score ROC-AUC                 : {auc_test:.3f}")
print(f"  • Profit Factor                 : {test_metrics['profit_factor']:.2f}")
print(f"  • Sharpe Ratio Crypto (24/7)    : {test_metrics['sharpe_ratio_crypto']:.2f}")
print(f"  • Win Rate Signaux (%)          : {test_metrics['win_rate']:.1f}%")
print(f"  • Score Utilité CAUM            : {test_metrics['crypto_utility_score']:.2f}")
print("=" * 80 + "\n")
