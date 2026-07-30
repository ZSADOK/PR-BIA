"""
Script d'Expérimentation & Comparaison Quantitative :
1. Spécialisation Profonde sur Un Seul Actif (Ex: NVDA) avec Calibration sur-mesure du Bruit.
2. Calibration Adaptative par Actif sur Univers Réduit.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, roc_auc_score

print("=" * 80)
print(" 🔬 EXPÉRIMENTATION : SPÉCIALISATION ACTIF UNIQUE vs ADAPTATION PAR ACTIF")
print("=" * 80)

TICKER = "NVDA"
data = yf.download(TICKER, start="2018-01-01", interval="1d", progress=False)

# 1. Calibration du Bruit Spécifique à NVDA
close = data["Close"][TICKER].dropna() if isinstance(data["Close"], pd.DataFrame) else data["Close"].dropna()
vol = data["Volume"][TICKER].dropna() if isinstance(data["Volume"], pd.DataFrame) else data["Volume"].dropna()
log_returns = np.log(close / close.shift(1))

# Volatilité et Variance Spécifique à NVDA
nvda_vol_21d = log_returns.rolling(21).std() * np.sqrt(252)
nvda_snr = 10 * np.log10(np.var(close.rolling(21).mean().dropna()) / np.var(close - close.rolling(21).mean().dropna()))

print(f"\n📊 PROFIL DE BRUIT SPÉCIFIQUE À {TICKER} :")
print(f"  • Rapport Signal/Bruit (SNR) : {float(nvda_snr):.2f} dB")
print(f"  • Volatilité Moyenne Annualisée : {float(nvda_vol_21d.mean())*100:.1f}%")
print(f"  • Ratio Bruit Micro-Intraday : {float(log_returns.std())*100:.2f}% / jour")

# Features Calibrées sur-mesure pour NVDA (Adaptées à sa forte volatilité)
feat = pd.DataFrame(index=close.index)
feat["trend_dev_custom"] = (close - close.rolling(21).mean()) / close.rolling(21).mean()
feat["vol_normalized"] = (log_returns.rolling(5).std() - log_returns.rolling(63).mean()) / (log_returns.rolling(63).std() + 1e-8)
feat["volume_zscore"] = (vol - vol.rolling(21).mean()) / (vol.rolling(21).std() + 1e-8)
feat["target_10d"] = (log_returns.shift(-10).rolling(10).sum() > 0.015).astype(int)

df_clean = feat.dropna()
split = int(len(df_clean) * 0.8)

train_df = df_clean.iloc[:split]
test_df = df_clean.iloc[split:]

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=25, weights="distance")
clf.fit(train_df.drop(columns=["target_10d"]), train_df["target_10d"])

probs = clf.predict_proba(test_df.drop(columns=["target_10d"]))[:, 1]
preds = (probs > 0.52).astype(int)

acc = accuracy_score(test_df["target_10d"], preds)
auc = roc_auc_score(test_df["target_10d"], probs)

print(f"\n🏆 RÉSULTAT DU MODÈLE SPÉCIALISÉ UNIQUE SUR {TICKER} (2018 - 2026) :")
print(f"  • Précision de Prédiction : {acc*100:.2f}%")
print(f"  • Score ROC-AUC : {auc:.3f}")
print("=" * 80)
