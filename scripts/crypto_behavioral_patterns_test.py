"""
Script d'Analyse Quantitatives des Patterns Comportementaux & Volatilité Crypto (BTC, ETH, SOL, AVAX)
Mesure la répétabilité des structures de breakout et la force des signaux TabFM / TimesFM sur les Cryptomonnaies.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score

print("=" * 80)
print(" 🔬 EXPERIMENTATION CRYPTO : PATTERNS COMPORTEMENTAUX & VOLATILITÉ (2020 - 2026)")
print("=" * 80)

CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]
data = yf.download(CRYPTO_TICKERS, start="2020-01-01", interval="1d", group_by="ticker", progress=False)

def analyze_crypto_behavioral_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    close = df["Close"].dropna()
    volume = df["Volume"].dropna()
    log_ret = np.log(close / close.shift(1))
    
    # Indicateurs de Liquidation & Momentum Comportemental
    feat = pd.DataFrame(index=close.index)
    
    # 1. Amplitude de Volatilité Intraday (High-Low Range)
    high_low_ratio = (df["High"] - df["Low"]) / close
    feat["volatility_range_5d"] = high_low_ratio.rolling(5).mean()
    
    # 2. Déviation de Tendance par rapport aux Moyennes Mobiles Intraday 24/7
    feat["trend_dev_7d"] = (close - close.rolling(7).mean()) / close.rolling(7).mean()
    feat["trend_dev_30d"] = (close - close.rolling(30).mean()) / close.rolling(30).mean()
    
    # 3. Z-Score de Volume & Pression de Liquidation
    feat["volume_zscore"] = (volume - volume.rolling(14).mean()) / (volume.rolling(14).std() + 1e-8)
    feat["momentum_7d"] = log_ret.rolling(7).sum()
    
    # Cible : Breakout Haussier Majeur à 5 Jours (Gain > +3.0%)
    future_ret_5d = log_ret.shift(-5).rolling(5).sum()
    feat["target_crypto_breakout"] = (future_ret_5d > 0.03).astype(int)
    
    return feat.dropna()

results = []

for t in CRYPTO_TICKERS:
    try:
        df_t = data[t]
        feat = analyze_crypto_behavioral_features(df_t, t)
        
        split = int(len(feat) * 0.8)
        train = feat.iloc[:split]
        test = feat.iloc[split:]
        
        X_tr = train.drop(columns=["target_crypto_breakout"])
        y_tr = train["target_crypto_breakout"].values
        X_te = test.drop(columns=["target_crypto_breakout"])
        y_te = test["target_crypto_breakout"].values
        
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=25, weights="distance")
        clf.fit(X_tr, y_tr)
        
        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs > 0.54).astype(int)
        
        acc = accuracy_score(y_te, preds)
        auc = roc_auc_score(y_te, probs)
        prec = precision_score(y_te, preds, zero_division=0)
        
        results.append({
            "Cryptomonnaie": t,
            "Rendement Cible (5j)": "> +3.0%",
            "Précision Modèle": f"{acc*100:.2f}%",
            "Score ROC-AUC": f"{auc:.3f}",
            "Précision Signaux Achat": f"{prec*100:.2f}%"
        })
    except Exception as e:
        print(f"Erreur pour {t}: {e}")

print("\n🏆 RÉSULTATS DU TEST COMPORTEMENTAL SUR CRYPTO 24/7 (2020 - 2026) :")
print("=" * 80)
print(pd.DataFrame(results).to_string(index=False))
print("=" * 80)
