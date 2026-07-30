"""
Script de Validation de la Métrique Custom Crypto Utilité Asymétrique (CAUM)
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models import CryptoCustomUtilityMetric

print("=" * 80)
print(" 🔬 VALIDATION DE LA MÉTRIQUE CUSTOM CRYPTO (Crypto Asymmetric Utility Metric)")
print("=" * 80)

CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"]
data = yf.download(CRYPTO_TICKERS, start="2021-01-01", interval="1d", group_by="ticker", progress=False)

metric_evaluator = CryptoCustomUtilityMetric(target_profit_pct=3.0, stop_loss_pct=1.5)

results = []

for t in CRYPTO_TICKERS:
    try:
        close = data[t]["Close"].dropna()
        log_ret = np.log(close / close.shift(1))
        actual_5d_returns = log_ret.shift(-5).rolling(5).sum().dropna()
        
        # Inférence simulée de probabilités
        np.random.seed(42)
        probs = np.random.uniform(0.40, 0.85, size=len(actual_5d_returns))
        
        metrics = metric_evaluator.compute_asymmetric_utility(probs, actual_5d_returns.values)
        quality_score = metric_evaluator.evaluate_model_signal_quality(probs, None, actual_5d_returns.values)
        
        results.append({
            "Cryptomonnaie": t,
            "Profit Factor": f"{metrics['profit_factor']:.2f}",
            "Sharpe Crypto (24/7)": f"{metrics['sharpe_ratio_crypto']:.2f}",
            "Win Rate (%)": f"{metrics['win_rate']:.1f}%",
            "Score Utilité CAUM": f"{metrics['crypto_utility_score']:.2f}",
            "Quality Index": f"{quality_score:.3f}"
        })
    except Exception as e:
        print(f"Erreur pour {t}: {e}")

print("\n🏆 EVALUATION DU SCORE D'UTILITÉ CUSTOM SUR L'UNIVERS CRYPTO MAJEUR :")
print("=" * 80)
print(pd.DataFrame(results).to_string(index=False))
print("=" * 80)
