"""
Métrique Spécialisée et Custom Loss pour le Trading Crypto Comportemental (Crypto Asymmetric Utility Metric - CAUM).
Combine l'Espérance Mathématique Asymétrique, la Pénalité Convex de Drawdown et la Ratio Signal/Bruit.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

class CryptoCustomUtilityMetric:
    """
    Métrique Custom Institutionnelle dédiée aux Cryptomonnaies :
    Pondère l'asymétrie des gains (Breakouts +5% à +15%), pénalise l'enfoncement sous q10,
    et calcule la fonction d'utilité espérée de Kelly.
    """

    def __init__(self, target_profit_pct: float = 3.0, stop_loss_pct: float = 1.5, lambda_upside: float = 1.5, gamma_downside: float = 2.0):
        self.target_profit_pct = target_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.lambda_upside = lambda_upside
        self.gamma_downside = gamma_downside

    def compute_asymmetric_utility(self, predicted_probs: np.ndarray, actual_returns: np.ndarray) -> Dict[str, float]:
        """
        Calcule le score d'Utilité Asymétrique Crypto sur un jeu de prédictions.
        """
        trade_mask = predicted_probs >= 0.58
        if not np.any(trade_mask):
            return {"crypto_utility_score": 0.0, "profit_factor": 0.0, "expectancy_usd": 0.0, "win_rate": 0.0}

        selected_returns = actual_returns[trade_mask]
        
        # 1. Gain Asymétrique Pondéré
        gains = selected_returns[selected_returns > 0]
        losses = np.abs(selected_returns[selected_returns < 0])

        total_gain = np.sum(gains * (1.0 + self.lambda_upside * (gains / (self.target_profit_pct / 100.0)))) if len(gains) > 0 else 0.0
        
        # 2. Pénalité Convexe de Drawdown (Pénalité Quadratique sur les pertes excédant le Stop-Loss)
        excess_losses = np.maximum(0.0, losses - (self.stop_loss_pct / 100.0))
        total_loss_penalty = np.sum(losses + self.gamma_downside * (excess_losses ** 2)) if len(losses) > 0 else 1e-8

        # 3. Profit Factor et Score d'Utilité Globale
        profit_factor = total_gain / max(1e-8, total_loss_penalty)
        win_rate = (len(gains) / len(selected_returns)) * 100.0 if len(selected_returns) > 0 else 0.0
        
        avg_return = float(np.mean(selected_returns)) * 100.0
        volatility = float(np.std(selected_returns)) * 100.0 + 1e-8
        sharpe_ratio = (avg_return / volatility) * np.sqrt(365) # Crypto 365j 24/7

        utility_score = sharpe_ratio * profit_factor * (win_rate / 50.0)

        return {
            "crypto_utility_score": float(utility_score),
            "profit_factor": float(profit_factor),
            "expectancy_pct": float(avg_return),
            "win_rate": float(win_rate),
            "sharpe_ratio_crypto": float(sharpe_ratio)
        }

    def evaluate_model_signal_quality(self, probs: np.ndarray, y_true: np.ndarray, returns: np.ndarray) -> float:
        """
        Score Synthétique unique [0.0 - 1.0] pour l'optimisation des modèles.
        """
        metrics = self.compute_asymmetric_utility(probs, returns)
        # Normalisation Sigmoïde du score d'utilité
        score = 1.0 / (1.0 + np.exp(-0.5 * (metrics["crypto_utility_score"] - 1.5)))
        return float(score)
