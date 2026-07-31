"""
Module de Modélisation et Prédiction avec Google TimesFM (Zero-Shot & Fine-Tuned).
Convertit les séries temporelles brutes OHLCV de 1h en prédiction de prix continue
et génère un signal binaire de trading (1 = Achat/Long, 0 = Neutre/Vente/Short).
Charge automatiquement les poids fine-tunés depuis models/timesfm_eth_finetuned.pt s'ils existent.
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

FINETUNED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "timesfm_eth_finetuned.pt")

class TimesFMEngine:
    def __init__(self, context_len: int = 512, horizon_len: int = 1, backend: str = "cpu"):
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.backend = backend
        self.model = None
        self.is_finetuned = False
        self._init_model()

    def _init_model(self):
        """Initialise le modèle TimesFM et charge les poids fine-tunés Colab s'ils existent."""
        try:
            import timesfm
            logger.info("Chargement du modèle de fondation TimesFM...")
            self.model = timesfm.TimesFm(
                context_len=self.context_len,
                horizon_len=self.horizon_len,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
                backend=self.backend
            )
            # Chargement du checkpoint pré-entraîné de base
            self.model.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
            
            # Verification et chargement des poids fine-tunés Colab
            if HAS_TORCH and os.path.exists(FINETUNED_PATH):
                try:
                    logger.info(f"Détection des poids Fine-Tuned Colab : {FINETUNED_PATH}")
                    state_dict = torch.load(FINETUNED_PATH, map_location=self.backend)
                    if hasattr(self.model, '_model') and isinstance(state_dict, dict):
                        self.model._model.load_state_dict(state_dict, strict=False)
                        self.is_finetuned = True
                        logger.info("Poids Fine-Tuned chargés avec succès !")
                except Exception as e_ft:
                    logger.warning(f"Impossible d'injecter les poids fine-tunés ({e_ft}). Utilisation du modèle Zero-Shot par défaut.")

            logger.info("Modèle TimesFM prêt pour inférence.")
        except Exception as e:
            logger.warning(f"Impossible de charger la librairie native 'timesfm' ({e}). Utilisation du moteur d'inférence statistique et de fondation de fallback.")
            self.model = None

    def predict_next_price(self, close_series: pd.Series) -> float:
        """
        Prédit le prix du clôture (Close) de la bougie H+1.
        close_series: Série des prix historiques (longueur recommandée >= context_len).
        """
        if len(close_series) < 30:
            raise ValueError(f"Pas assez de données historiques. Obtenu: {len(close_series)}, requis minimum: 30")
        
        context_data = close_series.iloc[-self.context_len:].values.astype(np.float32)
        
        if self.model is not None:
            try:
                forecast_input = [context_data]
                forecast_results, _ = self.model.forecast(forecast_input, freq=[0])
                predicted_price = float(forecast_results[0][0])
                return predicted_price
            except Exception as ex:
                logger.error(f"Erreur lors de l'inférence native TimesFM: {ex}. Basculement sur fallback.")

        # Inférence de fallback (Exponential Smoothing / Momentum Invariance Ensembling)
        recent_changes = np.diff(context_data[-10:])
        weights = np.exp(np.linspace(-1, 0, len(recent_changes)))
        weighted_delta = np.average(recent_changes, weights=weights)
        
        last_price = float(context_data[-1])
        predicted_price = last_price + weighted_delta
        return float(predicted_price)

    def generate_signal(self, df: pd.DataFrame, screener_passed: bool = True) -> Dict[str, Any]:
        """
        Calcule la prédiction continue et produit le signal binaire :
        - signal = 1 (Achat / Long) si Retour Prédit > 0 ET screener_passed == True
        - signal = 0 (Neutre / Short) sinon.
        """
        close_prices = df['Close']
        current_price = float(close_prices.iloc[-1])
        predicted_price = self.predict_next_price(close_prices)
        
        predicted_return = (predicted_price - current_price) / current_price
        
        is_bullish_forecast = predicted_return > 0.0005
        signal_binary = 1 if (is_bullish_forecast and screener_passed) else 0
        
        confidence = min(0.95, max(0.50, 0.50 + abs(predicted_return) * 50)) if signal_binary == 1 else 0.40
        
        return {
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_return_pct": predicted_return * 100.0,
            "screener_passed": screener_passed,
            "signal_binary": signal_binary,
            "signal_label": "BUY (LONG)" if signal_binary == 1 else "SELL / NEUTRAL",
            "confidence": confidence,
            "is_finetuned": self.is_finetuned
        }
