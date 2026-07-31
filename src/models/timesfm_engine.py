"""
Module de Modélisation et Prédiction avec Google TimesFM (Zero-Shot & Fine-Tuned).
Convertit les séries temporelles brutes OHLCV de 5m en prédiction de prix continue
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
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

FINETUNED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "timesfm_eth_finetuned.pt")

if HAS_TORCH:
    class PyTorchTimesFMModel(nn.Module):
        def __init__(self, context_len: int = 512, d_model: int = 256, nhead: int = 8, num_layers: int = 4):
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            self.pos_encoder = nn.Parameter(torch.randn(1, context_len, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, dropout=0.1)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            h = self.input_proj(x) + self.pos_encoder[:, :x.size(1), :]
            out = self.transformer(h)
            pred = self.head(out[:, -1, :])
            return pred

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
        # 1. Tentative d'importation dynamique native TimesFM
        try:
            import timesfm
            if hasattr(timesfm, 'TimesFm'):
                TimesFmClass = timesfm.TimesFm
            elif hasattr(timesfm, 'TimesFM'):
                TimesFmClass = timesfm.TimesFM
            else:
                from timesfm import TimesFm as TimesFmClass
                
            logger.info("Chargement du modèle de fondation Google TimesFM...")
            self.model = TimesFmClass(
                context_len=self.context_len,
                horizon_len=self.horizon_len,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
                backend=self.backend
            )
            self.model.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
        except Exception as e:
            logger.warning(f"Note : Inférence native TimesFM ({e}). Utilisation du PyTorch Transformer Engine.")
            if HAS_TORCH:
                self.model = PyTorchTimesFMModel(context_len=self.context_len)
                if self.backend == "cuda" and torch.cuda.is_available():
                    self.model.to("cuda")
                self.model.eval()

        # 2. Chargement des Poids Fine-Tuned (.pt) s'ils existent
        if HAS_TORCH and os.path.exists(FINETUNED_PATH):
            try:
                logger.info(f"Détection et chargement des poids Fine-Tuned : {FINETUNED_PATH}")
                state_dict = torch.load(FINETUNED_PATH, map_location=self.backend)
                
                if hasattr(self, 'model') and self.model is not None:
                    if hasattr(self.model, '_model'):
                        self.model._model.load_state_dict(state_dict, strict=False)
                    elif isinstance(self.model, PyTorchTimesFMModel):
                        self.model.load_state_dict(state_dict, strict=False)
                    self.is_finetuned = True
                    logger.info("Poids Fine-Tuned chargés avec succès !")
            except Exception as e_ft:
                logger.warning(f"Impossible d'injecter les poids fine-tunés ({e_ft}).")

    def predict_next_price(self, close_series: pd.Series) -> float:
        """Prédit le prix de clôture H+1 (5m)."""
        if len(close_series) < 30:
            raise ValueError(f"Pas assez de données historiques. Obtenu: {len(close_series)}, requis minimum: 30")
        
        context_data = close_series.iloc[-self.context_len:].values.astype(np.float32)
        
        if self.model is not None:
            try:
                if hasattr(self.model, 'forecast'):
                    forecast_input = [context_data]
                    forecast_results, _ = self.model.forecast(forecast_input, freq=[0])
                    return float(forecast_results[0][0])
                elif isinstance(self.model, PyTorchTimesFMModel) and HAS_TORCH:
                    mean = np.mean(context_data)
                    std = np.std(context_data) + 1e-8
                    norm_ctx = (context_data - mean) / std
                    
                    tensor_ctx = torch.tensor(norm_ctx, dtype=torch.float32).unsqueeze(0)
                    if self.backend == "cuda" and torch.cuda.is_available():
                        tensor_ctx = tensor_ctx.to("cuda")
                        
                    with torch.no_grad():
                        norm_pred = self.model(tensor_ctx).item()
                        
                    pred_price = (norm_pred * std) + mean
                    return float(pred_price)
            except Exception as ex:
                logger.error(f"Erreur lors de l'inférence native: {ex}. Fallback.")

        # Fallback statistique
        recent_changes = np.diff(context_data[-10:])
        weights = np.exp(np.linspace(-1, 0, len(recent_changes)))
        weighted_delta = np.average(recent_changes, weights=weights)
        
        last_price = float(context_data[-1])
        return float(last_price + weighted_delta)

    def generate_signal(self, df: pd.DataFrame, screener_passed: bool = True) -> Dict[str, Any]:
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
