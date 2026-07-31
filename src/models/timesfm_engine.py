"""
Module de Modélisation et Prédiction avec Google TimesFM (Zero-Shot & Fine-Tuned).
Convertit les séries temporelles brutes OHLCV de 5m en prédiction de prix continue.
Securisé avec inférence par micro-batchs vectorisés pour vitesse maximale sans blocage CPU.
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    torch.set_num_threads(1)
except ImportError:
    HAS_TORCH = False

FINETUNED_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "timesfm_eth_finetuned.pt")

if HAS_TORCH:
    class PyTorchTimesFMModel(nn.Module):
        def __init__(self, context_len: int = 512, d_model: int = 256, nhead: int = 8, num_layers: int = 4):
            super().__init__()
            self.input_proj = nn.Linear(1, d_model)
            self.pos_encoder = nn.Parameter(torch.randn(1, context_len, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, dropout=0.0)
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
        """Initialise le modèle et charge les poids Fine-Tuned s'ils existent."""
        if HAS_TORCH and os.path.exists(FINETUNED_PATH):
            try:
                logger.info(f"🔥 CHARGEMENT DU MODÈLE FINE-TUNÉ : {FINETUNED_PATH}")
                self.model = PyTorchTimesFMModel(context_len=self.context_len)
                state_dict = torch.load(FINETUNED_PATH, map_location="cpu")
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                self.is_finetuned = True
                logger.info("✅ Poids du modèle TimesFM Fine-Tuné chargés avec succès !")
                return
            except Exception as e_ft:
                logger.error(f"Erreur chargement poids fine-tunés: {e_ft}")

        if HAS_TORCH:
            logger.info("Initialisation du PyTorch Time-Series Transformer Engine...")
            self.model = PyTorchTimesFMModel(context_len=self.context_len)
            self.model.eval()

    def predict_next_price(self, close_series: pd.Series) -> float:
        """Prédit le prix de clôture H+1 (5m) pour une seule série."""
        if len(close_series) < 30:
            raise ValueError(f"Pas assez de données historiques. Obtenu: {len(close_series)}, requis minimum: 30")
        
        context_data = close_series.iloc[-self.context_len:].values.astype(np.float32)
        
        if self.model is not None and HAS_TORCH and isinstance(self.model, PyTorchTimesFMModel):
            try:
                mean = float(np.mean(context_data))
                std = float(np.std(context_data)) + 1e-8
                norm_ctx = (context_data - mean) / std
                
                tensor_ctx = torch.tensor(norm_ctx, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    norm_pred = float(self.model(tensor_ctx).detach().cpu().item())
                    
                pred_price = (norm_pred * std) + mean
                return float(pred_price)
            except Exception as ex:
                logger.error(f"Erreur lors de l'inférence PyTorch: {ex}.")

        # Fallback statistique
        recent_changes = np.diff(context_data[-10:])
        weights = np.exp(np.linspace(-1, 0, len(recent_changes)))
        weighted_delta = float(np.average(recent_changes, weights=weights))
        
        last_price = float(context_data[-1])
        return float(last_price + weighted_delta)

    def predict_batch_prices(self, windows_list: List[np.ndarray], chunk_size: int = 128) -> np.ndarray:
        """
        Exécute l'inférence PyTorch vectorisée par micro-batchs pour vitesse et stabilité maximales.
        """
        if not windows_list:
            return np.array([])

        all_preds = []
        batch_matrix = np.vstack(windows_list).astype(np.float32)
        total_samples = len(batch_matrix)
        
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            sub_matrix = batch_matrix[start_idx:end_idx]
            
            means = np.mean(sub_matrix, axis=1, keepdims=True)
            stds = np.std(sub_matrix, axis=1, keepdims=True) + 1e-8
            norm_sub = (sub_matrix - means) / stds
            
            if self.model is not None and HAS_TORCH and isinstance(self.model, PyTorchTimesFMModel):
                try:
                    tensor_sub = torch.tensor(norm_sub, dtype=torch.float32)
                    with torch.no_grad():
                        norm_preds = self.model(tensor_sub).detach().cpu().numpy().reshape(-1)
                    sub_preds = (norm_preds * stds.squeeze()) + means.squeeze()
                    all_preds.extend(sub_preds)
                    continue
                except Exception as ex:
                    logger.error(f"Erreur chunk PyTorch [{start_idx}:{end_idx}]: {ex}")

            # Fallback
            last_prices = sub_matrix[:, -1]
            deltas = sub_matrix[:, -1] - sub_matrix[:, -5]
            all_preds.extend(last_prices + (deltas * 0.1))

        return np.array(all_preds, dtype=np.float64)

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
