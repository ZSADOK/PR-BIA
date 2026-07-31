"""
Script d'Entraînement & Fine-Tuning Ultra-Optimisé pour Google TimesFM 1.0 (ETH 1h).
Affiche la Loss (MSE), la Loss de Validation et la Précision Directionnelle (Win Rate %) à chaque époque.

Usage Colab :
!python scripts/train_timesfm.py --epochs 10 --days 730 --lr 5e-5
"""
import os
import sys
import argparse
import time
import logging
from datetime import datetime, timezone

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TimesFMTrainer")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object
    logger.error("PyTorch est requis. Lancez: pip install torch")

try:
    import ccxt
except ImportError:
    logger.error("CCXT est requis. Lancez: pip install ccxt")

# Tentative d'importation de la librairie TimesFM de Google
HAS_TIMESFM = False
TimesFmClass = None
try:
    import timesfm
    if hasattr(timesfm, 'TimesFm'):
        TimesFmClass = timesfm.TimesFm
    elif hasattr(timesfm, 'TimesFM'):
        TimesFmClass = timesfm.TimesFM
    HAS_TIMESFM = True
except Exception as e:
    logger.warning(f"Note : 'timesfm' natif non importé ({e}). Utilisation de l'architecture PyTorch Transformer TimesFM.")

# ==============================================================================
# ARCHITECTURE PYTORCH TIMESERIES TRANSFORMER (BACKBONE TIMESFM)
# ==============================================================================
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
            # x shape: [batch, context_len] -> [batch, context_len, 1]
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            h = self.input_proj(x) + self.pos_encoder[:, :x.size(1), :]
            out = self.transformer(h)
            pred = self.head(out[:, -1, :]) # Prédiction H+1
            return pred

# ==============================================================================
# DATASET PYTORCH POUR SÉRIES TEMPORELLES
# ==============================================================================
class TimesFMTimeSeriesDataset(Dataset):
    def __init__(self, prices: np.ndarray, context_len: int = 512, horizon_len: int = 1):
        self.prices = prices.astype(np.float32)
        self.context_len = context_len
        self.horizon_len = horizon_len

    def __len__(self):
        return len(self.prices) - self.context_len - self.horizon_len + 1

    def __getitem__(self, idx):
        context = self.prices[idx : idx + self.context_len]
        target = self.prices[idx + self.context_len : idx + self.context_len + self.horizon_len]

        mean = np.mean(context)
        std = np.std(context) + 1e-8

        norm_context = (context - mean) / std
        norm_target = (target - mean) / std

        return (
            torch.tensor(norm_context, dtype=torch.float32),
            torch.tensor(norm_target, dtype=torch.float32),
            torch.tensor(context[-1], dtype=torch.float32),
            torch.tensor(self.prices[idx + self.context_len], dtype=torch.float32)
        )

# ==============================================================================
# INGESTION DES DONNÉES ETH/USDT (CCXT BINANCE)
# ==============================================================================
def download_historical_data(symbol: str = "ETH/USDT", timeframe: str = "1h", days_back: int = 730) -> pd.DataFrame:
    logger.info(f"📥 Téléchargement de {days_back} jours (~{days_back*24} bougies 1h) pour {symbol} via Binance...")
    exchange = ccxt.binance({'enableRateLimit': True})
    now_ms = exchange.milliseconds()
    since_ms = now_ms - (days_back * 24 * 60 * 60 * 1000)

    all_ohlcv = []
    curr_since = since_ms

    while curr_since < now_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=curr_since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if last_ts <= curr_since:
                break
            curr_since = last_ts + 1
            time.sleep(0.05)
        except Exception as e:
            logger.error(f"Erreur téléchargement: {e}")
            break

    if not all_ohlcv:
        raise RuntimeError("Impossible de télécharger les données depuis CCXT Binance.")

    df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
    df.set_index('Timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    logger.info(f"✅ {len(df)} bougies 1h récupérées. Période: {df.index[0]} à {df.index[-1]}")
    return df

# ==============================================================================
# PIPELINE DE FINE-TUNING & AFFICHAGE DES MÉTRIQUES (LOSS / ACCURACY)
# ==============================================================================
def train(
    symbol: str = "ETH/USDT",
    days_back: int = 730,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 5e-5,
    context_len: int = 512,
    output_dir: str = "models"
):
    if not HAS_TORCH:
        raise ModuleNotFoundError("PyTorch est requis.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"⚡ Dispositif d'entraînement : {device}")
    if device.type == "cuda":
        logger.info(f"🔥 GPU Détecté : {torch.cuda.get_device_name(0)}")

    # 1. Téléchargement des Données
    df = download_historical_data(symbol=symbol, timeframe="1h", days_back=days_back)
    close_prices = df['Close'].values

    # 2. Split Temporel (85% Train, 15% Validation)
    split_idx = int(len(close_prices) * 0.85)
    train_dataset = TimesFMTimeSeriesDataset(close_prices[:split_idx], context_len=context_len, horizon_len=1)
    val_dataset = TimesFMTimeSeriesDataset(close_prices[split_idx:], context_len=context_len, horizon_len=1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"📊 Dataset configuré : {len(train_dataset)} échantillons Train | {len(val_dataset)} échantillons Validation.")

    # 3. Initialisation de l'architecture modèle
    model = None
    if HAS_TIMESFM and TimesFmClass is not None:
        try:
            logger.info("🧠 Chargement de Google TimesFM 1.0 (200M)...")
            tfm = TimesFmClass(
                context_len=context_len, horizon_len=1, input_patch_len=32,
                output_patch_len=128, num_layers=20, model_dims=1280,
                backend="gpu" if device.type == "cuda" else "cpu"
            )
            tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
            model = getattr(tfm, '_model', None)
        except Exception as ex:
            logger.warning(f"Moteur natif TimesFM non disponible ({ex}). Utilisation du PyTorch Transformer Engine.")
            model = None

    if model is None:
        logger.info("⚡ Utilisation du PyTorch Time-Series Transformer Engine (Backbone TimesFM)...")
        model = PyTorchTimesFMModel(context_len=context_len, d_model=256, nhead=8, num_layers=4)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "timesfm_eth_finetuned.pt")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.SmoothL1Loss() # Huber Loss

    best_val_loss = float('inf')
    logger.info(f"\n=================== DÉBUT DU FINE-TUNING ({epochs} ÉPOQUES) ===================")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_ctx, batch_tgt, _, _ in train_loader:
            batch_ctx, batch_tgt = batch_ctx.to(device), batch_tgt.to(device)
            optimizer.zero_grad()

            preds = model(batch_ctx)
            if preds.dim() == 1:
                preds = preds.unsqueeze(-1)
                
            loss = criterion(preds[:, -1:], batch_tgt)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # 4. Évaluation sur la Validation (Calcul Loss & Accuracy Directionnelle)
        model.eval()
        val_loss = 0.0
        correct_direction = 0
        total_eval = 0
        
        with torch.no_grad():
            for batch_ctx, batch_tgt, last_p, target_p in val_loader:
                batch_ctx, batch_tgt = batch_ctx.to(device), batch_tgt.to(device)
                preds = model(batch_ctx)
                if preds.dim() == 1:
                    preds = preds.unsqueeze(-1)
                    
                v_loss = criterion(preds[:, -1:], batch_tgt)
                val_loss += v_loss.item()
                
                # Précision Directionnelle (Win Rate %)
                pred_delta = preds[:, -1].cpu().numpy()
                actual_delta = target_p.numpy() - last_p.numpy()
                
                direction_match = ((pred_delta > 0) & (actual_delta > 0)) | ((pred_delta <= 0) & (actual_delta <= 0))
                correct_direction += direction_match.sum()
                total_eval += len(direction_match)

        val_loss /= len(val_loader)
        accuracy_pct = (correct_direction / total_eval) * 100.0 if total_eval > 0 else 0.0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            best_tag = "🔥 [NOUVEAU MEILLEUR MODÈLE SAUVEGARDÉ]"
        else:
            best_tag = ""

        logger.info(f"Époque [{epoch:02d}/{epochs:02d}] | Train Loss (MSE): {train_loss:.6f} | Val Loss: {val_loss:.6f} | Win Rate Directionnel: {accuracy_pct:.2f}% {best_tag}")

    logger.info(f"===========================================================================")
    logger.info(f"🎉 FINE-TUNING TERMINÉ avec succès !")
    logger.info(f"📌 Meilleure Loss de Validation : {best_val_loss:.6f}")
    logger.info(f"💾 Poids sauvegardés dans : {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-Tuning TimesFM ETH 1h")
    parser.add_argument("--symbol", type=str, default="ETH/USDT")
    parser.add_argument("--days", type=int, default=730, help="Jours d'historique (730d = 2 ans)")
    parser.add_argument("--epochs", type=int, default=10, help="Nombre d'époques")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille de batch")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning Rate")
    parser.add_argument("--output_dir", type=str, default="models")

    args = parser.parse_args()
    train(
        symbol=args.symbol,
        days_back=args.days,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir
    )
