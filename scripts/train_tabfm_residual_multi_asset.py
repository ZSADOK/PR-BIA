"""
Pipeline d'Entraînement SOTA Transformer Résiduel Multi-Actifs (25.8M Paramètres)
Optimisé avec Early Stopping Rapide, Pre-Screening Volume & Momentum, et Métrique CAUM.
(Pas de régime Double Descent sur-paramétré : Arrêt optimal strict sur le Jeu de Validation)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.crypto_utility_metric import CryptoCustomUtilityMetric

# 1. Pre-Screening Volume & Momentum (Skill Rule)
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def apply_triple_barrier_and_features(df: pd.DataFrame, pt_mult=1.5, sl_mult=1.0, max_holding=12, apply_prescreen=True, interval: str = "5m") -> pd.DataFrame:
    close_series = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]
    volume_series = df["Volume"].iloc[:, 0] if isinstance(df["Volume"], pd.DataFrame) else df["Volume"]
    
    close = close_series.dropna()
    volume = volume_series.dropna()
    log_ret = np.log(close / close.shift(1))
    
    close_vals = close.values.flatten()
    if len(close_vals) > 25:
        clean_signal = savgol_filter(close_vals, window_length=21, polyorder=2, mode='nearest')
    else:
        clean_signal = close_vals

    trend_dev = (close_vals - clean_signal) / (clean_signal + 1e-8)
    
    feat = pd.DataFrame(index=close.index)
    feat["trend_dev_smooth"] = trend_dev
    feat["ret_1step"] = log_ret
    feat["ret_6step"] = log_ret.rolling(6).sum()
    feat["ret_24step"] = log_ret.rolling(24).sum()
    
    vol_24step = log_ret.rolling(24).std()
    vol_168step = log_ret.rolling(168).std()
    feat["noise_ratio"] = vol_24step / (vol_168step + 1e-8)
    feat["vol_zscore"] = (vol_24step - vol_24step.rolling(168).mean()) / (vol_24step.rolling(168).std() + 1e-8)
    feat["volume_zscore"] = (volume - volume.rolling(24).mean()) / (volume.rolling(24).std() + 1e-8)
    
    # Indicateurs Pre-Screening (Skill Momentum Screener)
    steps_per_day = 1440 if interval == "1m" else (288 if interval == "5m" else 24)
    roll_w = min(len(volume) - 1, 20 * steps_per_day)
    vol_20d_mean = volume.rolling(roll_w, min_periods=50).mean()
    feat["rvol"] = volume / (vol_20d_mean + 1e-8)
    feat["sma50"] = close.rolling(50).mean()
    feat["sma200"] = close.rolling(200).mean()
    feat["rsi14"] = compute_rsi(close, 14)
    feat["above_sma50"] = (close > feat["sma50"]).astype(float)
    feat["above_sma200"] = (close > feat["sma200"]).astype(float)
    
    # 3. Triple Barrier Method de Marcos López de Prado
    targets = []
    future_rets = []
    
    prices = close.values.flatten()
    vols = vol_24step.values.flatten()
    
    for i in range(len(prices)):
        if i + max_holding >= len(prices) or np.isnan(vols[i]):
            targets.append(np.nan)
            future_rets.append(np.nan)
            continue
            
        target_pt = prices[i] * (1.0 + pt_mult * vols[i])
        target_sl = prices[i] * (1.0 - sl_mult * vols[i])
        
        barrier_hit = 0
        fut_ret = (prices[i + max_holding] - prices[i]) / prices[i]
        
        for h in range(1, max_holding + 1):
            p_future = prices[i + h]
            if p_future >= target_pt:
                barrier_hit = 1
                fut_ret = (target_pt - prices[i]) / prices[i]
                break
            elif p_future <= target_sl:
                barrier_hit = -1
                fut_ret = (target_sl - prices[i]) / prices[i]
                break
                
        targets.append(1 if barrier_hit == 1 else 0)
        future_rets.append(fut_ret)
        
    feat["target_triple_barrier"] = targets
    feat["future_return"] = future_rets
    
    df_clean = feat.dropna().copy()
    
    if apply_prescreen:
        # Pre-screening: RVOL > 0.9, Au-dessus de SMA50 ou SMA200 (Momentum & Liquidité)
        prescreen_mask = (df_clean["rvol"] >= 0.9) & ((df_clean["above_sma50"] == 1) | (df_clean["above_sma200"] == 1))
        if prescreen_mask.sum() > 1000:
            df_clean = df_clean[prescreen_mask].copy()
            
    return df_clean

# 2. Architecture Deep Residual Tabular Transformer (25.8M Paramètres)
class ResidualTransformerBlock(nn.Module):
    def __init__(self, embed_dim=1024, n_heads=8, dropout=0.15):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(h))
        f = self.ffn(x)
        x = self.norm2(x + self.dropout(f))
        return x

class MultiAssetResidualTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=1024, n_blocks=3, n_heads=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList([ResidualTransformerBlock(embed_dim, n_heads) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.input_proj(x).unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        return self.head(h.squeeze(1)).squeeze(-1)

def train_single_asset(ticker: str, max_epochs: int = 150, patience: int = 20, batch_size: int = 256, prescreen: bool = True, eval_only: bool = False, test_ratio: float = 0.20, period: str = "60d", interval: str = "5m"):
    print("\n" + "=" * 90)
    mode_str = "ÉVALUATION SUR TEST HOLDOUT ÉTENDU" if eval_only else f"ENTRAÎNEMENT TRANSFORMER RÉSIDUEL SOTA (25.8M PARAMS - {interval})"
    print(f" 🚀 {mode_str} SUR : {ticker}")
    print("=" * 90)
    
    if interval == "1m":
        period = "7d"  # Limite Yahoo Finance pour 1m
    elif interval == "5m" and period == "730d":
        period = "60d"  # Limite Yahoo Finance pour 5m

    print(f"[1/5] 📈 Téléchargement du Dataset {interval} pour {ticker} (Période: {period})...")
    try:
        raw_data = yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception as e:
        print(f"❌ Erreur téléchargement {ticker}: {e}")
        return None

    if raw_data.empty or len(raw_data) < 1000:
        print(f"⚠️ Données insuffisantes pour {ticker} ({len(raw_data)} bougies). Actif ignoré.")
        return None

    full_df = apply_triple_barrier_and_features(raw_data, apply_prescreen=prescreen, interval=interval)
    n_total = len(full_df)
    
    train_ratio = max(0.50, 1.0 - test_ratio - 0.15)
    val_ratio = 0.15
    
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * (train_ratio + val_ratio))

    train_df = full_df.iloc[:n_train]
    val_df = full_df.iloc[n_train:n_val]
    test_df = full_df.iloc[n_val:]

    feature_cols = [c for c in train_df.columns if c not in ["target_triple_barrier", "future_return", "sma50", "sma200"]]

    mean = train_df[feature_cols].mean()
    std = train_df[feature_cols].std() + 1e-8

    X_train_norm = (train_df[feature_cols] - mean) / std
    X_val_norm = (val_df[feature_cols] - mean) / std
    X_test_norm = (test_df[feature_cols] - mean) / std

    y_train = train_df["target_triple_barrier"].values
    y_val = val_df["target_triple_barrier"].values
    ret_val = val_df["future_return"].values
    y_test = test_df["target_triple_barrier"].values
    ret_test = test_df["future_return"].values

    print(f"\n[2/5] ✂️ Découpage Temporel ({ticker} {interval}, Total N = {n_total:,} échantillons) :")
    print(f"  • Train Set ({train_ratio*100:.0f}%)     : {len(X_train_norm):,} échantillons")
    print(f"  • Validation Set ({val_ratio*100:.0f}%): {len(X_val_norm):,} échantillons")
    print(f"  • Test Holdout ÉTENDU ({len(X_test_norm)/n_total*100:.1f}%): {len(X_test_norm):,} ÉCHANTILLONS TOTALEMENT NON-VUS")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    train_dataset = TensorDataset(torch.tensor(X_train_norm.values, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_tensor = torch.tensor(X_val_norm.values, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_norm.values, dtype=torch.float32).to(device)

    model = MultiAssetResidualTransformer(len(feature_cols)).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n[3/5] 🧠 Transformer Initialisé ({ticker} {interval}) sur [{device.type.upper()}] : {num_params:,} Paramètres")

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    evaluator = CryptoCustomUtilityMetric(target_profit_pct=3.0, stop_loss_pct=1.5)

    os.makedirs("models", exist_ok=True)
    clean_ticker_str = ticker.replace("-", "_").replace("/", "_")
    checkpoint_path = f"models/tabfm_residual_{clean_ticker_str}_{interval}.pt" if interval != "1h" else f"models/tabfm_residual_{clean_ticker_str}.pt"

    best_caum_score = -999.0
    best_epoch_num = 0

    if not eval_only:
        patience_counter = 0

        print(f"\n[4/5] 🚀 DÉMARRAGE ENTRAÎNEMENT AVEC EARLY STOPPING (PATIENCE = {patience} EPOCHS)...")
        print("-" * 105)
        print(f"{'Epoch':<8} | {'Train Loss':<11} | {'Val Acc (%)':<11} | {'Val WinRate':<11} | {'Val CAUM Score':<14} | Statut Early Stopping")
        print("-" * 105)

        for epoch in range(1, max_epochs + 1):
            model.train()
            running_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(bx)

            train_loss = running_loss / len(train_df)

            model.eval()
            with torch.no_grad():
                val_probs = model(X_val_tensor).cpu().numpy()

            val_preds = (val_probs >= 0.54).astype(int)
            val_acc = accuracy_score(y_val, val_preds) * 100.0
            val_metrics = evaluator.compute_asymmetric_utility(val_probs, ret_val)
            val_caum = val_metrics["crypto_utility_score"]
            val_wr = val_metrics["win_rate"]

            status = ""
            if val_caum > best_caum_score:
                best_caum_score = val_caum
                best_epoch_num = epoch
                torch.save(model.state_dict(), checkpoint_path)
                patience_counter = 0
                status = f"🏆 [RECORD BEST CHECKPOINT @ Epoch #{epoch}]"
            else:
                patience_counter += 1
                status = f"⏳ Patience ({patience_counter}/{patience})"

            print(f"Epoch {epoch:<3}/{max_epochs} | Loss: {train_loss:.4f}  | Acc: {val_acc:5.1f}%  | WinRate: {val_wr:5.1f}% | CAUM: {val_caum:7.2f}     | {status}")

            if patience_counter >= patience:
                print("-" * 105)
                print(f"🛑 EARLY STOPPING DÉCLENCHÉ à l'Epoch #{epoch} ! Aucun progrès de Val CAUM pendant {patience} epochs.")
                break

        print("-" * 105)
        print(f"  🏆 MEILLEUR CHECKPOINT POUR {ticker} : Epoch #{best_epoch_num} (Score CAUM: {best_caum_score:.2f})")
    else:
        print(f"\n[4/5] ⏩ MODE ÉVALUATION SEULEMENT : Ignorer la phase d'entraînement, chargement des poids existants...")
        if not os.path.exists(checkpoint_path):
            print(f"❌ Aucun fichier de poids trouvé pour {ticker} dans {checkpoint_path}")
            return None

    # 5. Évaluation Finale sur Test Holdout NON-VU
    print(f"\n[5/5] 🧪 ÉVALUATION DU CHECKPOINT SUR LE JEU DE TEST HOLDOUT ({ticker})...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        test_probs = model(X_test_tensor).cpu().numpy()

    test_preds = (test_probs >= 0.54).astype(int)
    test_acc = accuracy_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    test_prec = precision_score(y_test, test_preds, zero_division=0)
    test_metrics = evaluator.compute_asymmetric_utility(test_probs, ret_test)

    results_summary = {
        "ticker": ticker,
        "test_acc": test_acc * 100.0,
        "test_prec": test_prec * 100.0,
        "test_auc": test_auc,
        "test_caum": test_metrics.get("crypto_utility_score", 0.0),
        "test_winrate": test_metrics.get("win_rate", 0.0),
        "test_sharpe": test_metrics.get("sharpe_ratio_crypto", 0.0),
        "test_profit_factor": test_metrics.get("profit_factor", 0.0)
    }

    print("\n" + "=" * 90)
    print(f" 🏆 RÉSULTATS D'ÉVALUATION POUR {ticker} (HOLDOUT NON-VU)")
    print("=" * 90)
    print(f"  • Actif                        : {ticker}")
    print(f"  • Fichier Poids Chargé         : {checkpoint_path}")
    print(f"  • Précision Globale (Accuracy)   : {test_acc*100:.2f}%")
    print(f"  • Précision ACHAT (Precision)  : {test_prec*100:.2f}%")
    print(f"  • Win Rate Signaux (%)          : {test_metrics.get('win_rate', 0.0):.1f}%")
    print(f"  • Score ROC-AUC                 : {test_auc:.3f}")
    print(f"  • Profit Factor                 : {test_metrics.get('profit_factor', 0.0):.2f}")
    print(f"  • Sharpe Ratio Crypto (24/7)    : {test_metrics.get('sharpe_ratio_crypto', 0.0):.2f}")
    print(f"  • Score Utilité CAUM Holdout    : {test_metrics.get('crypto_utility_score', 0.0):.2f}")
    print("=" * 90 + "\n")

    return results_summary

def main():
    parser = argparse.ArgumentParser(description="Entraînement Transformer Résiduel Multi-Actifs SOTA avec Early Stopping")
    parser.add_argument("--tickers", type=str, default="BTC-USD,ETH-USD,SOL-USD,AVAX-USD", help="Liste des tickers séparés par des virgules")
    parser.add_argument("--max_epochs", type=int, default=100, help="Nombre d'epochs max")
    parser.add_argument("--patience", type=int, default=12, help="Patience Early Stopping")
    parser.add_argument("--batch_size", type=int, default=256, help="Taille de batch")
    parser.add_argument("--no_prescreen", action="store_true", help="Désactiver le pre-screening Volume/Momentum")
    parser.add_argument("--eval_only", action="store_true", help="Évaluer directement le fichier de poids sans ré-entraîner")
    parser.add_argument("--test_ratio", type=float, default=0.25, help="Proportion de l'ensemble de données dédiée au jeu de test non-vu (défaut: 0.25 = 25%)")
    parser.add_argument("--period", type=str, default="60d", help="Période historique (ex: 60d pour 5m, 730d pour 1h)")
    parser.add_argument("--interval", type=str, default="5m", choices=["5m", "15m", "1h", "1d"], help="Fréquence temporelle des bougies (défaut: 5m)")
    args = parser.parse_args()

    ticker_list = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    all_results = []

    print("=" * 100)
    mode_header = "ÉVALUATION SEULEMENT (TEST ÉTENDU)" if args.eval_only else "ENTRAÎNEMENT & ÉVALUATION"
    print(f" 🏛️ DÉMARRAGE ({mode_header}) - TRANSFORMER RÉSIDUEL (25.8M PARAMS - {args.interval})")
    print(f" Actifs cibles : {', '.join(ticker_list)} | Fréquence : {args.interval} | Test Ratio : {args.test_ratio*100:.0f}%")
    print("=" * 100)

    for ticker in ticker_list:
        res = train_single_asset(
            ticker=ticker,
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            prescreen=not args.no_prescreen,
            eval_only=args.eval_only,
            test_ratio=args.test_ratio,
            period=args.period,
            interval=args.interval
        )
        if res is not None:
            all_results.append(res)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        print("\n" + "=" * 100)
        print(" 📊 TABLEAU COMPARATIF DES RÉSULTATS (TEST HOLDOUT NON-VU)")
        print("=" * 100)
        print(summary_df.to_string(index=False))
        print("=" * 100 + "\n")

if __name__ == "__main__":
    main()
