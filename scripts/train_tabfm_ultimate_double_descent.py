"""
Pipeline d'Entraînement Ultime Double Descente SOTA (10,000 Epochs & Triple Barrier Method)
Focus 100% Spécialisation Actif Maître Unique : BTC-USD (Bitcoin) Haute Densité Intraday
1. Dataset Spécialiste Unique (BTC-USD Haute Densité)
2. Labeling par Méthode Triple Barrier (López de Prado) & Lissage Savitzky-Golay
3. Deep Residual Tabular Transformer avec Skip Connections (10M Paramètres)
4. Scheduler Cosine Decay Continu sur 10 000 Epochs
"""

import os
import sys
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

print("=" * 100)
print(" 🏛️ PIPELINE DEEP LEARNING ULTIME : SPÉCIALISATION ACTIF MAÎTRE UNIQUE (BTC-USD)")
print(" Double Descente Sur-Paramétrée (10,000 Epochs, Triple Barrier Method & Transformer 10M)")
print("=" * 100)

# 1. Dataset Spécialiste Unique Haute Densité Intraday (BTC-USD)
SINGLE_TICKER = "BTC-USD"
print(f"\n[1/5] 📈 Téléchargement du Dataset Spécialiste Unique Intraday ({SINGLE_TICKER})...")
data = yf.download(SINGLE_TICKER, period="730d", interval="1h", progress=False)

def apply_triple_barrier_and_features(df: pd.DataFrame, pt_mult=1.5, sl_mult=1.0, max_holding=12) -> pd.DataFrame:
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
    feat["ret_1h"] = log_ret
    feat["ret_6h"] = log_ret.rolling(6).sum()
    feat["ret_24h"] = log_ret.rolling(24).sum()
    
    vol_24h = log_ret.rolling(24).std()
    vol_168h = log_ret.rolling(168).std()
    feat["noise_ratio"] = vol_24h / (vol_168h + 1e-8)
    feat["vol_zscore"] = (vol_24h - vol_24h.rolling(168).mean()) / (vol_24h.rolling(168).std() + 1e-8)
    feat["volume_zscore"] = (volume - volume.rolling(24).mean()) / (volume.rolling(24).std() + 1e-8)
    
    # 3. Triple Barrier Method de Marcos López de Prado
    targets = []
    future_rets = []
    
    prices = close.values
    vols = vol_24h.values
    
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
    
    return feat.dropna()

full_df = apply_triple_barrier_and_features(data)

# 4. Découpage Temporel Chronologique (Train 70% / Val 15% / Test Holdout 15%)
n_total = len(full_df)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.85)

train_df = full_df.iloc[:n_train]
val_df = full_df.iloc[n_train:n_val]
test_df = full_df.iloc[n_val:]

feature_cols = [c for c in train_df.columns if c not in ["target_triple_barrier", "future_return"]]

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

print(f"\n[2/5] ✂️ Découpage Spécialiste {SINGLE_TICKER} N = {n_total:,} échantillons :")
print(f"  • Train Set (70%)     : {len(X_train_norm):,} échantillons")
print(f"  • Validation Set (15%): {len(X_val_norm):,} échantillons")
print(f"  • Test Holdout (15%)  : {len(X_test_norm):,} échantillons")

train_dataset = TensorDataset(torch.tensor(X_train_norm.values, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

X_val_tensor = torch.tensor(X_val_norm.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_norm.values, dtype=torch.float32)

# 5. Architecture Deep Residual Tabular Transformer avec Skip Connections (10M Paramètres)
class ResidualTransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(h))
        f = self.ffn(x)
        x = self.norm2(x + self.dropout(f))
        return x

class SingleAsset10MDeepResidualTransformer(nn.Module):
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

model = SingleAsset10MDeepResidualTransformer(len(feature_cols))
num_params = sum(p.numel() for p in model.parameters())
print(f"\n[3/5] 🧠 Transformer Résiduel Spécialiste {SINGLE_TICKER} : {num_params:,} Paramètres")

criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)

EPOCHS = 10000
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
evaluator = CryptoCustomUtilityMetric(target_profit_pct=3.0, stop_loss_pct=1.5)

best_caum_score = -999.0
best_epoch_num = 0
checkpoint_path = f"models/single_asset_{SINGLE_TICKER}_double_descent_10m.pt"
os.makedirs("models", exist_ok=True)

print(f"\n[4/5] 🚀 ENTRAÎNEMENT CONTINU SUR 10,000 EPOCHS (COSINE DECAY SIMPLE)...")
print("-" * 110)
print(f"{'Epoch':<10} | {'Train Loss':<12} | {'Val Acc (%)':<12} | {'Val WinRate':<12} | {'Val CAUM Score':<15} | {'LR':<10} | Régime")
print("-" * 110)

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(bx)

    train_loss = running_loss / len(train_df)
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()

    if epoch % 100 == 0 or epoch in [1, 10, 500, 1000, 3000, 5000, 7500, 10000]:
        model.eval()
        with torch.no_grad():
            val_probs = model(X_val_tensor).numpy()

        val_preds = (val_probs >= 0.54).astype(int)
        val_acc = accuracy_score(y_val, val_preds) * 100.0
        val_metrics = evaluator.compute_asymmetric_utility(val_probs, ret_val)
        val_caum = val_metrics["crypto_utility_score"]
        val_wr = val_metrics["win_rate"]

        regime = "1er Régime" if epoch < 500 else ("Interpolation" if epoch < 2000 else "🔥 2nd Descente")

        status = ""
        if val_caum > best_caum_score:
            best_caum_score = val_caum
            best_epoch_num = epoch
            torch.save(model.state_dict(), checkpoint_path)
            status = f"[RECORD CHECKPOINT @ Epoch #{epoch}]"

        print(f"Epoch {epoch:<5}/{EPOCHS} | Loss: {train_loss:.4f}  | Acc: {val_acc:5.1f}%  | WinRate: {val_wr:5.1f}% | CAUM: {val_caum:7.2f}     | LR: {current_lr:.1e} | {regime} {status}")

print("-" * 110)
print(f"  🏆 LE MEILLEUR MODÈLE EN DOUBLE DESCENTE SUR {SINGLE_TICKER} A ÉTÉ OBTENU À L'EPOCH #{best_epoch_num} (Score CAUM: {best_caum_score:.2f})")

# 5. Évaluation Finale sur Test Holdout NON-VU avec le modèle retenu
print(f"\n[5/5] 🧪 ÉVALUATION DU MEILLEUR MODÈLE SPÉCIALISTE {SINGLE_TICKER} SUR LE JEU DE TEST HOLDOUT...")
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

with torch.no_grad():
    test_probs = model(X_test_tensor).numpy()

test_preds = (test_probs >= 0.54).astype(int)
test_acc = accuracy_score(y_test, test_preds)
test_auc = roc_auc_score(y_test, test_probs)
test_prec = precision_score(y_test, test_preds, zero_division=0)
test_metrics = evaluator.compute_asymmetric_utility(test_probs, ret_test)

print("\n" + "=" * 95)
print(f" 🏆 RÉSULTATS DU MODÈLE ULTIME DEEP DESCENT 10M SPÉCIALISTE {SINGLE_TICKER} (EPOCH #{best_epoch_num} / {EPOCHS})")
print("=" * 95)
print(f"  • Actif Spécialiste            : {SINGLE_TICKER} (Bitcoin)")
print(f"  • Epoch Optimale Retenue       : Epoch #{best_epoch_num} / {EPOCHS}")
print(f"  • Score CAUM de Validation      : {best_caum_score:.2f}")
print(f"  • Précision Globale (Accuracy)   : {test_acc*100:.2f}%")
print(f"  • Précision Signaux ACHAT       : {test_prec*100:.2f}%")
print(f"  • Score ROC-AUC                 : {test_auc:.3f}")
print(f"  • Profit Factor                 : {test_metrics.get('profit_factor', 0.0):.2f}")
print(f"  • Sharpe Ratio Crypto (24/7)    : {test_metrics.get('sharpe_ratio_crypto', 0.0):.2f}")
print(f"  • Win Rate Signaux (%)          : {test_metrics.get('win_rate', 0.0):.1f}%")
print(f"  • Score Utilité CAUM            : {test_metrics.get('crypto_utility_score', 0.0):.2f}")
print("=" * 95 + "\n")
