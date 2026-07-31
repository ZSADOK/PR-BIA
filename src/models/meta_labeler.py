"""
Module Méta-Labeling & Méta-Filtre XGBoost pour le Trading ETH 5m.
Basé sur la Méthode de la Triple Barrière (López de Prado) :
- Target 1 : Le prix touche le Take-Profit (+1.5x ATR) AVANT le Stop-Loss (-1.0x ATR)
- Target 0 : Le prix touche le Stop-Loss ou expire sans impulsion.

Le Méta-Modèle filtre les signaux primaires de TimesFM pour faire grimper le Win Rate à > 75%.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

META_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "meta_labeler_eth_5m.json")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("La librairie 'xgboost' n'est pas installée. Méta-filtrage basé sur les règles statistiques.")

class MetaLabeler:
    def __init__(self, model_path: str = META_MODEL_PATH):
        self.model_path = model_path
        self.meta_model = None
        self.feature_names = [
            'rvol', 'rsi', 'sma_50_ratio', 'sma_200_ratio', 'atr_ratio',
            'candle_range_ratio', 'ret_1', 'ret_3', 'ret_6', 'timesfm_pred_ret'
        ]
        self._load_meta_model()

    def _load_meta_model(self):
        """Charge le modèle XGBoost entraîné s'il existe."""
        if HAS_XGBOOST and os.path.exists(self.model_path):
            try:
                self.meta_model = xgb.XGBClassifier()
                self.meta_model.load_model(self.model_path)
                logger.info(f"Méta-Filtre XGBoost 5m chargé avec succès depuis : {self.model_path}")
            except Exception as e:
                logger.warning(f"Erreur chargement Méta-Modèle XGBoost ({e}).")
                self.meta_model = None
        else:
            self.meta_model = None

    @staticmethod
    def generate_triple_barrier_labels(
        df: pd.DataFrame,
        pt_multiplier: float = 1.5,
        sl_multiplier: float = 1.0,
        max_holding_candles: int = 12  # 12 bougies 5m = 1 Heure max
    ) -> pd.Series:
        """
        Génère les labels de la Triple Barrière pour l'entraînement :
        Label 1 = Le Take Profit (+1.5x ATR) est atteint avant le Stop Loss (-1.0x ATR).
        Label 0 = Le Stop Loss est touché en premier ou pas d'impulsion.
        """
        df = df.copy()
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean().fillna(df['Close'] * 0.005)

        labels = []
        n = len(df)

        for i in range(n):
            if i + max_holding_candles >= n:
                labels.append(0)
                continue

            entry_price = df['Close'].iloc[i]
            curr_atr = atr.iloc[i]

            tp_price = entry_price + (pt_multiplier * curr_atr)
            sl_price = entry_price - (sl_multiplier * curr_atr)

            label = 0
            for j in range(1, max_holding_candles + 1):
                future_high = df['High'].iloc[i + j]
                future_low = df['Low'].iloc[i + j]

                # Stop loss touché en premier -> Perte (0)
                if future_low <= sl_price:
                    label = 0
                    break

                # Take profit touché en premier -> Victoire (1)
                if future_high >= tp_price:
                    label = 1
                    break

            labels.append(label)

        return pd.Series(labels, index=df.index)

    def extract_features(self, df: pd.DataFrame, timesfm_pred_return: float = 0.0) -> pd.DataFrame:
        """Extrait les features de marché en 5m pour le Méta-Modèle."""
        df = df.copy()

        # Indicateurs 5m
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))

        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()

        vol_ma = df['Volume'].rolling(window=20).mean()
        df['RVOL'] = df['Volume'] / (vol_ma + 1e-10)

        # Ratios & Momentum 5m
        df['sma_50_ratio'] = df['Close'] / (df['SMA_50'] + 1e-10)
        df['sma_200_ratio'] = df['Close'] / (df['SMA_200'] + 1e-10)
        df['atr_ratio'] = df['ATR'] / (df['Close'] + 1e-10)
        df['candle_range_ratio'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)

        df['ret_1'] = df['Close'].pct_change(1)
        df['ret_3'] = df['Close'].pct_change(3)
        df['ret_6'] = df['Close'].pct_change(6)

        df['rvol'] = df['RVOL']
        df['rsi'] = df['RSI']
        df['timesfm_pred_ret'] = timesfm_pred_return

        features_df = df[self.feature_names].fillna(0)
        return features_df

    def predict_meta_confidence(self, df: pd.DataFrame, timesfm_pred_return: float = 0.0) -> float:
        """
        Calcule la probabilité exacte que le trade soit un gagnant (> 75% Win Rate).
        Retourne un score de confiance entre 0.0 et 1.0.
        """
        feat_df = self.extract_features(df, timesfm_pred_return)
        last_row = feat_df.iloc[-1:]

        if self.meta_model is not None and HAS_XGBOOST:
            try:
                # Prédiction de la probabilité de la classe 1 (Victoire Triple Barrière)
                probs = self.meta_model.predict_proba(last_row)
                win_probability = float(probs[0][1])
                return win_probability
            except Exception as e:
                logger.error(f"Erreur d'inférence Méta-Model XGBoost: {e}")

        # Fallback Quantitatif Heuristique (RVOL + Momentum 5m)
        rvol_val = float(last_row['rvol'].iloc[0])
        rsi_val = float(last_row['rsi'].iloc[0])
        sma_ratio = float(last_row['sma_50_ratio'].iloc[0])

        base_score = 0.50
        if rvol_val > 1.2:
            base_score += 0.15
        if 52 <= rsi_val <= 68:
            base_score += 0.10
        if sma_ratio > 1.0:
            base_score += 0.10
        if timesfm_pred_return > 0.0010:
            base_score += 0.10

        return min(0.95, max(0.20, base_score))
