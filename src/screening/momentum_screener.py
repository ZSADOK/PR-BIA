"""
Module de Pré-Screening par Volume Relatif (RVOL) et Momentum Volatilité.
Conforme à la règle AGENTS.md :
- Volume Relatif RVOL > 1.2x (volume actuel / SMA volume 20 périodes)
- Tendance : Prix > SMA 50 et Prix > SMA 200
- Momentum RSI(14) entre 50 et 72
"""
import pandas as pd
import numpy as np


class MomentumScreener:
    def __init__(self, rvol_threshold: float = 1.2, rsi_min: float = 50.0, rsi_max: float = 72.0):
        self.rvol_threshold = rvol_threshold
        self.rsi_min = rsi_min
        self.rsi_max = rsi_max

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs requis pour le filtrage pré-ML.
        df doit contenir les colonnes ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        df = df.copy()
        
        # SMA 50 et SMA 200 (sur bougies 1h)
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Volume Relatif (RVOL) = Volume 1h / Moyenne mobile du volume (20 bougies 1h = ~20h)
        df['Volume_MA20'] = df['Volume'].rolling(window=20).mean()
        df['RVOL'] = df['Volume'] / (df['Volume_MA20'] + 1e-10)
        
        # RSI 14
        df['RSI'] = self.calculate_rsi(df['Close'], period=14)
        
        # Conditions d'éligibilité
        df['Trend_OK'] = (df['Close'] > df['SMA_50']) & (df['Close'] > df['SMA_200'])
        df['RVOL_OK'] = df['RVOL'] > self.rvol_threshold
        df['RSI_OK'] = (df['RSI'] >= self.rsi_min) & (df['RSI'] <= self.rsi_max)
        
        # Statut global d'éligibilité pré-ML
        df['Screening_Passed'] = df['Trend_OK'] & df['RVOL_OK'] & df['RSI_OK']
        
        return df

    def evaluate_latest(self, df: pd.DataFrame) -> dict:
        """Évalue la dernière bougie 1h disponible."""
        processed = self.compute_indicators(df)
        last_row = processed.iloc[-1]
        return {
            "passed": bool(last_row['Screening_Passed']),
            "rvol": float(last_row['RVOL']),
            "trend_ok": bool(last_row['Trend_OK']),
            "rsi": float(last_row['RSI']),
            "rsi_ok": bool(last_row['RSI_OK']),
            "close": float(last_row['Close']),
            "sma_50": float(last_row['SMA_50']) if pd.notna(last_row['SMA_50']) else None,
            "sma_200": float(last_row['SMA_200']) if pd.notna(last_row['SMA_200']) else None
        }
