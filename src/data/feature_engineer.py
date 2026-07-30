import os
import logging
import pandas as pd
import numpy as np
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Module d'ingénierie des caractéristiques avancées (Tendance, Régimes de Volatilité, Divergences)
    et normalisation Z-Score adaptée aux modèles In-Context comme TabFM.
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = (ema_fast - ema_slow) / series
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - signal_line
        return macd, signal_line, macd_hist

    @staticmethod
    def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        pct_b = (series - lower_band) / (upper_band - lower_band + 1e-8)
        band_width = (upper_band - lower_band) / sma
        return pct_b, band_width

    @staticmethod
    def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr / close

    def build_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule l'ensemble des indicateurs techniques stationnaires + caractéristiques de régime.
        """
        res = df.copy()
        close = res["Close"]
        high = res["High"]
        low = res["Low"]
        volume = res["Volume"]

        # 1. Rendements passés & Momentum
        res["feat_ret_1d"] = np.log(close / close.shift(1))
        res["feat_ret_3d"] = np.log(close / close.shift(3))
        res["feat_ret_5d"] = np.log(close / close.shift(5))

        # 2. Ratios & Distances Moyennes Mobiles
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        
        res["feat_dist_sma20"] = (close - sma20) / sma20
        res["feat_dist_sma50"] = (close - sma50) / sma50
        res["feat_dist_sma200"] = (close - sma200) / sma200
        
        # 3. Régimes de Tendance (Trend Regime Filter)
        res["feat_regime_bullish_trend"] = ((close > sma50) & (sma20 > sma50)).astype(float)
        res["feat_regime_golden_cross"] = (sma50 > sma200).astype(float)

        # 4. Oscillateurs
        rsi = self.compute_rsi(close, period=14)
        res["feat_rsi_14"] = (rsi - 50.0) / 50.0 # Centré sur 0 (-1 à +1)
        res["feat_rsi_overbought"] = (rsi > 70).astype(float)
        res["feat_rsi_oversold"] = (rsi < 30).astype(float)

        res["feat_macd"], res["feat_macd_signal"], res["feat_macd_hist"] = self.compute_macd(close)

        # 5. Volatilité, Compression & TTM Squeeze (Secret Quant #1)
        res["feat_bollinger_pct_b"], res["feat_bollinger_width"] = self.compute_bollinger_bands(close)
        res["feat_norm_atr"] = self.compute_atr(high, low, close)
        res["feat_rolling_vol_20d"] = res["feat_ret_1d"].rolling(20).std()

        # Secret Quant #1 : TTM Volatility Compression Squeeze (Détecte les explosions imminentes)
        sma20_v = close.rolling(20).mean()
        std20_v = close.rolling(20).std()
        upper_b = sma20_v + (std20_v * 2.0)
        lower_b = sma20_v - (std20_v * 2.0)
        atr_raw = self.compute_atr(high, low, close) * close
        kc_upper = sma20_v + (atr_raw * 1.5)
        kc_lower = sma20_v - (atr_raw * 1.5)
        res["feat_ttm_squeeze"] = ((upper_b < kc_upper) & (lower_b > kc_lower)).astype(float)

        # 6. Microstructure, Volume & Ancre Institutionnelle VWAP (Secret Quant #2)
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std() + 1e-8
        res["feat_vol_zscore"] = (volume - vol_mean) / vol_std

        # Secret Quant #2 : VWAP (Volume-Weighted Average Price - L'ancre des Market Makers)
        typical_price = (high + low + close) / 3.0
        cum_tp_vol = (typical_price * volume).cumsum()
        cum_vol = volume.cumsum() + 1e-8
        vwap = cum_tp_vol / cum_vol
        res["feat_dist_vwap"] = (close - vwap) / (vwap + 1e-8)

        # 7. Géométrie des Bougies Japonaises (Japanese Candlesticks Feature Extractor)
        open_price = res["Open"]
        candle_range = (high - low) + 1e-8
        body = (close - open_price).abs()
        upper_wick = high - np.maximum(open_price, close)
        lower_wick = np.minimum(open_price, close) - low

        res["feat_candle_body_ratio"] = body / candle_range
        res["feat_candle_upper_wick"] = upper_wick / candle_range
        res["feat_candle_lower_wick"] = lower_wick / candle_range

        # Motifs Clés : Marteau (Rejet Baissier / Rebond Haussier) & Étoile Filante (Épuisement)
        res["feat_candle_hammer"] = ((lower_wick >= 1.5 * body) & (upper_wick <= 0.2 * candle_range)).astype(float)
        res["feat_candle_shooting_star"] = ((upper_wick >= 1.5 * body) & (lower_wick <= 0.2 * candle_range)).astype(float)

        # Englobante Haussière (Bullish Engulfing Momentum)
        prev_open = open_price.shift(1)
        prev_close = close.shift(1)
        res["feat_candle_engulfing"] = ((close > prev_open) & (open_price < prev_close) & (close > prev_close)).astype(float)

        logger.info("Features techniques & géométrie des bougies japonaises calculées.")
        return res

    def merge_market_and_sentiment(self, df_market: pd.DataFrame, df_sentiment: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df_m = df_market.copy()
        df_s = df_sentiment.copy()

        df_m.index = pd.to_datetime(df_m.index)
        df_s.index = pd.to_datetime(df_s.index)

        df_merged = df_m.join(df_s, how="left")
        df_merged = df_merged.ffill().bfill()
        
        feature_cols = [c for c in df_merged.columns if c.startswith("feat_") or c.startswith("sentiment_") or c in ["bullish_ratio", "crypto_fear_greed_index"]]
        
        # Standardisation Z-Score robuste pour TabFM / TabPFN (centrage et réduction)
        scaler_cols = [c for c in feature_cols if not c.startswith("feat_regime_") and not c.startswith("feat_rsi_over")]
        for col in scaler_cols:
            mean = df_merged[col].mean()
            std = df_merged[col].std() + 1e-8
            df_merged[col] = (df_merged[col] - mean) / std

        logger.info(f"Dataset fusionné & normalisé Z-score : {len(df_merged)} lignes, {len(feature_cols)} features.")
        return df_merged, feature_cols

if __name__ == "__main__":
    print("FeatureEngineer initialisé.")
