"""
Module d'Ingestion de Données Historiques Masives pour ETH/USDT (CCXT / Binance).
Permet de charger des milliers de bougies 1h (1 à 3 ans de données historiques)
avec mise en cache automatique dans data/raw/eth_1h_historical.csv.
"""
import os
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw", "eth_1h_historical.csv")

def get_large_eth_data(
    symbol: str = "ETH/USDT",
    timeframe: str = "1h",
    days_back: int = 365,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Récupère des données historiques massives via CCXT (Binance).
    365 jours = 8 760 bougies 1h.
    730 jours = 17 520 bougies 1h.
    Mise en cache automatique dans CSV.
    """
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    
    # 1. Verification du cache CSV si présent et récent
    if not force_refresh and os.path.exists(CACHE_PATH):
        try:
            df = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
            logger.info(f"Chargement depuis le cache local CSV ({len(df)} bougies 1h).")
            return df
        except Exception as e:
            logger.warning(f"Erreur de lecture du cache local: {e}. Nouveau téléchargement...")

    # 2. Téléchargement via CCXT Binance Pagination
    exchange = ccxt.binance({'enableRateLimit': True})
    now_ms = exchange.milliseconds()
    since_ms = now_ms - (days_back * 24 * 60 * 60 * 1000)
    
    all_ohlcv = []
    limit = 1000
    curr_since = since_ms
    
    logger.info(f"Téléchargement de {days_back} jours (~{days_back*24} bougies 1h) pour {symbol}...")
    
    while curr_since < now_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=curr_since, limit=limit)
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            
            if last_timestamp <= curr_since:
                break
                
            curr_since = last_timestamp + 1
            time.sleep(0.05)
        except Exception as e:
            logger.error(f"Erreur téléchargement CCXT: {e}")
            break

    if not all_ohlcv:
        # Fallback yfinance si CCXT rencontre un problème réseau
        import yfinance as yf
        logger.warning("CCXT indisponible. Fallback sur yfinance (60 jours).")
        ticker = yf.Ticker("ETH-USD")
        df_raw = ticker.history(period="60d", interval="1h")
        df = df_raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()
        return df

    df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
    df.set_index('Timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    
    # Sauvegarde dans le cache CSV
    df.to_csv(CACHE_PATH)
    logger.info(f"Historique de {len(df)} bougies 1h sauvegardé dans {CACHE_PATH}")
    return df
