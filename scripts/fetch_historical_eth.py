"""
Script d'Ingestion de Données Historiques Masives pour ETH/USDT 1h via CCXT / Binance.
Permet d'obtenir plusieurs années de bougies 1h (ex: 2 à 5 ans, soit 17 000 à 40 000+ bougies 1h)
au lieu de la limite restreinte de 60 jours de yfinance.
"""
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

def fetch_large_ohlcv(symbol: str = "ETH/USDT", timeframe: str = "1h", days_back: int = 730) -> pd.DataFrame:
    """
    Télécharge l'historique complet par pagination CCXT.
    730 jours ~ 2 ans = 17 520 bougies 1h.
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    
    now_ms = exchange.milliseconds()
    since_ms = now_ms - (days_back * 24 * 60 * 60 * 1000)
    
    all_ohlcv = []
    limit = 1000 # Binance max limit par appel
    
    print(f"[DATA FETCH] Téléchargement de {days_back} jours (~{days_back*24} bougies 1h) pour {symbol}...")
    
    curr_since = since_ms
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
            print(f" -> Téléchargé jusqu me: {datetime.fromtimestamp(last_timestamp/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} (Total: {len(all_ohlcv)} bougies)")
            time.sleep(0.1) # Respect du rate limit Binance
        except Exception as e:
            print(f"Erreur fetch: {e}")
            time.sleep(1.0)
            
    df = pd.DataFrame(all_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
    df.set_index('Timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    return df

if __name__ == "__main__":
    df_eth = fetch_large_ohlcv("ETH/USDT", timeframe="1h", days_back=365) # 1 an = ~8760 bougies
    print(f"\n[SUCCESS] DataFrame final : {len(df_eth)} bougies 1h.")
    print(df_eth.head(2))
    print(df_eth.tail(2))
