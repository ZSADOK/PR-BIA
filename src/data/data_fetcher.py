import os
import logging
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FinancialDataFetcher:
    """
    Gestionnaire d'acquisition et de mise en cache au format Parquet
    pour les données de marchés financiers (Yahoo Finance API).
    """

    def __init__(self, cache_dir: str = "data/raw"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, ticker: str, interval: str) -> str:
        clean_ticker = ticker.replace("^", "INDEX_").replace("=", "_").replace("-", "_")
        return os.path.join(self.cache_dir, f"{clean_ticker}_{interval}.parquet")

    def fetch_batch_tickers(
        self,
        tickers: List[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Téléchargement groupé ultra-rapide (1 seule requête HTTP pour 50 actifs).
        """
        clean_tickers = [t for t in tickers if t]
        if not clean_tickers:
            return {}

        logger.info(f"Téléchargement groupé en 1 requête HTTP de {len(clean_tickers)} actifs via yfinance...")
        try:
            batch_data = yf.download(
                tickers=clean_tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False
            )
            result = {}
            for t in clean_tickers:
                try:
                    if len(clean_tickers) == 1:
                        df_t = batch_data.copy()
                    else:
                        df_t = batch_data[t].copy() if t in batch_data else pd.DataFrame()
                    
                    if not df_t.empty:
                        df_t.index = pd.to_datetime(df_t.index)
                        if df_t.index.tz is not None:
                            df_t.index = df_t.index.tz_localize(None)
                        df_t.index.name = "Date"
                        expected_cols = ["Open", "High", "Low", "Close", "Volume"]
                        avail = [c for c in expected_cols if c in df_t.columns]
                        df_t = df_t[avail].dropna(how="all")
                        if not df_t.empty:
                            cache_path = self._get_cache_path(t, interval)
                            df_t.to_parquet(cache_path)
                            result[t] = df_t
                except Exception:
                    pass
            return result
        except Exception as e:
            logger.warning(f"Erreur téléchargement groupé yfinance: {e}")
            return {}

    def fetch_ticker(
        self,
        ticker: str,
        start_date: Optional[str] = "2015-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d",
        force_redownload: bool = False,
    ) -> pd.DataFrame:
        """
        Télécharge ou charge depuis le cache Parquet les données historiques d'un actif.
        """
        cache_path = self._get_cache_path(ticker, interval)

        if not force_redownload and os.path.exists(cache_path):
            logger.info(f"Chargement depuis le cache local Parquet : {cache_path}")
            df = pd.read_parquet(cache_path)
            return df

        logger.info(f"Téléchargement en ligne de {ticker} via yfinance ({interval})...")
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)

        if df.empty:
            logger.warning(f"Aucune donnée récupérée pour {ticker}")
            return pd.DataFrame()

        # Nettoyage et standardisation
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)  # Suppression du timezone pour compatibilité

        df.index.name = "Date"
        
        # Conserver les colonnes OHLCV standard
        expected_cols = ["Open", "High", "Low", "Close", "Volume"]
        available_cols = [c for c in expected_cols if c in df.columns]
        df = df[available_cols].copy()

        # Remplissage des trous (jours fériés / micro-interruptions)
        df = df.ffill().bfill()

        # Sauvegarde en Parquet
        df.to_parquet(cache_path, compression="snappy")
        logger.info(f"Sauvegardé dans le cache Parquet : {cache_path} ({len(df)} lignes)")
        return df

    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str = "2015-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d",
        force_redownload: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Télécharge un portefeuille multi-actifs (Indices, Actions, Forex, Crypto).
        """
        data_dict = {}
        for ticker in tickers:
            df = self.fetch_ticker(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                force_redownload=force_redownload,
            )
            if not df.empty:
                data_dict[ticker] = df
        return data_dict
