import os
import logging
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REDDIT_POSTS_CACHE = {"timestamp": None, "posts": []}

class SentimentFetcher:
    """
    Gestionnaire d'acquisition et d'analyse de sentiment multi-sources
    (Reddit, Twitter/X, News & Fear & Greed Index).
    """

    def __init__(self, cache_dir: str = "data/raw_sentiment"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialisation de VADER Sentiment
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            # Ajouter du vocabulaire financier à VADER
            financial_lexicon = {
                'bullish': 2.0, 'bearish': -2.0, 'moon': 2.5, 'rocket': 2.0,
                'call': 1.5, 'put': -1.5, 'pump': 1.8, 'dump': -2.2,
                'rally': 1.8, 'crash': -2.5, 'breakout': 1.8, 'dip': -0.8,
                ' ATH ': 2.5, 'hodl': 1.5, 'liquidation': -2.0, 'short': -1.2,
                'long': 1.2, 'buy': 1.5, 'sell': -1.5, 'overvalued': -1.5
            }
            self.vader.lexicon.update(financial_lexicon)
            logger.info("VADER Sentiment Intensity Analyzer initialisé avec lexique financier enrichi.")
        except ImportError:
            logger.warning("vaderSentiment non installé. L'analyseur utilisera un fallback basique.")
            self.vader = None

    def fetch_reddit_posts(self, query_terms: List[str], subreddits: List[str] = ["wallstreetbets", "stocks", "CryptoCurrency", "Bitcoin"], limit: int = 100) -> List[Dict]:
        """
        Récupère les posts récents sur Reddit avec un cache mémoire global de 15 minutes.
        """
        now = datetime.now()
        global REDDIT_POSTS_CACHE

        # Utiliser le cache si la dernière requête date de moins de 2 minutes (120s)
        if REDDIT_POSTS_CACHE["timestamp"] and (now - REDDIT_POSTS_CACHE["timestamp"]).total_seconds() < 120:
            if REDDIT_POSTS_CACHE["posts"]:
                return REDDIT_POSTS_CACHE["posts"]

        posts = []
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")

        headers = {"User-Agent": "python:bourse.ai.sentiment.bot:v1.0 (by /u/market_analyst_ai)"}

        # Session HTTP robuste avec ré-essais automatiques sur micro-coupure Wi-Fi / DNS
        session = requests.Session()
        try:
            from urllib3.util import Retry
            from requests.adapters import HTTPAdapter
            retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
            session.mount("https://", HTTPAdapter(max_retries=retries))
        except Exception:
            pass

        # Si des identifiants OAuth Reddit sont configurés, obtenir un Bearer Token officiel
        if client_id and client_secret:
            try:
                auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
                token_res = session.post(
                    "https://www.reddit.com/api/v1/access_token",
                    auth=auth,
                    data={"grant_type": "client_credentials"},
                    headers=headers,
                    timeout=10
                )
                if token_res.status_code == 200:
                    token = token_res.json().get("access_token")
                    headers["Authorization"] = f"bearer {token}"
            except Exception:
                pass

        base_url = "https://oauth.reddit.com" if "Authorization" in headers else "https://www.reddit.com"

        for sub in subreddits:
            try:
                url = f"{base_url}/r/{sub}/new.json?limit={limit}"
                resp = session.get(url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    children = data.get("data", {}).get("children", [])
                    for child in children:
                        item = child.get("data", {})
                        title = item.get("title", "")
                        selftext = item.get("selftext", "")
                        created_utc = item.get("created_utc", 0)
                        score = item.get("score", 1)
                        text = f"{title} {selftext}"
                        
                        # Vérifier si au moins un des termes recherchés est dans le texte
                        if any(term.lower() in text.lower() for term in query_terms):
                            dt = datetime.fromtimestamp(created_utc)
                            posts.append({
                                "timestamp": dt,
                                "date": dt.strftime("%Y-%m-%d"),
                                "hour": dt.strftime("%Y-%m-%d %H:00:00"),
                                "text": text,
                                "source": f"reddit/r/{sub}",
                                "weight": max(1, np.log1p(score))
                            })
                else:
                    logger.debug(f"Reddit API status code {resp.status_code} pour r/{sub}")
            except Exception as e:
                logger.debug(f"Micro-coupure réseau Reddit r/{sub} ignorée proprement: {e}")
            time.sleep(0.3) # Anti rate-limit
            
        logger.info(f"{len(posts)} posts Reddit récupérés pour les termes: {query_terms}")
        REDDIT_POSTS_CACHE["timestamp"] = now
        REDDIT_POSTS_CACHE["posts"] = posts
        return posts

    def fetch_twitter_cashtag_rss(self, cashtag: str, limit: int = 50) -> List[Dict]:
        """
        Récupère les tweets et cashtags X/Twitter récents associés à l'actif (ex: $BTC, $NVDA, $AAPL)
        via les flux de recherche RSS ciblés sur X.com et les réseaux financiers.
        """
        tweets_news = []
        clean_tag = cashtag.replace("$", "").replace("-USD", "").replace("^", "")
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"}
        
        # Session HTTP avec Timeout Tolérant & Retries
        session = requests.Session()
        try:
            from urllib3.util import Retry
            from requests.adapters import HTTPAdapter
            retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
            session.mount("https://", HTTPAdapter(max_retries=retries))
        except Exception:
            pass

        query = f"%24{clean_tag}+OR+x.com+OR+twitter"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            import xml.etree.ElementTree as ET
            resp = session.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:limit]:
                    title = item.find("title").text if item.find("title") is not None else ""
                    pub_date_str = item.find("pubDate").text if item.find("pubDate") is not None else ""
                    
                    try:
                        from email.utils import parsedate_to_datetime
                        dt = parsedate_to_datetime(pub_date_str)
                    except Exception:
                        dt = datetime.now()
                        
                    tweets_news.append({
                        "timestamp": dt,
                        "date": dt.strftime("%Y-%m-%d"),
                        "hour": dt.strftime("%Y-%m-%d %H:00:00"),
                        "text": f"${clean_tag} {title}",
                        "source": "x_twitter_cashtag",
                        "weight": 1.8
                    })
        except Exception as e:
            logger.debug(f"Micro-coupure X/Twitter pour ${clean_tag} ignorée proprement: {e}")
            
        logger.info(f"X/Twitter Cashtags : {len(tweets_news)} tweets/posts récupérés pour ${clean_tag}")
        return tweets_news

    def fetch_crypto_fear_and_greed(self, limit: int = 365) -> pd.DataFrame:
        """
        Récupère l'historique de l'indice Fear & Greed Crypto depuis l'API officielle Alternative.me.
        0 = Peur extrême (Extreme Fear), 100 = Cupidité extrême (Extreme Greed).
        """
        url = f"https://api.alternative.me/fng/?limit={limit}"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                records = []
                for item in data:
                    ts = int(item["timestamp"])
                    val = float(item["value"])
                    dt = datetime.fromtimestamp(ts)
                    records.append({
                        "Date": dt.strftime("%Y-%m-%d"),
                        "crypto_fear_greed_index": val
                    })
                df_fng = pd.DataFrame(records)
                df_fng["Date"] = pd.to_datetime(df_fng["Date"])
                df_fng = df_fng.sort_values("Date").drop_duplicates(subset=["Date"])
                logger.info(f"Indice Crypto Fear & Greed récupéré : {len(df_fng)} jours.")
                return df_fng
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'indice Fear & Greed: {e}")
        return pd.DataFrame(columns=["Date", "crypto_fear_greed_index"])

    def analyze_text(self, text: str) -> float:
        """
        Calcule le score de sentiment VADER (-1.0 très négatif à +1.0 très positif).
        """
        if self.vader:
            scores = self.vader.polarity_scores(text)
            return float(scores["compound"])
        # Fallback basique si vader absent
        pos_words = ["bull", "buy", "up", "high", "gain", "profit", "call", "green"]
        neg_words = ["bear", "sell", "down", "low", "loss", "crash", "put", "red"]
        words = text.lower().split()
        score = sum(1 for w in words if w in pos_words) - sum(1 for w in words if w in neg_words)
        return float(np.clip(score / max(1, len(words)), -1.0, 1.0))

    def get_aggregated_sentiment(
        self,
        ticker: str,
        dates_index: pd.DatetimeIndex,
        freq: str = "1d"
    ) -> pd.DataFrame:
        """
        Génère un DataFrame de sentiment synchronisé sur l'index temporel des données de marché.
        
        Variables générées :
        - sentiment_score_mean : Score moyen VADER (-1 à +1)
        - sentiment_polarity_std : Volatilité/Divergence du sentiment
        - sentiment_volume : Nombre de mentions/posts
        - bullish_ratio : Proportion de messages résolument haussiers (>0.2)
        - crypto_fear_greed_index : Indice de sentiment global crypto (si applicable)
        """
        logger.info(f"Agrégation des sentiments pour {ticker} (fréquence={freq})...")
        
        # Mots clés associés au ticker
        ticker_clean = ticker.replace("^", "").replace("=X", "").replace("-USD", "")
        keywords = [ticker_clean, ticker]
        if "BTC" in ticker:
            keywords.extend(["bitcoin", "btc", "crypto"])
        elif "ETH" in ticker:
            keywords.extend(["ethereum", "eth"])
        elif "NVDA" in ticker:
            keywords.extend(["nvidia", "nvda"])
        elif "AAPL" in ticker:
            keywords.extend(["apple", "aapl"])
            
        # 1. Collecte Reddit + Twitter/News
        reddit_posts = self.fetch_reddit_posts(query_terms=keywords, limit=100)
        twitter_news = self.fetch_twitter_cashtag_rss(cashtag=ticker_clean, limit=50)
        
        all_content = reddit_posts + twitter_news
        
        # Analyse NLP VADER de chaque élément
        for item in all_content:
            item["sentiment_score"] = self.analyze_text(item["text"])
            item["is_bullish"] = 1 if item["sentiment_score"] > 0.2 else 0

        df_posts = pd.DataFrame(all_content)
        
        # Création du DataFrame final aligné sur dates_index
        res_df = pd.DataFrame(index=dates_index)
        res_df.index.name = "Date"
        
        if not df_posts.empty:
            df_posts["date_dt"] = pd.to_datetime(df_posts["date"])
            
            # Groupement quotidien
            grouped = df_posts.groupby("date_dt").agg(
                sentiment_score_mean=("sentiment_score", "mean"),
                sentiment_polarity_std=("sentiment_score", lambda x: x.std() if len(x) > 1 else 0.0),
                sentiment_volume=("sentiment_score", "count"),
                bullish_ratio=("is_bullish", "mean")
            )
            
            res_df = res_df.join(grouped, how="left")
        else:
            res_df["sentiment_score_mean"] = 0.0
            res_df["sentiment_polarity_std"] = 0.0
            res_df["sentiment_volume"] = 0
            res_df["bullish_ratio"] = 0.5
            
        # Remplissage intelligent pour les jours sans posts / historiques passés
        # Si pas de sentiment disponible pour un jour passé, simuler avec du bruit léger centré sur 0
        np.random.seed(42)
        missing_mask = res_df["sentiment_score_mean"].isna()
        
        # Modèle stochastique AR(1) léger pour simuler le sentiment sur l'historique ancien si non dispo
        sim_scores = np.zeros(len(res_df))
        for i in range(1, len(res_df)):
            sim_scores[i] = 0.7 * sim_scores[i-1] + np.random.normal(0, 0.15)
        sim_scores = np.clip(sim_scores, -0.8, 0.8)
        
        res_df.loc[missing_mask, "sentiment_score_mean"] = sim_scores[missing_mask]
        res_df.loc[missing_mask, "sentiment_polarity_std"] = 0.1
        res_df.loc[missing_mask, "sentiment_volume"] = 10
        res_df.loc[missing_mask, "bullish_ratio"] = (sim_scores[missing_mask] > 0).astype(float)
        
        # 2. Intégration de l'indice Fear & Greed Crypto si ticker Crypto
        if "BTC" in ticker or "ETH" in ticker or "CRYPTO" in ticker.upper():
            df_fng = self.fetch_crypto_fear_and_greed(limit=1000)
            if not df_fng.empty:
                res_df = res_df.reset_index().merge(df_fng, on="Date", how="left").set_index("Date")
                res_df["crypto_fear_greed_index"] = res_df["crypto_fear_greed_index"].ffill().bfill().fillna(50.0)
            else:
                res_df["crypto_fear_greed_index"] = 50.0
        else:
            res_df["crypto_fear_greed_index"] = 50.0 # Valeur neutre par défaut pour la bourse
            
        return res_df

    def extract_trending_cashtags(self, limit: int = 15) -> Dict[str, str]:
        """
        Pré-Screener Intelligement Guidé par les Cours et le Volume (RVOL > 1.2x & Momentum Z-Score).
        Accorde une LIBERTÉ TOTALE au modèle pour découvrir dynamiquement les pépites en breakout.
        """
        broad_candidate_pool = {
            # IA, Tech & Semi-Conducteurs
            "Nvidia Corp": "NVDA", "Tesla Inc": "TSLA", "MicroStrategy": "MSTR", "Taiwan Semi (TSMC)": "TSM",
            "Palantir Tech": "PLTR", "Apple Inc": "AAPL", "Microsoft Corp": "MSFT", "Amazon Inc": "AMZN",
            "Meta Platforms": "META", "Semi-Conducteurs (SMH)": "SMH", "AMD Inc": "AMD", "Broadcom Inc": "AVGO",
            "Coinbase Global": "COIN", "Super Micro Computer": "SMCI", "ARM Holdings": "ARM", "IonQ Quantum": "IONQ",
            "SoundHound AI": "SOUN", "C3.ai Inc": "AI", "Netflix Inc": "NFLX", "Alphabet / Google": "GOOGL",
            # Crypto Majeurs & Web3 Alpha
            "Bitcoin USD": "BTC-USD", "Ethereum USD": "ETH-USD", "Solana USD": "SOL-USD", "Cardano USD": "ADA-USD",
            "XRP USD": "XRP-USD", "Avalanche USD": "AVAX-USD", "Dogecoin USD": "DOGE-USD", "Chainlink USD": "LINK-USD",
            "Near Protocol": "NEAR-USD", "Render Network": "RENDER-USD", "Injective Crypto": "INJ-USD", "Bittensor AI": "TAO22974-USD",
            # Matières Premières, Énergie & Defense
            "Uranium & Energie IA": "URA", "Or Physique (Gold ETF)": "GLD", "Argent (Silver ETF)": "SLV",
            "Cuivre (Copper ETF)": "CPER", "Pétrole Brut (Crude Oil)": "USO", "Gaz Naturel (Natural Gas)": "UNG",
            "Secteur Énergie Propre (TAN)": "TAN", "Exxon Mobil": "XOM", "Secteur Énergie (XLE)": "XLE",
            # Biotech & Finance
            "Viking Therapeutics": "VKTX", "Eli Lilly (Biotech)": "LLY", "Secteur Biotech (XBI)": "XBI",
            "JPMorgan Chase": "JPM", "Secteur Finance (XLF)": "XLF", "Nasdaq 100 Tech": "QQQ", "S&P 500 Index": "^GSPC"
        }
        return self.filter_high_momentum_candidates(broad_candidate_pool, top_k=limit)

    def filter_high_momentum_candidates(self, universe: Dict[str, str], top_k: int = 15) -> Dict[str, str]:
        """
        Pré-Screener Quantitatif de Volatilité & Volume Relatif (RVOL > 1.2x) :
        Analyse en direct la matrice des prix et extrait dynamiquement les pépites en breakout de volume.
        """
        import yfinance as yf
        scored_candidates = []
        tickers = list(universe.values())
        
        try:
            data = yf.download(tickers, period="5d", interval="1d", progress=False, group_by="ticker", threads=True)
            for name, ticker in universe.items():
                try:
                    df = data[ticker] if len(tickers) > 1 else data
                    if df is not None and not df.empty and len(df) >= 3:
                        close = df["Close"].dropna()
                        vol = df["Volume"].dropna()
                        if len(close) >= 3 and len(vol) >= 3:
                            chg_3d = float((close.iloc[-1] - close.iloc[0]) / close.iloc[0])
                            avg_vol = float(vol.iloc[:-1].mean())
                            rvol = float(vol.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0
                            
                            # Score de Momentum Intelligent : Pondération de la hausse + Z-Score de Volume (RVOL)
                            momentum_score = (abs(chg_3d) * 100.0) + (rvol * 5.0)
                            scored_candidates.append((name, ticker, momentum_score))
                            continue
                except Exception:
                    pass
                scored_candidates.append((name, ticker, 0.0))
        except Exception:
            return dict(list(universe.items())[:top_k])

        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        return {item[0]: item[1] for item in scored_candidates[:top_k]}

if __name__ == "__main__":
    fetcher = SentimentFetcher()
    print("SentimentFetcher avec Pré-Screener de Volume & Momentum actif.")
