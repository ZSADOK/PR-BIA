import os
import json
import logging
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)

class GeminiSocialAnalyzer:
    """
    Analyseur de Sentiment & Tendance Sociale propulsé par Google Gemini LLM.
    """

    def __init__(self, api_key: Optional[str] = None):
        raw_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.api_key = raw_key.strip()
        self.endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 20)

    def analyze_social_posts(self, posts_text: List[str]) -> List[Dict]:
        """
        Analyse les posts sociaux via Gemini LLM pour en extraire :
        - Ticker
        - Nom de l'actif
        - Sentiment (-1.0 baissier à +1.0 haussier)
        - Explication synthétique du consensus
        """
        if not self.is_available():
            logger.warning("Clé API Gemini non configurée.")
            return []

        from datetime import datetime
        now_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = f"""
        DATE ET HEURE EN TEMPS RÉEL (DU JOUR) : {now_date_str}
        Tu es un Analyste Financier Quantitatif Senior chez Goldman Sachs / Citadel.
        Voici les données du jour et les messages récents des flux d'investisseurs (Reddit, X/Twitter, Google Finance, Bloomberg) :
        
        ---
        {json.dumps(posts_text[:80], ensure_ascii=False)}
        ---
        
        Ta mission d'élite :
        1. Détecte les 8 actifs (Actions ou Cryptos) les plus chauds avec un VRAI catalyseur de marché.
        2. Pour chaque actif, attribue un score de sentiment entre -1.0 (panique/vente) et +1.0 (hype institutionnelle/achat).
        3. Identifie le TYPE DE CATALYSEUR exact parmi :
           ["EARNINGS_SURPRISE", "FDA_APPROVAL", "SHORT_SQUEEZE_ALERT", "AI_BREAKOUT", "PARTNERSHIP", "MACRO_NEWS", "NONE"]
        4. Attribue un score d'impact du catalyseur `catalyst_power` de 1 (faible bruit) à 10 (catalyseur majeur historique).
        5. Rédige une thèse synthétique en 1 phrase.

        Réponds UNIQUEMENT sous forme de tableau JSON valide au format suivant :
        [
          {{"ticker": "NVDA", "nom": "Nvidia Corp", "sentiment": 0.95, "catalyst": "AI_BREAKOUT", "catalyst_power": 9, "raison": "Puces Blackwell sur-commandées par Microsoft et Meta."}},
          {{"ticker": "VKTX", "nom": "Viking Therapeutics", "sentiment": 0.85, "catalyst": "FDA_APPROVAL", "catalyst_power": 8, "raison": "Essai clinique de phase 3 concluant pour l'anti-obésité."}}
        ]
        """

        session = requests.Session()
        try:
            from urllib3.util import Retry
            from requests.adapters import HTTPAdapter
            retries = Retry(total=2, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504])
            session.mount("https://", HTTPAdapter(max_retries=retries))
        except Exception:
            pass

        try:
            url = f"{self.endpoint}?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            res = session.post(url, headers=headers, json=payload, timeout=30)
            if res.status_code == 200:
                data = res.json()
                text_out = data["candidates"][0]["content"]["parts"][0]["text"]
                if "```json" in text_out:
                    text_out = text_out.split("```json")[1].split("```")[0]
                elif "```" in text_out:
                    text_out = text_out.split("```")[1].split("```")[0]
                parsed = json.loads(text_out.strip())
                
                # Booster le score de sentiment si un catalyseur majeur (power >= 7) est détecté !
                for item in parsed:
                    c_power = item.get("catalyst_power", 5)
                    if c_power >= 7:
                        item["sentiment"] = min(1.0, item["sentiment"] * 1.2)
                return parsed
            else:
                logger.debug(f"Code API Gemini ({res.status_code}): {res.text[:200]}")
                return []
        except Exception as e:
            logger.info(f"Délai d'attente Gemini LLM dépassé (micro-coupure réseau/serveur) : fallback neutre actif.")
            return []

    def analyze_institutional_reports(self, report_texts: List[str]) -> List[Dict]:
        """
        Analyse les bilans institutionnels SEC 13F & rapports BlackRock / Vanguard via Gemini LLM
        pour extraire les accumulations d'actifs par la 'Smart Money'.
        """
        if not self.is_available():
            return []

        prompt = f"""
        Tu es un analyste de recherche Macro & Institutional Holdings pour un Hedge Fund Quantitatif.
        Voici des extraits récents de déclarations institutionnelles SEC 13F, bilans et rapports d'investissement (BlackRock, Vanguard, Berkshire Hathaway) :
        
        ---
        {json.dumps(report_texts[:15], ensure_ascii=False)}
        ---
        
        Ta tâche :
        1. Identifie les actifs (Actions, ETF, Cryptos, Matières Premières) où les fonds institutionnels augmentent massivement leurs positions (Smart Money Accumulation).
        2. Attribue un score d'accumulation institutionnelle de -1.0 (Liquidation/Vente massive) à +1.0 (Achat massif par BlackRock/Vanguard).
        3. Explique la thèse d'investissement institutionnel en 1 sentence.

        Réponds UNIQUEMENT sous forme de tableau JSON valide au format suivant :
        [
          {{"ticker": "NVDA", "nom": "Nvidia Corp", "smart_money_score": 0.90, "thesis": "Accumulation majeure par BlackRock et Vanguard pour le supercycle IA."}},
          {{"ticker": "GLD", "nom": "SPDR Gold Shares", "smart_money_score": 0.75, "thesis": "Couverture institutionnelle massive contre l'inflation et la dette."}}
        ]
        """
        try:
            url = f"{self.endpoint}?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            res = requests.post(url, headers=headers, json=payload, timeout=12)
            if res.status_code == 200:
                data = res.json()
                text_out = data["candidates"][0]["content"]["parts"][0]["text"]
                if "```json" in text_out:
                    text_out = text_out.split("```json")[1].split("```")[0]
                elif "```" in text_out:
                    text_out = text_out.split("```")[1].split("```")[0]
                return json.loads(text_out.strip())
            return []
        except Exception as e:
            logger.error(f"Échec analyse institutionnelle Gemini: {e}")
            return []
