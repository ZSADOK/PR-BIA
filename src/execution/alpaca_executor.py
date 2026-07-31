"""
Module d'Exécution Alpaca Paper Trading pour Bot de Trading Algorithmique.
Connecte le bot à l'API Alpaca Paper Trading (Cryptos 24/7 & Actions US).
Lit le signal TimesFM + Méta-Labeler (Étape 2/5m) et le Sizing Risk Manager (Étape 3)
pour exécuter automatiquement les ordres d'Achat/Vente sur Alpaca.
"""
import os
import requests
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"

class AlpacaExecutor:
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = ALPACA_PAPER_URL
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Content-Type": "application/json"
        }

    def fetch_account(self) -> Dict[str, Any]:
        """Récupère les informations du compte Alpaca (Cash, Buying Power, Valeur Portefeuille)."""
        url = f"{self.base_url}/v2/account"
        try:
            res = requests.get(url, headers=self.headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                return {
                    "status": "connected",
                    "account_number": data.get("account_number", "PA3T5NINSLGS"),
                    "cash": float(data.get("cash", 100000.0)),
                    "portfolio_value": float(data.get("portfolio_value", 100000.0)),
                    "buying_power": float(data.get("buying_power", 400000.0)),
                    "status_code": data.get("status", "ACTIVE")
                }
            else:
                logger.error(f"Erreur API Alpaca ({res.status_code}): {res.text}")
                return {"status": "error", "message": res.text, "cash": 0.0, "account_number": "N/A"}
        except Exception as e:
            logger.error(f"Impossible de se connecter à l'API Alpaca: {e}")
            return {"status": "error", "message": str(e), "cash": 0.0, "account_number": "N/A"}

    def fetch_positions(self) -> list:
        """Récupère la liste des positions ouvertes sur Alpaca."""
        url = f"{self.base_url}/v2/positions"
        try:
            res = requests.get(url, headers=self.headers, timeout=5)
            if res.status_code == 200:
                return res.json()
            return []
        except Exception as e:
            logger.error(f"Erreur fetch_positions Alpaca: {e}")
            return []

    def execute_order(
        self,
        symbol: str = "ETH/USD",
        side: str = "buy",
        notional: Optional[float] = None,
        qty: Optional[float] = None
    ) -> Dict[str, Any]:
        """Transmet un ordre d'Achat ou Vente au marché sur Alpaca Paper Trading."""
        url = f"{self.base_url}/v2/orders"
        alpaca_symbol = symbol.replace("-", "/")
        
        payload = {
            "symbol": alpaca_symbol,
            "side": side.lower(),
            "type": "market",
            "time_in_force": "gtc" if "/" in alpaca_symbol else "day"
        }
        
        if notional is not None and notional > 0:
            payload["notional"] = str(round(notional, 2))
        elif qty is not None and qty > 0:
            payload["qty"] = str(round(qty, 4))
        else:
            return {"status": "rejected", "reason": "Ni notional ni qty spécifié"}
            
        logger.info(f"[ALPACA ORDER] Transmission ordre {side.upper()} {payload.get('notional', payload.get('qty'))} $ sur {alpaca_symbol}...")
        
        try:
            res = requests.post(url, json=payload, headers=self.headers, timeout=5)
            if res.status_code in [200, 201]:
                order_data = res.json()
                logger.info(f"✔ Ordre Alpaca transmis avec succès ! ID: {order_data.get('id')}")
                return {"status": "executed", "order_id": order_data.get('id'), "raw": order_data}
            else:
                logger.error(f"Erreur transmission ordre Alpaca ({res.status_code}): {res.text}")
                return {"status": "error", "reason": res.text}
        except Exception as e:
            logger.error(f"Exception lors de la transmission d'ordre Alpaca: {e}")
            return {"status": "error", "reason": str(e)}

    def execute_bot_cycle(
        self,
        signal_dict: Dict[str, Any],
        position_size_dict: Dict[str, Any],
        symbol: str = "ETH/USD"
    ) -> Dict[str, Any]:
        signal_binary = signal_dict.get('signal_binary', 0)
        capital_allocated = position_size_dict.get('capital_allocated', 0.0)
        
        if signal_binary == 1 and capital_allocated > 10.0:
            order_res = self.execute_order(
                symbol=symbol,
                side="buy",
                notional=capital_allocated
            )
            return {
                "action": "BUY_EXECUTED_ALPACA",
                "symbol": symbol,
                "notional": capital_allocated,
                "order_details": order_res
            }
        else:
            return {
                "action": "NEUTRAL_HOLD",
                "symbol": symbol,
                "reason": "Signal neutre (0) ou capital alloué insuffisant"
            }
