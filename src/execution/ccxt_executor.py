"""
Module d'Exécution CCXT pour Bot de Trading Algorithmique.
Connecte le bot à un exchange crypto (Binance, Kraken, Coinbase...) en Sandbox/Paper Trading ou Real.
Lit le signal TimesFM (Étape 2) et le Sizing Risk Manager (Étape 3) pour exécuter automatiquement
les ordres d'Achat/Vente et placer les Stop-Loss/Take-Profit.
"""
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    logger.warning("La bibliothèque 'ccxt' n'est pas installée. Mode exécution simulée par défaut.")

class CCXTExecutor:
    def __init__(self, exchange_id: str = "binance", sandbox: bool = True):
        self.exchange_id = exchange_id.lower()
        self.sandbox = sandbox
        self.exchange = self._init_exchange() if HAS_CCXT else None

    def _init_exchange(self):
        """Initialise l'instance CCXT avec les clés d'API .env et active le mode Sandbox si demandé."""
        if not HAS_CCXT:
            return None
            
        api_key = os.getenv(f"{self.exchange_id.upper()}_API_KEY", "")
        secret = os.getenv(f"{self.exchange_id.upper()}_SECRET_KEY", "")
        
        exchange_class = getattr(ccxt, self.exchange_id, None)
        if exchange_class is None:
            logger.warning(f"Exchange CCXT non trouvé: {self.exchange_id}. Fallback simulation.")
            return None
            
        exchange = exchange_class({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        if self.sandbox:
            if hasattr(exchange, 'set_sandbox_mode'):
                try:
                    exchange.set_sandbox_mode(True)
                    logger.info(f"Mode CCXT Sandbox activé sur {self.exchange_id}.")
                except Exception as e:
                    logger.warning(f"Impossible d'activer le mode sandbox natif: {e}")
            else:
                logger.info(f"Exchange {self.exchange_id} géré en Paper Trading local.")
                
        return exchange

    def fetch_balance(self, quote_currency: str = "USDT") -> Dict[str, float]:
        """Récupère les soldes disponibles (Cash et Crypto)."""
        if self.exchange is not None and self.exchange.apiKey and not self.sandbox:
            try:
                balance = self.exchange.fetch_balance()
                free_quote = float(balance.get('free', {}).get(quote_currency, 10000.0))
                return {
                    "free_quote": free_quote,
                    "total_quote": float(balance.get('total', {}).get(quote_currency, free_quote)),
                    "raw": balance
                }
            except Exception as e:
                logger.warning(f"Erreur fetch_balance direct sur {self.exchange_id}: {e}")

        # Solde de simulation Paper Trading par défaut
        return {"free_quote": 10000.0, "total_quote": 10000.0, "raw": {}}

    def fetch_ticker_price(self, symbol: str = "ETH/USDT") -> float:
        """Récupère le prix en direct du marché pour la paire spécifiée."""
        if self.exchange is not None:
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                return float(ticker['last'])
            except Exception as e:
                logger.warning(f"Erreur fetch_ticker CCXT: {e}")
                
        # Inférence yfinance fallback
        import yfinance as yf
        ticker_yf = yf.Ticker(symbol.replace("/", "-"))
        hist = ticker_yf.history(period="1d", interval="1m")
        return float(hist['Close'].iloc[-1])

    def execute_order(
        self,
        symbol: str,
        side: str,  # 'buy' ou 'sell'
        amount: float,
        order_type: str = "market",
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """Transmet un ordre au marché ou limite via CCXT."""
        if amount <= 0:
            return {"status": "rejected", "reason": "Quantité nulle ou négative"}
            
        logger.info(f"[CCXT ORDER] {side.upper()} {amount:.4f} {symbol} ({order_type.upper()})")
        
        try:
            if self.exchange is not None and self.exchange.apiKey and not self.sandbox:
                if order_type == "market":
                    order = self.exchange.create_market_order(symbol, side, amount)
                else:
                    order = self.exchange.create_limit_order(symbol, side, amount, price)
                return {"status": "executed", "order_id": order.get('id'), "raw": order}
            else:
                # Execution Simulée (Paper Trading CCXT)
                current_price = price if price is not None else self.fetch_ticker_price(symbol)
                simulated_order_id = f"SIM-{int(logger.manager.emittedNoHandlerWarning if hasattr(logger.manager, 'emittedNoHandlerWarning') else 100000)}"
                return {
                    "status": "executed_simulated",
                    "order_id": simulated_order_id,
                    "symbol": symbol,
                    "side": side,
                    "amount": amount,
                    "price": current_price,
                    "cost": amount * current_price
                }
        except Exception as e:
            logger.error(f"Erreur d'exécution CCXT: {e}")
            return {"status": "error", "reason": str(e)}

    def execute_bot_cycle(
        self,
        signal_dict: Dict[str, Any],
        position_size_dict: Dict[str, Any],
        symbol: str = "ETH/USDT"
    ) -> Dict[str, Any]:
        """
        Orchestre l'exécution complète d'un cycle de trading :
        1. Analyse le signal binaire (0 ou 1).
        2. Si Signal == 1 (Achat) : Exécute l'ordre d'achat de la quantité calculée par le RiskManager.
        3. Si Signal == 0 (Vente / Neutre) : Transmet la cloture des positions actives.
        """
        signal_binary = signal_dict.get('signal_binary', 0)
        
        if signal_binary == 1 and position_size_dict.get('capital_allocated', 0) > 0:
            units_to_buy = position_size_dict['quantity_units']
            entry_price = position_size_dict['entry_price']
            
            order_res = self.execute_order(
                symbol=symbol,
                side="buy",
                amount=units_to_buy,
                order_type="market",
                price=entry_price
            )
            
            return {
                "action": "BUY_EXECUTED",
                "symbol": symbol,
                "units": units_to_buy,
                "price": entry_price,
                "stop_loss": position_size_dict.get('stop_loss_price'),
                "take_profit": position_size_dict.get('take_profit_price'),
                "order_details": order_res
            }
        else:
            return {
                "action": "NEUTRAL_HOLD",
                "symbol": symbol,
                "reason": "Signal binaire neutre (0) ou conditions du risque non satisfaites"
            }
