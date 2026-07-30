import os
import time
import json
import logging
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from rich.console import Console

load_dotenv(override=True)

# Configuration de la console et désactivation des logs bruyants
console = Console()
logging.disable(logging.INFO)
logging.basicConfig(level=logging.ERROR)

for mod in ["src.sentiment_fetcher", "src.data_fetcher", "src.feature_engineer", "src.model_trainer", "src.backtester", "run_ai", "urllib3", "urllib3.connectionpool", "requests"]:
    logging.getLogger(mod).setLevel(logging.ERROR)

API_KEY = os.getenv("ALPACA_API_KEY", "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

if not API_KEY or not SECRET_KEY:
    console.print("[bold red]Erreur : Clés d'API Alpaca non trouvées dans le fichier .env ![/bold red]")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

TICKER_MAP = {
    "^GSPC": "SPY",
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
    "SOL-USD": "SOL/USD",
    "AVAX-USD": "AVAX/USD",
    "DOGE-USD": "DOGE/USD",
    "LINK-USD": "LINK/USD"
}

SECTOR_MAP = {
    "NVDA": "SEMI", "SMH": "SEMI", "TSM": "SEMI", "AMD": "SEMI", "AVGO": "SEMI", "ARM": "SEMI", "SOUN": "SEMI",
    "AAPL": "TECH", "MSFT": "TECH", "GOOGL": "TECH", "AMZN": "TECH", "META": "TECH", "NFLX": "TECH", "PLTR": "TECH", "AI": "TECH", "SMCI": "TECH",
    "BTCUSD": "CRYPTO", "ETHUSD": "CRYPTO", "SOLUSD": "CRYPTO", "ADAUSD": "CRYPTO", "XRPUSD": "CRYPTO", "DOGEUSD": "CRYPTO", "AVAXUSD": "CRYPTO", "COIN": "CRYPTO",
    "BTC-USD": "CRYPTO", "ETH-USD": "CRYPTO", "SOL-USD": "CRYPTO", "ADA-USD": "CRYPTO", "XRP-USD": "CRYPTO", "DOGE-USD": "CRYPTO", "AVAX-USD": "CRYPTO",
    "XOM": "ENERGY", "USO": "ENERGY", "XLE": "ENERGY", "UNG": "ENERGY", "TAN": "ENERGY", "URA": "ENERGY",
    "LLY": "BIOTECH", "VKTX": "BIOTECH", "XBI": "BIOTECH",
    "GLD": "METALS", "SLV": "METALS", "CPER": "METALS", "JPM": "FINANCE", "XLF": "FINANCE"
}

POSITION_PEAK_PL = {}
AI_PORTFOLIO_CACHE = {}
PENDING_EXITS = set()
STOP_LOSS_COOLDOWN = {}
UNSUPPORTED_ALPACA_ASSETS = set()

CRYPTO_TICKERS = {
    "BTCUSD", "ETHUSD", "SOLUSD", "ADAUSD", "XRPUSD", "DOGEUSD", "AVAXUSD",
    "NEARUSD", "RENDERUSD", "TAO22974USD", "INJUSD", "LINKUSD", "PEPEUSD",
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD", "DOGE-USD", "AVAX-USD",
    "NEAR-USD", "RENDER-USD", "TAO22974-USD", "INJ-USD", "LINK-USD", "PEPE-USD",
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "XRP/USD", "DOGE/USD", "AVAX/USD",
    "NEAR/USD", "RENDER/USD", "TAO22974/USD", "INJ/USD", "LINK/USD", "PEPE/USD"
}

def is_crypto_asset(symbol: str) -> bool:
    clean = symbol.replace("-", "").replace("/", "").upper()
    return clean in CRYPTO_TICKERS

def map_symbol_to_alpaca(ticker: str) -> str:
    if ticker in TICKER_MAP:
        return TICKER_MAP[ticker]
    return ticker.replace("-USD", "/USD").replace("^", "")

MEMORY_FILE_PATH = "data/bot_persistent_state.json"

def save_persistent_state():
    """
    Sauvegarde l'état mémoire sur disque JSON.
    """
    import json
    os.makedirs("data", exist_ok=True)
    state = {
        "position_peak_pl": POSITION_PEAK_PL,
        "stop_loss_cooldown": STOP_LOSS_COOLDOWN,
        "unsupported_assets": list(UNSUPPORTED_ALPACA_ASSETS),
        "last_updated": time.time()
    }
    try:
        with open(MEMORY_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

def load_persistent_state():
    """
    Charge l'état mémoire au démarrage du script.
    """
    import json
    if os.path.exists(MEMORY_FILE_PATH):
        try:
            with open(MEMORY_FILE_PATH, "r", encoding="utf-8") as f:
                state = json.load(f)
                POSITION_PEAK_PL.update(state.get("position_peak_pl", {}))
                now = time.time()
                for k, v in state.get("stop_loss_cooldown", {}).items():
                    if v > now:
                        STOP_LOSS_COOLDOWN[k] = v
                for asset in state.get("unsupported_assets", []):
                    UNSUPPORTED_ALPACA_ASSETS.add(asset)
        except Exception:
            pass

load_persistent_state()
save_persistent_state()
