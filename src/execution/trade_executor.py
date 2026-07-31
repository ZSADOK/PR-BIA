import time
from datetime import datetime
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from src.trading_config import (
    console, trading_client, map_symbol_to_alpaca, is_crypto_asset,
    POSITION_PEAK_PL, AI_PORTFOLIO_CACHE, PENDING_EXITS, STOP_LOSS_COOLDOWN, UNSUPPORTED_ALPACA_ASSETS, save_persistent_state
)
from src.execution.risk_manager import compute_dynamic_kelly_notional, calculate_expected_value
from src.execution.trade_logger import TradeLogger

trade_logger = TradeLogger()

def is_us_stock_market_open() -> bool:
    """
    Vérifie si les marchés boursiers US (NYSE/NASDAQ) sont actuellement ouverts (15h30 - 22h00 heure FR).
    """
    now = datetime.now()
    if now.weekday() >= 5: # Samedi ou Dimanche
        return False
    
    current_minutes = now.hour * 60 + now.minute
    market_open = 15 * 60 + 30  # 15h30
    market_close = 22 * 60      # 22h00
    
    return market_open <= current_minutes < market_close

def execute_sell_all():
    """
    Vire immédiatement toutes les positions du portefeuille Alpaca Paper Trading et remet le solde à 100% Cash.
    """
    console.print("\n[bold red]🔴 LIQUIDATION TOTALE DU PORTEFEUILLE SOLICITÉE...[/bold red]")
    try:
        trading_client.close_all_positions(cancel_orders=True)
        console.print("[bold green]✔ Toutes les positions ont été fermées avec succès ! Portefeuille réinitialisé à 100% Cash.[/bold green]\n")
    except Exception as e:
        console.print(f"[red]Erreur lors de la fermeture des positions: {e}[/red]\n")

def execute_trade_signals(results, threshold: float = 0.58, notional: float = 1000.0, max_budget: float = 75000.0, max_trade_cap: float = 5000.0):
    """
    Exécution automatique des signaux d'achat filtrés par le Risk Manager.
    """
    top_candidates = [
        r for r in results 
        if float(r.get("confidence", r.get("Confiance", 0.0))) >= threshold and not r.get("HasPos", False)
    ]
    if not top_candidates:
        return

    for target_asset in top_candidates:
        target_name = target_asset.get("Nom", target_asset.get("ticker", "BTC-USD"))
        target_symbol = target_asset.get("Ticker", target_asset.get("ticker", "BTC-USD"))
        target_prob = float(target_asset.get("confidence", target_asset.get("Confiance", 0.50)))
        target_alpaca = map_symbol_to_alpaca(target_symbol)

        try:
            live_positions = trading_client.get_all_positions()
            live_clean_symbols = [p.symbol.replace("/", "").upper() for p in live_positions]
        except Exception:
            live_clean_symbols = []

        target_clean = target_alpaca.replace("/", "").upper()

        # Contrôle des Heures de Session Stock US (Les cryptos tournent 24h/24 7j/7)
        if not is_crypto_asset(target_alpaca) and not is_us_stock_market_open():
            console.print(f"[dim white]• {target_name} ({target_alpaca}) : Bourse US fermée (Session 15h30-22h00 FR). Ordre suspendu pour éliminer le slippage.[/dim white]")
            continue

        if target_clean in UNSUPPORTED_ALPACA_ASSETS or target_alpaca in UNSUPPORTED_ALPACA_ASSETS:
            console.print(f"[dim white]• {target_name} ({target_alpaca}) non négociable sur Alpaca Paper Trading. Ignoré.[/dim white]")
            continue

        if time.time() < STOP_LOSS_COOLDOWN.get(target_clean, 0.0):
            rem_min = int((STOP_LOSS_COOLDOWN[target_clean] - time.time()) / 60.0) + 1
            console.print(f"[bold red]• {target_name} ({target_alpaca}) en quarantaine post-Stop Loss ({rem_min} min restantes). Rachat bloqué.[/bold red]")
            continue

        if target_clean in live_clean_symbols:
            console.print(f"[dim white]• {target_name} ({target_alpaca}) déjà en portefeuille. Ignoré.[/dim white]")
            continue

        # Espérance Mathématique Nette (EV)
        expected_value = calculate_expected_value(target_prob)
        if expected_value < 0.35:
            console.print(f"[dim white]• {target_name} ({target_alpaca}) : Espérance Mathématique Insuffisante (EV = {expected_value:+.2f}% < +0.35%). Signal rejeté.[/dim white]")
            continue

        try:
            account = trading_client.get_account()
            available_cash = max(0.0, float(account.cash))
            deployed_capital = sum([float(p.market_value) for p in live_positions])
            remaining_budget = max(0.0, max_budget - deployed_capital)

            dynamic_amount = compute_dynamic_kelly_notional(target_prob, notional)
            dynamic_amount = min(dynamic_amount, max_trade_cap)

            if remaining_budget < 1.00 or available_cash < 1.00:
                console.print(f"[bold yellow][CAPITAL ENTIÈREMENT DÉPLOYÉ] ${deployed_capital:,.2f} investis. Capital préservé en Cash.[/bold yellow]")
                break

            actual_order_notional = min(dynamic_amount, remaining_budget, available_cash)
            if actual_order_notional < 10.0:
                console.print(f"[dim white]Fonds insuffisants pour placer l'ordre sur {target_alpaca}.[/dim white]")
                continue

            console.print(f"\n[bold green][ORDER BUY] Confiance IA {target_prob*100:.1f}% (EV: {expected_value:+.2f}%). Transmission ordre de ${actual_order_notional:,.2f} sur {target_alpaca}...[/bold green]")
            tif = TimeInForce.GTC if "/" in target_alpaca else TimeInForce.DAY
            order_data = MarketOrderRequest(
                symbol=target_alpaca,
                notional=round(actual_order_notional, 2),
                side=OrderSide.BUY,
                time_in_force=tif
            )
            order = trading_client.submit_order(order_data=order_data)
            console.print(f"[green][OK] Ordre transmis à Alpaca avec succès (ID: {order.id})[/green]")
            time.sleep(1.0)
        except Exception as e:
            err_str = str(e)
            if "is not active" in err_str or "40010001" in err_str:
                UNSUPPORTED_ALPACA_ASSETS.add(target_clean)
                UNSUPPORTED_ALPACA_ASSETS.add(target_alpaca)
                save_persistent_state()
                console.print(f"[bold yellow][WARN] {target_alpaca} n'est pas actif sur Alpaca Paper Trading. Ignoré.[/bold yellow]")
            else:
                console.print(f"[red]Erreur lors de l'exécution d'achat sur {target_alpaca}: {e}[/red]")

def execute_sell_signal(ticker: str = "BTC-USD"):
    """
    Exécution immédiate de vente / clôture de position sur Alpaca Paper Trading.
    """
    target_alpaca = map_symbol_to_alpaca(ticker)
    target_clean = target_alpaca.replace("/", "").upper()

    try:
        positions = trading_client.get_all_positions()
        for p in positions:
            p_clean = p.symbol.replace("/", "").upper()
            if p_clean == target_clean or p.symbol == target_alpaca:
                trading_client.close_position(p.symbol)
                console.print(f"[bold red][ORDER SELL] Vente immédiate exécutée sur {p.symbol} (Alpaca Paper Trading).[/bold red]")
                return True
    except Exception as e:
        console.print(f"[red]Erreur lors de l'exécution de la vente sur {target_alpaca}: {e}[/red]")
    return False
