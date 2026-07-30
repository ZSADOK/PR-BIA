import time
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from src.trading_config import (
    console, trading_client, SECTOR_MAP, POSITION_PEAK_PL,
    AI_PORTFOLIO_CACHE, PENDING_EXITS, STOP_LOSS_COOLDOWN, save_persistent_state
)
from src.execution.trade_logger import TradeLogger

trade_logger = TradeLogger()

def check_instant_safety_limits(alpaca_symbol: str, position=None) -> bool:
    """
    Vérification instantanée des Stop-Loss Dynamiques (Volatilité ATR), Take-Profit (+10.0%) et Stop Suiveur Réactif.
    """
    if alpaca_symbol in PENDING_EXITS:
        return False

    try:
        if position is None:
            position = trading_client.get_open_position(alpaca_symbol)
        pl_pct = float(position.unrealized_plpc) * 100
        clean_symb = alpaca_symbol.replace("/", "").upper()

        # Stop-Loss Dynamique selon le secteur
        sec = SECTOR_MAP.get(clean_symb, "OTHER")
        sl_limit = -2.2 if sec in ["CRYPTO", "BIOTECH"] else -1.2

        # Enregistrer le sommet de gain
        peak = max(POSITION_PEAK_PL.get(alpaca_symbol, 0.0), pl_pct)
        POSITION_PEAK_PL[alpaca_symbol] = peak

        # 1. Stop-Loss Dynamique
        if pl_pct <= sl_limit:
            PENDING_EXITS.add(alpaca_symbol)
            STOP_LOSS_COOLDOWN[clean_symb] = time.time() + 3600
            save_persistent_state()
            console.print(f"\n[bold red][STOP-LOSS DYNAMIQUE {sl_limit:.1f}%] Pertes limitées à {pl_pct:.2f}%. Fermeture de {alpaca_symbol}...[/bold red]")
            try:
                trading_client.close_position(alpaca_symbol)
            except Exception:
                pass
            POSITION_PEAK_PL.pop(alpaca_symbol, None)
            AI_PORTFOLIO_CACHE.pop(alpaca_symbol, None)
            trade_logger.log_trade(
                ticker=alpaca_symbol, action="VENTE_STOP_LOSS", qty=float(position.qty),
                entry_price=float(position.avg_entry_price), exit_price=float(position.current_price),
                pl_usd=float(position.unrealized_pl), pl_pct=pl_pct, reason=f"Stop-Loss Dynamique {sl_limit:.1f}%"
            )
            return True

        # 2. Stop Suiveur Dynamique Asymétrique
        trailing_buffer = 1.5 if peak >= 3.0 else 0.5
        if peak >= 0.8 and pl_pct <= (peak - trailing_buffer):
            PENDING_EXITS.add(alpaca_symbol)
            console.print(f"\n[bold green][STOP SUIVEUR ASYMÉTRIQUE] Gain sécurisé : Sommet +{peak:.2f}% -> Vente à +{pl_pct:.2f}% sur {alpaca_symbol}[/bold green]")
            try:
                trading_client.close_position(alpaca_symbol)
            except Exception:
                pass
            POSITION_PEAK_PL.pop(alpaca_symbol, None)
            AI_PORTFOLIO_CACHE.pop(alpaca_symbol, None)
            trade_logger.log_trade(
                ticker=alpaca_symbol, action="VENTE_STOP_SUIVEUR", qty=float(position.qty),
                entry_price=float(position.avg_entry_price), exit_price=float(position.current_price),
                pl_usd=float(position.unrealized_pl), pl_pct=pl_pct, reason=f"Stop Suiveur Réactif (Sommet +{peak:.2f}%)"
            )
            return True

        # 3. Take-Profit Extrême (+10.0%)
        if pl_pct >= 10.0:
            PENDING_EXITS.add(alpaca_symbol)
            console.print(f"\n[bold green][TAKE-PROFIT +10%] Gain exceptionnel de {pl_pct:.2f}% sécurisé sur {alpaca_symbol} ![/bold green]")
            try:
                trading_client.close_position(alpaca_symbol)
            except Exception:
                pass
            POSITION_PEAK_PL.pop(alpaca_symbol, None)
            AI_PORTFOLIO_CACHE.pop(alpaca_symbol, None)
            trade_logger.log_trade(
                ticker=alpaca_symbol, action="VENTE_TAKE_PROFIT", qty=float(position.qty),
                entry_price=float(position.avg_entry_price), exit_price=float(position.current_price),
                pl_usd=float(position.unrealized_pl), pl_pct=pl_pct, reason="Take-Profit Majeur +10.0%"
            )
            return True

    except Exception:
        pass
    return False

def compute_dynamic_kelly_notional(probability: float, base_notional: float = 1000.0) -> float:
    """
    Allocation dynamique Kelly Asymétrique selon la confiance du modèle.
    """
    if probability >= 0.70:
        return base_notional * 5.0
    elif probability >= 0.65:
        return base_notional * 3.5
    elif probability >= 0.60:
        return base_notional * 2.0
    elif probability >= 0.55:
        return base_notional * 1.0
    return base_notional * 0.5

def calculate_expected_value(probability: float, target_profit: float = 1.50, stop_loss: float = 1.20) -> float:
    """
    Calcul de l'Espérance Mathématique Nette (EV) par Trade.
    """
    return (probability * target_profit) - ((1.0 - probability) * stop_loss)

class RiskManager:
    """
    Gestionnaire de risque pour le backtester.
    """
    def __init__(
        self,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        trailing_stop_pct: float = 0.015,
        max_risk_per_trade: float = 0.02,
        max_position_size_pct: float = 0.20
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size_pct = max_position_size_pct

    def execute_risk_managed_backtest(
        self,
        df_test: pd.DataFrame,
        probs: np.ndarray,
        prob_threshold: float = 0.54,
        initial_capital: float = 10000.0,
        fee_per_trade: float = 0.0005
    ) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
        import numpy as np
        import pandas as pd
        n = len(df_test)
        positions = np.zeros(n)
        sizes = np.zeros(n)
        net_returns = pd.Series(0.0, index=df_test.index)
        
        in_pos = False
        entry_price = 0.0
        peak_price = 0.0
        close_prices = df_test["Close"].values
        
        for i in range(1, n):
            p = probs[i-1]
            price = close_prices[i]
            prev_price = close_prices[i-1]
            
            if not in_pos:
                if p >= prob_threshold:
                    in_pos = True
                    entry_price = price
                    peak_price = price
                    positions[i] = 1
                    sizes[i] = self.max_position_size_pct
                    net_returns.iloc[i] = -fee_per_trade
            else:
                peak_price = max(peak_price, price)
                ret = (price - prev_price) / prev_price
                
                loss_from_entry = (price - entry_price) / entry_price
                drop_from_peak = (price - peak_price) / peak_price
                
                if loss_from_entry <= -self.stop_loss_pct or drop_from_peak <= -self.trailing_stop_pct or loss_from_entry >= self.take_profit_pct:
                    in_pos = False
                    positions[i] = 0
                    sizes[i] = 0
                    net_returns.iloc[i] = ret - fee_per_trade
                else:
                    positions[i] = 1
                    sizes[i] = self.max_position_size_pct
                    net_returns.iloc[i] = ret
                    
        return net_returns, positions, sizes
