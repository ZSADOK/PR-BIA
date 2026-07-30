import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from src.execution.risk_manager import RiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Backtester:
    """
    Module de Backtesting Événementiel avec Moteur de Gestion des Risques Intégré
    (Stop-Loss, Take-Profit, Trailing Stop et Volatility Sizing).
    """

    def __init__(self, initial_capital: float = 10000.0, fee_per_trade: float = 0.0005):
        self.initial_capital = initial_capital
        self.fee_per_trade = fee_per_trade
        self.risk_mgr = RiskManager(
            stop_loss_pct=0.02,       # Stop-Loss à -2.0% max
            take_profit_pct=0.04,     # Take-Profit à +4.0%
            trailing_stop_pct=0.015,  # Trailing stop à -1.5%
            max_risk_per_trade=0.02
        )

    def run_backtest(
        self,
        df_test: pd.DataFrame,
        preds: np.ndarray,
        probs: np.ndarray,
        model_name: str = "IA Strategy",
        prob_threshold: float = 0.54,
        use_risk_management: bool = True
    ) -> Dict:
        res = df_test.copy()
        real_ret = res["Close"].pct_change().fillna(0.0)

        if use_risk_management:
            # Exécution événementielle avec Moteur de Gestion des Risques
            net_strategy_ret, positions, sizes = self.risk_mgr.execute_risk_managed_backtest(
                df_test=res,
                probs=probs,
                prob_threshold=prob_threshold,
                initial_capital=self.initial_capital,
                fee_per_trade=self.fee_per_trade
            )
        else:
            positions = (probs > prob_threshold).astype(int)
            strategy_ret = positions * real_ret
            trades = np.abs(np.diff(positions, prepend=0))
            costs = trades * self.fee_per_trade
            net_strategy_ret = strategy_ret - costs

        # Courbes de capital (Equity Curves)
        equity_benchmark = pd.Series(self.initial_capital * (1.0 + real_ret).cumprod(), index=res.index)
        equity_strategy = pd.Series(self.initial_capital * (1.0 + net_strategy_ret).cumprod(), index=res.index)

        total_ret_bench = (equity_benchmark.iloc[-1] / self.initial_capital - 1.0) * 100
        total_ret_strat = (equity_strategy.iloc[-1] / self.initial_capital - 1.0) * 100

        # Sharpe Ratio
        daily_mean = net_strategy_ret.mean()
        daily_std = net_strategy_ret.std() + 1e-8
        sharpe_ratio = (daily_mean / daily_std) * np.sqrt(252)

        # Max Drawdown
        peak = equity_strategy.cummax()
        drawdown = (equity_strategy - peak) / peak
        max_drawdown = drawdown.min() * 100

        # Win Rate
        active_days = net_strategy_ret[positions == 1]
        win_rate = (active_days > 0).mean() * 100 if len(active_days) > 0 else 0.0

        results = {
            "model_name": model_name,
            "initial_capital": self.initial_capital,
            "final_capital_bench": float(equity_benchmark.iloc[-1]),
            "final_capital_strat": float(equity_strategy.iloc[-1]),
            "return_benchmark_pct": float(total_ret_bench),
            "return_strategy_pct": float(total_ret_strat),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown_pct": float(max_drawdown),
            "win_rate_pct": float(win_rate),
            "equity_benchmark": equity_benchmark,
            "equity_strategy": equity_strategy,
            "positions": positions
        }

        logger.info(f"[{model_name}] (Risk-Managed) Rdt: {total_ret_strat:.2f}% vs Bench: {total_ret_bench:.2f}% | Sharpe: {sharpe_ratio:.2f} | MaxDD: {max_drawdown:.2f}%")
        return results

    def plot_backtest_results(self, results_dict: Dict[str, Dict], save_path: str = "data/backtest_chart.png"):
        plt.figure(figsize=(12, 6))

        first_res = list(results_dict.values())[0]
        plt.plot(first_res["equity_benchmark"].index, first_res["equity_benchmark"], label="Benchmark Buy & Hold", color="black", linestyle="--", alpha=0.7)

        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
        for idx, (name, res) in enumerate(results_dict.items()):
            clean_name = name.replace("🔥 ", "").replace("⭐ ", "")
            color = colors[idx % len(colors)]
            label = f"{clean_name} (Rdt: {res['return_strategy_pct']:.1f}%, Sharpe: {res['sharpe_ratio']:.2f}, MaxDD: {res['max_drawdown_pct']:.1f}%)"
            plt.plot(res["equity_strategy"].index, res["equity_strategy"], label=label, linewidth=2, color=color)

        plt.title("Backtest avec Moteur de Sécurité (Stop-Loss -2%, Take-Profit +4%, Trailing Stop)", fontsize=13, fontweight="bold")
        plt.xlabel("Date")
        plt.ylabel("Capital Portfolio ($)")
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        plt.legend(loc="upper left")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Graphique sécurisé sauvegardé : {save_path}")

if __name__ == "__main__":
    print("Backtester avec RiskManager initialisé.")
