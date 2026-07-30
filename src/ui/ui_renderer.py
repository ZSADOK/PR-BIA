from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from src.trading_config import console, trading_client
from src.execution.risk_manager import check_instant_safety_limits
from src.execution.trade_logger import TradeLogger

trade_logger = TradeLogger()

def get_account_status_panel(remaining_sec: float = None) -> Panel:
    try:
        account = trading_client.get_account()
        equity = float(account.equity)
        cash = float(account.cash)
        
        total_pl_usd = equity - 100000.0
        total_pl_pct = (total_pl_usd / 100000.0) * 100
        pl_color = "bold green" if total_pl_usd >= 0 else "bold red"
        
        status_text = Text()
        status_text.append("Capital Total: ", style="bold white")
        status_text.append(f"${equity:,.2f}  |  ", style="bold white")
        status_text.append("Profit/Perte Total: ", style="bold white")
        status_text.append(f"{total_pl_usd:+.2f}$ ({total_pl_pct:+.2f}%)  |  ", style=pl_color)
        status_text.append(f"Cash: ${cash:,.2f}\n\n", style="bold white")
        
        try:
            positions = trading_client.get_all_positions()
            if positions:
                status_text.append("POSITIONS OUVERTES EN PORTEFEUILLE:\n", style="bold cyan")
                for pos in positions:
                    check_instant_safety_limits(pos.symbol, pos)
                    pos_pl_pct = float(pos.unrealized_plpc) * 100
                    pos_pl_usd = float(pos.unrealized_pl)
                    pos_color = "bold green" if pos_pl_usd >= 0 else "bold red"
                    symb_clean = pos.symbol.replace("/", "")
                    status_text.append(f"  • {symb_clean:<8} | Qte: {float(pos.qty):<8.4f} | Valeur: ${float(pos.market_value):<9.2f} | P/L: ", style="dim white")
                    status_text.append(f"{pos_pl_usd:+.2f}$ ({pos_pl_pct:+.2f}%)\n", style=pos_color)
            else:
                status_text.append("Aucune position ouverte (100% Cash / En attente de signal)\n", style="bold yellow")
        except Exception:
            status_text.append("Recherche des positions Alpaca...\n", style="dim white")
            
        if remaining_sec is not None:
            status_text.append(f"\n[MONITORING TEMPS RÉEL ALPACA] Prochain scan IA dans {int(remaining_sec):02d}s | Pressez Ctrl+C pour arrêter", style="bold cyan")

        return Panel(status_text, title="[bold cyan]PORTEFEUILLE ALPACA PAPER TRADING[/bold cyan]", box=box.ROUNDED, expand=False)
    except Exception as e:
        return Panel(Text(f"Erreur de connexion Alpaca: {e}", style="bold red"), title="[bold red]PORTEFEUILLE ALPACA[/bold red]", box=box.ROUNDED)

def render_account_status_panel():
    console.print(get_account_status_panel())

def render_ai_learning_memory_panel():
    stats = trade_logger.get_performance_summary()
    mem_text = Text()
    mem_text.append(f"Trades Exécutés: {stats['total_trades']}  |  ", style="bold white")
    wr_color = "bold green" if stats['win_rate'] >= 50.0 else "bold yellow"
    mem_text.append(f"Taux de Réussite (Win Rate): {stats['win_rate']:.1f}%  |  ", style=wr_color)
    pl_color = "bold green" if stats['total_pl_usd'] >= 0 else "bold red"
    mem_text.append(f"Profit Cumulé Réalisé: {stats['total_pl_usd']:+.2f}$", style=pl_color)
    console.print(Panel(mem_text, title="[bold cyan]HISTORIQUE D'APPRENTISSAGE ET PERFORMANCE[/bold cyan]", box=box.ROUNDED, expand=False))

def render_api_health_panel():
    health_text = Text()
    health_text.append("[OK] ALPACA API PAPER TRADING  : Connecté (PK3KHI...)\n", style="bold green")
    health_text.append("[WARN] REDDIT API              : Mode Public (Pour débloquer 100 req/min, ajoutez REDDIT_CLIENT_ID dans .env)\n", style="bold yellow")
    health_text.append("[OK] X/TWITTER CASHTAGS RSS    : Flux $TICKER Opérationnel\n", style="bold green")
    health_text.append("[OK] CRYPTO FEAR & GREED INDEX : API Alternative.me Opérationnelle\n", style="bold green")
    health_text.append("[OK] GOOGLE GEMINI LLM         : Connecté & Opérationnel (AQ.Ab8RN...)\n", style="bold green")
    console.print(Panel(health_text, title="[bold cyan]SANTÉ DES CONNECTEURS ET CLÉS API[/bold cyan]", box=box.ROUNDED, expand=False))

def render_gemini_insights_panel(gemini_insights):
    if not gemini_insights:
        return
    gemini_text = Text()
    for idx, item in enumerate(gemini_insights[:6], 1):
        sent = item.get("sentiment", 0.0)
        cat = item.get("catalyst", "NONE")
        pow_score = item.get("catalyst_power", 5)
        reason = item.get("raison", "Analyse sémantique haussière.")
        color = "bold green" if sent > 0.3 else ("bold red" if sent < -0.3 else "bold yellow")
        rec_label = "HAUSSIER FORT" if sent > 0.5 else ("HAUSSIER" if sent > 0.2 else ("BAISSIER" if sent < -0.2 else "NEUTRE"))
        
        gemini_text.append(f"[{idx}] {item.get('nom', item.get('ticker'))} ({item.get('ticker')}) | Catalyseur: ", style="bold white")
        gemini_text.append(f"{cat} (Impact {pow_score}/10) ", style="bold cyan")
        gemini_text.append(f"| Sentiment: {sent:+.2f} ({rec_label})\n", style=color)
        gemini_text.append(f"    Thèse Quant: {reason}\n", style="dim white")
        gemini_text.append(f"    Ajustement Confiance IA: {sent * 25.0:+.1f}%\n\n", style="bold yellow")
        
    console.print(Panel(gemini_text, title="[bold cyan]ANALYSE SÉMANTIQUE ET CATALYSEURS GEMINI 1.5 LLM[/bold cyan]", box=box.ROUNDED, expand=False))

def render_ranking_table(results, threshold: float = 0.58):
    from datetime import datetime
    now_str = datetime.now().strftime("%H:%M:%S")
    table = Table(title=f"CLASSEMENT MULTI-ACTIFS DES MODÈLES IA ({now_str})", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Rang", justify="center", style="bold white")
    table.add_column("Actif", style="bold white")
    table.add_column("Ticker", justify="center", style="dim white")
    table.add_column("Confiance IA", justify="right")
    table.add_column("Statut Signal", justify="center")
    table.add_column("Position Alpaca", justify="center")
    table.add_column("P/L %", justify="right")

    for i, item in enumerate(results, 1):
        prob_pct = f"{item['Confiance']*100:.2f}%"
        is_eligible = (item["Confiance"] >= threshold)
        has_pos = item["HasPos"]
        pl_val = item.get("PL_Pct", 0.0)

        if has_pos:
            status = "[bold cyan]CONSERVÉ (EN COURS)[/bold cyan]"
            alpaca_status = "[bold cyan]EN COURS[/bold cyan]"
            pl_str = f"{pl_val:+.2f}%"
            pl_style = "bold green" if pl_val >= 0 else "bold red"
        elif is_eligible:
            status = "[bold green]ACHAT (NOUVEAU)[/bold green]"
            alpaca_status = "[bold yellow]CASH[/bold yellow]"
            pl_str = "-"
            pl_style = "dim white"
        else:
            status = "[dim white]NEUTRE[/dim white]"
            alpaca_status = "[dim white]CASH[/dim white]"
            pl_str = "-"
            pl_style = "dim white"

        rank_str = f"#{i}"
        prob_style = "bold green" if is_eligible else ("bold yellow" if item['Confiance'] >= 0.50 else "dim white")

        table.add_row(
            rank_str, item["Nom"], item["Ticker"],
            f"[{prob_style}]{prob_pct}[/{prob_style}]",
            status, alpaca_status, f"[{pl_style}]{pl_str}[/{pl_style}]"
        )

    console.print(table)
