import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ÉTAPE 2 : Étude des Données ETH 1h et Modélisation Zero-Shot TimesFM\n",
    "\n",
    "Ce notebook orchestre :\n",
    "1. **Le téléchargement massif de données historiques Ethereum (ETH/USDT 1h)** sur 1 an complet (8 760+ bougies 1h via CCXT/Binance).\n",
    "2. **L'analyse exploratoire (EDA)** et l'application du **Pré-Screening Volume & Momentum** (RVOL > 1.2x, SMA 50/200, RSI).\n",
    "3. **La modélisation avec TimesFM** (Google Time-Series Foundation Model) pour prédire le prix de clôture de l'heure suivante $P_{t+1}$.\n",
    "4. **La conversion du signal continu en signal binaire de trading** (1 = Achat/Long, 0 = Neutre/Vente) et l'évaluation de sa précision directionnelle."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import os\n",
    "sys.path.append('..')\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from datetime import datetime, timedelta\n",
    "\n",
    "from src.data_loader import get_large_eth_data\n",
    "from src.screening.momentum_screener import MomentumScreener\n",
    "from src.models.timesfm_engine import TimesFMEngine\n",
    "from config.settings import config\n",
    "\n",
    "print(f\"[INIT] Modèle configuré : Symbol={config.symbol}, TF={config.timeframe}, ContextLen={config.context_len}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Téléchargement Massif des Données Historiques Ethereum (ETH/USDT 1h - 1 An)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ingestion de 365 jours de données 1h (~8 760 bougies continuous)\n",
    "print(\"[DATA] Ingestion massive des bougies 1h pour ETH/USDT (1 An complet)...\")\n",
    "df = get_large_eth_data(symbol=config.symbol, timeframe=config.timeframe, days_back=365, force_refresh=False)\n",
    "\n",
    "print(f\"[DATA SUCCESS] {len(df)} bougies de 1 heure chargées avec succès.\")\n",
    "print(f\" - Début de l'historique : {df.index[0]}\")\n",
    "print(f\" - Fin de l'historique   : {df.index[-1]}\")\n",
    "print(df.tail())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Analyse Exploratoire (EDA) & Application du Pré-Screening Volume/Momentum\n",
    "\n",
    "Conformément aux règles quantitatives (`AGENTS.md`) :\n",
    "- **Volume Relatif (RVOL)** : `Volume / SMA_Volume(20) > 1.2x` (Validation de l'activité institutionnelle).\n",
    "- **Filtre de Tendance** : `Close > SMA(50)` et `Close > SMA(200)`.\n",
    "- **Filtre RSI** : `50 <= RSI(14) <= 72` (Momentum haussier sans être en surachat)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "screener = MomentumScreener(rvol_threshold=config.rvol_threshold)\n",
    "df_screened = screener.compute_indicators(df)\n",
    "\n",
    "eligible_count = df_screened['Screening_Passed'].sum()\n",
    "eligibility_rate = (eligible_count / len(df_screened)) * 100\n",
    "\n",
    "print(f\"[EDA] Bougies validées par le pré-screener : {eligible_count} / {len(df_screened)} ({eligibility_rate:.2f}% des opportunités sur 1 an)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualisation de la distribution des données sur l'année\n",
    "plt.figure(figsize=(14, 6))\n",
    "plt.plot(df_screened.index, df_screened['Close'], label='ETH/USDT Close Price (1h)', alpha=0.7, color='blue')\n",
    "plt.plot(df_screened.index, df_screened['SMA_50'], label='SMA 50h', linestyle='--', color='orange', alpha=0.6)\n",
    "plt.plot(df_screened.index, df_screened['SMA_200'], label='SMA 200h', linestyle='--', color='red', alpha=0.6)\n",
    "\n",
    "# Surbrillance des opportunités qualifiées\n",
    "valid_points = df_screened[df_screened['Screening_Passed']]\n",
    "plt.scatter(valid_points.index, valid_points['Close'], color='green', label='Screener Validated (RVOL > 1.2 & Trend OK)', s=10, zorder=5)\n",
    "\n",
    "plt.title(f'Ethereum (ETH/USDT) 1h - Historique {len(df)} Heures (1 An)')\n",
    "plt.xlabel('Date')\n",
    "plt.ylabel('Prix (USDT)')\n",
    "plt.legend()\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Configuration & Inférence du Modèle TimesFM\n",
    "\n",
    "Nous configurons le modèle **TimesFM** avec une fenêtre de contexte de **512 bougies 1h** (~21 jours de mémoire) pour effectuer des prédictions Zero-Shot sur l'horizon **H+1**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "timesfm_engine = TimesFMEngine(context_len=config.context_len, horizon_len=config.horizon_len, backend=config.backend)\n",
    "\n",
    "# Inférence de démonstration sur la dernière bougie\n",
    "latest_signal = timesfm_engine.generate_signal(df_screened, screener_passed=df_screened.iloc[-1]['Screening_Passed'])\n",
    "\n",
    "print(\"=== RÉSULTAT DU SIGNAL TEMPS RÉEL ETH (H+1) ===\")\n",
    "for key, val in latest_signal.items():\n",
    "    print(f\" - {key}: {val}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Backtest In-Sample & Signal Binaire (0/1)\n",
    "\n",
    "Nous simulons l'exécution du modèle TimesFM sur les 500 dernières heures pour évaluer la précision du signal binaire."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "results = []\n",
    "window_eval = 500  # Évaluation sur 500 heures d'historique\n",
    "\n",
    "print(f\"[MODEL] Lancement du backtest directionnel TimesFM sur {window_eval} bougies 1h...\")\n",
    "for i in range(len(df_screened) - window_eval, len(df_screened) - 1):\n",
    "    sub_df = df_screened.iloc[:i+1]\n",
    "    screener_passed = bool(sub_df.iloc[-1]['Screening_Passed'])\n",
    "    \n",
    "    res = timesfm_engine.generate_signal(sub_df, screener_passed=screener_passed)\n",
    "    actual_next_price = float(df_screened.iloc[i+1]['Close'])\n",
    "    actual_return = (actual_next_price - res['current_price']) / res['current_price']\n",
    "    actual_binary = 1 if actual_return > 0 else 0\n",
    "    \n",
    "    is_correct = (res['signal_binary'] == actual_binary) if res['signal_binary'] == 1 else True\n",
    "    \n",
    "    results.append({\n",
    "        'timestamp': df_screened.index[i],\n",
    "        'current_price': res['current_price'],\n",
    "        'predicted_price': res['predicted_price'],\n",
    "        'signal_binary': res['signal_binary'],\n",
    "        'actual_next_price': actual_next_price,\n",
    "        'actual_return_pct': actual_return * 100,\n",
    "        'is_correct': is_correct\n",
    "    })\n",
    "\n",
    "res_df = pd.DataFrame(results)\n",
    "long_trades = res_df[res_df['signal_binary'] == 1]\n",
    "win_rate = (long_trades['actual_return_pct'] > 0).mean() * 100 if len(long_trades) > 0 else 0\n",
    "\n",
    "print(f\"[METRICS] Signaux d'Achat générés (Signal = 1) : {len(long_trades)} / {len(res_df)}\")\n",
    "print(f\"[METRICS] Taux de réussite directionnel (Win Rate sur Achat) : {win_rate:.2f}%\")\n",
    "print(f\"[METRICS] Rendement Cumulé du Signal : {long_trades['actual_return_pct'].sum():.2f}%\")"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "etude-data.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1, ensure_ascii=False)

print(f"[SUCCESS] Notebook mis à jour avec 1 AN d'historique (8760 bougies) dans : {output_path}")
