"""
Génération du Notebook Google Colab Ultra-Simplifié (1 Clic pour Lancer l'Entraînement).
"""
import json
import os

colab_notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🚀 Fine-Tuning Optimisé Google TimesFM — Ethereum (ETH 1h)\n",
    "\n",
    "Ce notebook permet de re-entraîner **TimesFM** sur **2 ans d'historique 1h** (~17 500 bougies) avec un GPU Colab (T4 / V100 / A100) et d'exporter les poids du modèle (`models/timesfm_eth_finetuned.pt`).\n",
    "\n",
    "### 📌 Instructions (1 Clic) :\n",
    "1. Assurez-vous d'être sous **GPU** (**Exécution > Modifier le type d'exécution > T4 GPU**).\n",
    "2. Lancez les 2 cellules ci-dessous."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Étape 1 : Clone du Dépôt & Installation des Dépendances"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cloner le repo si besoin et installer les librairies requises\n",
    "!pip install -q timesfm \"torch>=2.0.0\" ccxt pandas numpy matplotlib huggingface_hub einops\n",
    "!mkdir -p models notebooks scripts src config data\n",
    "\n",
    "import torch\n",
    "print(f\"✅ Dispositif PyTorch : {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"🔥 Nom du GPU Colab : {torch.cuda.get_device_name(0)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Étape 2 : Lancement de l'Entraînement & Fine-Tuning (10 Époques, 2 Ans d'ETH Data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Entraînement complet avec 10 époques, Learning Rate Cosine 5e-5, Smooth L1 Loss & Gradient Clipping\n",
    "!python scripts/train_timesfm.py --epochs 10 --days 730 --lr 5e-5"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Étape 3 : Téléchargement du Modèle Ré-entraîné sur votre PC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import files\n",
    "print(\"📥 Téléchargement des poids ré-entraînés...\")\n",
    "try:\n",
    "    files.download(\"models/timesfm_eth_finetuned.pt\")\n",
    "except Exception as e:\n",
    "    print(\"Téléchargez manuellement models/timesfm_eth_finetuned.pt depuis le panneau latéral de Colab.\")"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "gpuType": "T4",
   "provenance": []
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "fine_tune_timesfm_colab.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(colab_notebook, f, indent=1, ensure_ascii=False)

print(f"[SUCCESS] Notebook Google Colab régénéré avec succès dans : {output_path}")
