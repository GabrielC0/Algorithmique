"""Réentraînement hebdomadaire du modèle de sentiment.

Relance l'entraînement complet sur les données les plus récentes de la table
`tweets`, remplace l'artefact models/sentiment_model.joblib (rechargé à chaud
par l'API) et journalise le résultat dans logs/retrain.log.

Ce script est appelé automatiquement chaque semaine :
- Linux   : via cron (voir scripts/crontab.example)
- Windows : via le Planificateur de tâches (voir scripts/setup_windows_task.ps1)

Usage manuel :
    python scripts/retrain.py
"""

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train import train_and_evaluate  # noqa: E402

LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "retrain.log"


def log(message: str) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    log("Début du réentraînement du modèle...")
    try:
        metrics = train_and_evaluate()
    except Exception as exc:
        log(f"ECHEC du réentraînement : {exc}")
        sys.exit(1)

    log(
        "Réentraînement terminé : "
        f"{metrics['dataset_size']} tweets, "
        f"accuracy positive={metrics['positive']['accuracy']:.2%}, "
        f"accuracy negative={metrics['negative']['accuracy']:.2%}"
    )
