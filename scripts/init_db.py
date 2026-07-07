"""Initialisation de la base de données MySQL.

Crée la base `socialmetrics` et la table `tweets` (scripts/schema.sql),
puis insère le dataset annoté (data/tweets_dataset.csv).

Le script est idempotent : si la table contient déjà des données,
le seed n'est pas réinséré.

Usage :
    python scripts/init_db.py
"""

import csv
import sys
from pathlib import Path

# Permet d'importer src/ quel que soit le répertoire d'exécution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import db  # noqa: E402

SCHEMA_FILE = ROOT / "scripts" / "schema.sql"
DATASET_FILE = ROOT / "data" / "tweets_dataset.csv"


def apply_schema() -> None:
    """Exécute schema.sql : création de la base et de la table tweets."""
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    conn = db.get_connection(with_database=False)
    try:
        cursor = conn.cursor()
        for statement in sql.split(";"):
            if statement.strip():
                cursor.execute(statement)
        conn.commit()
    finally:
        conn.close()
    print("Schéma appliqué : base et table `tweets` prêtes.")


def seed_dataset() -> None:
    """Insère le dataset CSV dans la table tweets (si elle est vide)."""
    existing = db.count_tweets()
    if existing > 0:
        print(f"La table `tweets` contient déjà {existing} lignes, seed ignoré.")
        return

    with open(DATASET_FILE, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [(r["text"], int(r["positive"]), int(r["negative"])) for r in reader]

    inserted = db.insert_tweets(rows)
    print(f"{inserted} tweets annotés insérés dans la table `tweets`.")


if __name__ == "__main__":
    apply_schema()
    seed_dataset()
    print(f"Total en base : {db.count_tweets()} tweets.")
