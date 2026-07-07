"""Entraînement du modèle de sentiment sur les tweets annotés de MySQL.

- Charge la table `tweets`.
- Sépare les données en train (80 %) / validation (20 %).
- Entraîne les deux régressions logistiques (labels positive et negative).
- Évalue sur le jeu de validation : matrices de confusion, précision,
  rappel, F1-score pour chaque label.
- Sauvegarde le modèle (models/sentiment_model.joblib) et les métriques
  (models/metrics.json), réutilisées par le rapport d'évaluation.

Usage :
    python -m src.train
"""

import json
import sys
from datetime import datetime, timezone

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

from src import db
from src.model import MODEL_DIR, SentimentModel

METRICS_FILE = MODEL_DIR / "metrics.json"
RANDOM_STATE = 42


def evaluate_label(y_true, y_pred) -> dict:
    """Matrice de confusion + précision/rappel/F1 pour un label binaire."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    return {
        "confusion_matrix": cm.tolist(),  # [[VN, FP], [FN, VP]]
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "classes": {
            str(label): {
                "precision": round(precision[i], 4),
                "recall": round(recall[i], 4),
                "f1_score": round(f1[i], 4),
                "support": int(support[i]),
            }
            for i, label in enumerate([0, 1])
        },
    }


def train_and_evaluate() -> dict:
    """Entraîne le modèle depuis la base et retourne les métriques de validation."""
    rows = db.fetch_all_tweets()
    if len(rows) < 10:
        raise RuntimeError(
            f"Seulement {len(rows)} tweets en base : initialisez la base "
            "(python scripts/init_db.py) avant d'entraîner."
        )

    texts = [r[0] for r in rows]
    y_pos = [int(r[1]) for r in rows]
    y_neg = [int(r[2]) for r in rows]

    # Stratification sur le couple (positive, negative) pour conserver la
    # répartition des classes dans le jeu de validation.
    strata = [f"{p}{n}" for p, n in zip(y_pos, y_neg)]
    idx_train, idx_val = train_test_split(
        range(len(texts)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=strata,
    )

    model = SentimentModel().fit(
        [texts[i] for i in idx_train],
        [y_pos[i] for i in idx_train],
        [y_neg[i] for i in idx_train],
    )

    val_texts = [texts[i] for i in idx_val]
    pred_pos, pred_neg = model.predict_labels(val_texts)

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(texts),
        "train_size": len(idx_train),
        "validation_size": len(idx_val),
        "positive": evaluate_label([y_pos[i] for i in idx_val], pred_pos),
        "negative": evaluate_label([y_neg[i] for i in idx_val], pred_neg),
    }

    model_path = model.save()
    METRICS_FILE.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Modèle sauvegardé : {model_path}")
    print(f"Métriques sauvegardées : {METRICS_FILE}")
    return metrics


def print_report(metrics: dict) -> None:
    """Affiche un résumé lisible des métriques de validation."""
    print(
        f"\nDataset : {metrics['dataset_size']} tweets "
        f"({metrics['train_size']} train / {metrics['validation_size']} validation)"
    )
    for label in ("positive", "negative"):
        m = metrics[label]
        vn, fp = m["confusion_matrix"][0]
        fn, vp = m["confusion_matrix"][1]
        cls = m["classes"]["1"]
        print(f"\n--- Label `{label}` (accuracy {m['accuracy']:.2%}) ---")
        print(f"Matrice de confusion : VN={vn} FP={fp} / FN={fn} VP={vp}")
        print(
            f"Classe 1 : précision={cls['precision']:.2%} "
            f"rappel={cls['recall']:.2%} F1={cls['f1_score']:.2%}"
        )


if __name__ == "__main__":
    try:
        print_report(train_and_evaluate())
    except Exception as exc:  # erreur BDD, base vide, etc.
        print(f"Erreur lors de l'entraînement : {exc}", file=sys.stderr)
        sys.exit(1)
