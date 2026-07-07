"""Modèle d'analyse de sentiments.

Deux régressions logistiques (scikit-learn) partagent une même vectorisation
TF-IDF :
- le modèle "positif" prédit le label `positive` de la table tweets ;
- le modèle "négatif" prédit le label `negative`.

Le score de sentiment d'un tweet est : P(positif) - P(négatif), borné
naturellement dans [-1, 1] (-1 très négatif, 1 très positif).
"""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = MODEL_DIR / "sentiment_model.joblib"


class SentimentModel:
    """TF-IDF + deux LogisticRegression (labels positive / negative)."""

    def __init__(self):
        # N-grammes de caractères (2 à 5, bornés aux mots) : plus robustes que
        # les mots entiers sur un petit corpus de tweets français, car ils
        # capturent la morphologie (catastrophe / catastrophique) et les
        # fautes de frappe. Configuration retenue par validation croisée.
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=(2, 5),
            sublinear_tf=True,
        )
        self.clf_positive = LogisticRegression(
            max_iter=2000, C=5.0, class_weight="balanced"
        )
        self.clf_negative = LogisticRegression(
            max_iter=2000, C=5.0, class_weight="balanced"
        )

    def fit(self, texts, y_positive, y_negative):
        """Entraîne les deux classifieurs sur les tweets annotés."""
        X = self.vectorizer.fit_transform(texts)
        self.clf_positive.fit(X, y_positive)
        self.clf_negative.fit(X, y_negative)
        return self

    def predict_scores(self, texts):
        """Retourne un score de sentiment dans [-1, 1] pour chaque texte."""
        X = self.vectorizer.transform(texts)
        proba_pos = self.clf_positive.predict_proba(X)[:, 1]
        proba_neg = self.clf_negative.predict_proba(X)[:, 1]
        return (proba_pos - proba_neg).tolist()

    def predict_labels(self, texts):
        """Retourne les labels binaires prédits (positive, negative) par texte."""
        X = self.vectorizer.transform(texts)
        return self.clf_positive.predict(X), self.clf_negative.predict(X)

    def save(self, path: Path = MODEL_FILE) -> Path:
        """Sauvegarde le modèle entraîné sur disque (joblib)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path = MODEL_FILE) -> "SentimentModel":
        """Charge un modèle entraîné depuis le disque."""
        if not path.exists():
            raise FileNotFoundError(
                f"Modèle introuvable ({path}). Lancez d'abord : python -m src.train"
            )
        return joblib.load(path)
