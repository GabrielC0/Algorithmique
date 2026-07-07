"""Modèle d'analyse de sentiments.

Deux régressions logistiques (scikit-learn) partagent une même vectorisation
TF-IDF :
- le modèle "positif" prédit le label `positive` de la table tweets ;
- le modèle "négatif" prédit le label `negative`.

Le score de sentiment d'un tweet est : P(positif) - P(négatif), borné
naturellement dans [-1, 1] (-1 très négatif, 1 très positif).

Les labels binaires (predict_labels) utilisent un seuil de décision propre à
chaque classifieur, calibré par validation croisée sur le jeu d'entraînement
(maximisation du F1-score de la classe 1) : cela compense le déséquilibre des
classes et améliore le rappel des classes minoritaires par rapport au seuil
fixe de 0.5.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

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
        # Seuils de décision des labels, calibrés pendant fit().
        self.threshold_positive = 0.5
        self.threshold_negative = 0.5

    def _calibrate_threshold(self, texts, y) -> float:
        """Choisit le seuil maximisant le F1 de la classe 1, par validation
        croisée sur le jeu d'entraînement (probabilités out-of-fold, donc sans
        fuite vers le jeu de validation)."""
        pipeline = make_pipeline(clone(self.vectorizer), clone(self.clf_positive))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        proba = cross_val_predict(pipeline, texts, y, cv=cv, method="predict_proba")[:, 1]
        thresholds = np.arange(0.30, 0.66, 0.01)
        scores = [f1_score(y, (proba >= t).astype(int)) for t in thresholds]
        return float(thresholds[int(np.argmax(scores))])

    def fit(self, texts, y_positive, y_negative):
        """Entraîne les deux classifieurs et calibre leurs seuils de décision."""
        self.threshold_positive = self._calibrate_threshold(texts, y_positive)
        self.threshold_negative = self._calibrate_threshold(texts, y_negative)
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
        """Retourne les labels binaires prédits (positive, negative) par texte,
        en appliquant les seuils de décision calibrés."""
        X = self.vectorizer.transform(texts)
        pred_pos = (self.clf_positive.predict_proba(X)[:, 1] >= self.threshold_positive).astype(int)
        pred_neg = (self.clf_negative.predict_proba(X)[:, 1] >= self.threshold_negative).astype(int)
        return pred_pos, pred_neg

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
