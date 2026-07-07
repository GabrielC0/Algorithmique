"""API Flask d'analyse de sentiments — SocialMetrics AI.

Endpoints :
- POST /analyze : reçoit une liste de tweets (string[]) et retourne un score
  de sentiment entre -1 (très négatif) et 1 (très positif) par tweet.
- GET /health : état de l'API (base de données + modèle).

Usage :
    python app.py
"""

import os
import threading

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from src import db
from src.model import MODEL_FILE, SentimentModel

load_dotenv()

app = Flask(__name__)
app.json.ensure_ascii = False  # réponses JSON avec accents lisibles

# Le modèle est chargé une seule fois puis rechargé automatiquement si le
# fichier change (après un réentraînement hebdomadaire, sans redémarrer l'API).
_model_lock = threading.Lock()
_model: SentimentModel | None = None
_model_mtime: float | None = None


def get_model() -> SentimentModel:
    """Retourne le modèle courant, rechargé si l'artefact a été mis à jour."""
    global _model, _model_mtime
    mtime = MODEL_FILE.stat().st_mtime  # FileNotFoundError si non entraîné
    with _model_lock:
        if _model is None or mtime != _model_mtime:
            _model = SentimentModel.load()
            _model_mtime = mtime
        return _model


def error_response(message: str, status: int):
    return jsonify({"error": message}), status


@app.post("/analyze")
def analyze():
    """Analyse le sentiment d'une liste de tweets.

    Body JSON attendu : ["tweet 1", "tweet 2", ...]
    Réponse : {"tweet 1": score1, "tweet 2": score2, ...} avec score ∈ [-1, 1].
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return error_response(
            "Le corps de la requête doit être du JSON valide "
            '(en-tête Content-Type: application/json attendu).',
            400,
        )
    if not isinstance(payload, list):
        return error_response(
            "Le corps de la requête doit être un tableau de chaînes "
            '(exemple : ["super produit", "quelle déception"]).',
            400,
        )
    if len(payload) == 0:
        return error_response("La liste de tweets est vide.", 400)
    if not all(isinstance(t, str) for t in payload):
        return error_response(
            "Tous les éléments de la liste doivent être des chaînes de caractères.",
            400,
        )
    if any(not t.strip() for t in payload):
        return error_response("Les tweets ne peuvent pas être des chaînes vides.", 400)

    try:
        model = get_model()
    except FileNotFoundError:
        return error_response(
            "Modèle non entraîné. Lancez d'abord : python -m src.train", 503
        )

    scores = model.predict_scores(payload)
    return jsonify({tweet: round(score, 3) for tweet, score in zip(payload, scores)})


@app.get("/health")
def health():
    """État de santé : connexion MySQL et disponibilité du modèle."""
    database_ok = db.check_connection()
    model_ok = MODEL_FILE.exists()
    status = 200 if (database_ok and model_ok) else 503
    return (
        jsonify(
            {
                "status": "ok" if status == 200 else "degraded",
                "database": "ok" if database_ok else "unreachable",
                "model": "ok" if model_ok else "not trained",
            }
        ),
        status,
    )


@app.errorhandler(404)
def not_found(_):
    return error_response("Route inconnue. Endpoints : POST /analyze, GET /health.", 404)


@app.errorhandler(405)
def method_not_allowed(_):
    return error_response("Méthode non autorisée sur cette route.", 405)


@app.errorhandler(500)
def internal_error(_):
    return error_response("Erreur interne du serveur.", 500)


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
    )
