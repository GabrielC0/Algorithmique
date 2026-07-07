# SocialMetrics AI — API d'Analyse de Sentiments

TP Final : API **Flask** d'analyse de sentiments de tweets, basée sur une
**régression logistique** (`scikit-learn`) entraînée à partir de tweets annotés
stockés dans **MySQL**, avec **réentraînement hebdomadaire automatisé** et
[rapport d'évaluation PDF](report/rapport_evaluation.pdf).

Pour chaque tweet envoyé à l'API, un **score de sentiment entre -1 (très
négatif) et 1 (très positif)** est retourné.

## Sommaire

- [Structure du projet](#structure-du-projet)
- [Installation](#installation)
- [Initialisation de la base de données](#initialisation-de-la-base-de-données)
- [Entraînement du modèle](#entraînement-du-modèle)
- [Utilisation de l'API](#utilisation-de-lapi)
- [Réentraînement automatisé](#réentraînement-automatisé)
- [Rapport d'évaluation](#rapport-dévaluation)
- [Fonctionnement du modèle](#fonctionnement-du-modèle)

## Structure du projet

```
├── app.py                     # API Flask (POST /analyze, GET /health)
├── src/
│   ├── db.py                  # Connexion MySQL + accès à la table tweets
│   ├── model.py               # Modèle de sentiment (TF-IDF + 2 LogisticRegression)
│   └── train.py               # Entraînement + évaluation sur jeu de validation
├── scripts/
│   ├── schema.sql             # Création de la base et de la table tweets
│   ├── init_db.py             # Initialisation de la BDD + insertion du dataset
│   ├── retrain.py             # Réentraînement du modèle (cron / tâche planifiée)
│   ├── crontab.example        # Planification hebdomadaire (Linux)
│   └── setup_windows_task.ps1 # Planification hebdomadaire (Windows)
├── data/tweets_dataset.csv    # Dataset de ~450 tweets français annotés
├── report/
│   ├── generate_report.py     # Génération du rapport d'évaluation (PDF)
│   └── rapport_evaluation.pdf # Rapport : matrices de confusion + analyse
├── models/                    # Artefacts du modèle entraîné (générés, non versionnés)
├── docker-compose.yml         # MySQL 8 via Docker
└── requirements.txt
```

## Installation

### Prérequis

- Python 3.10+
- Docker (pour MySQL) — ou un serveur MySQL local existant

### 1. Cloner et installer les dépendances

```bash
git clone <url-du-repo>
cd projet-final
python -m venv venv
# Windows : venv\Scripts\activate   |   Linux/macOS : source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurer l'environnement

```bash
cp .env.example .env
```

Variables du fichier `.env` :

| Variable | Description | Défaut |
|---|---|---|
| `DB_HOST` / `DB_PORT` | Hôte / port MySQL | `127.0.0.1` / `3306` |
| `DB_USER` / `DB_PASSWORD` | Identifiants MySQL | `socialmetrics` |
| `DB_NAME` | Nom de la base | `socialmetrics` |
| `FLASK_HOST` / `FLASK_PORT` | Écoute de l'API | `0.0.0.0` / `5000` |

> Si le mot de passe contient `#`, l'entourer de guillemets : `DB_PASSWORD="mon#mdp"`.

### 3. Lancer MySQL

**Avec Docker (recommandé)** :

```bash
docker compose up -d
```

**Avec un MySQL local** : renseigner dans `.env` un utilisateur ayant le droit
`CREATE DATABASE` (la base et la table sont créées à l'étape suivante).

## Initialisation de la base de données

```bash
python scripts/init_db.py
```

Le script applique `scripts/schema.sql` (création de la base `socialmetrics` et
de la table `tweets`) puis insère le dataset annoté `data/tweets_dataset.csv`
(~450 tweets français). Il est idempotent : relancé, il ne duplique pas les données.

Structure de la table `tweets` :

| Colonne | Type | Description |
|---|---|---|
| `id` | INT AUTO_INCREMENT | Identifiant unique |
| `text` | TEXT | Contenu du tweet |
| `positive` | TINYINT(1) | 1 si le tweet est jugé positif, 0 sinon |
| `negative` | TINYINT(1) | 1 si le tweet est jugé négatif, 0 sinon |

## Entraînement du modèle

```bash
python -m src.train
```

- Charge les tweets annotés depuis MySQL.
- Split 80 % entraînement / 20 % validation (stratifié, reproductible).
- Entraîne les deux régressions logistiques (labels `positive` et `negative`).
- Affiche matrices de confusion, précision, rappel et F1-score de validation.
- Sauvegarde `models/sentiment_model.joblib` et `models/metrics.json`.

## Utilisation de l'API

Démarrer l'API (modèle entraîné requis) :

```bash
python app.py
# API disponible sur http://localhost:5000
```

### `POST /analyze` — analyse de sentiments

Corps attendu : un tableau JSON de tweets (`string[]`).

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '["J'\''adore ce produit, il est génial !", "Service catastrophique, je suis furieux", "Le magasin ouvre demain matin"]'
```

Réponse `200 OK` — un score entre **-1 (très négatif)** et **1 (très positif)**
par tweet :

```json
{
  "J'adore ce produit, il est génial !": 0.672,
  "Service catastrophique, je suis furieux": -0.452,
  "Le magasin ouvre demain matin": -0.065
}
```

> Les clés d'un objet JSON étant uniques, les tweets envoyés en double
> n'apparaissent qu'une fois dans la réponse (le score d'un texte identique
> est identique).

### Gestion des erreurs

| Cas | Code | Réponse |
|---|---|---|
| Corps non JSON | 400 | `{"error": "Le corps de la requête doit être du JSON valide…"}` |
| JSON qui n'est pas un tableau | 400 | `{"error": "Le corps de la requête doit être un tableau de chaînes…"}` |
| Liste vide `[]` | 400 | `{"error": "La liste de tweets est vide."}` |
| Élément non-string (`[1, true]`) | 400 | `{"error": "Tous les éléments de la liste doivent être des chaînes…"}` |
| Chaîne vide ou espaces | 400 | `{"error": "Les tweets ne peuvent pas être des chaînes vides."}` |
| Modèle non entraîné | 503 | `{"error": "Modèle non entraîné. Lancez d'abord : python -m src.train"}` |
| Route inconnue / mauvaise méthode | 404 / 405 | message JSON explicite |

Exemples :

```bash
# Liste vide -> 400
curl -X POST http://localhost:5000/analyze -H "Content-Type: application/json" -d '[]'

# Mauvais format -> 400
curl -X POST http://localhost:5000/analyze -H "Content-Type: application/json" -d '{"tweet": "coucou"}'
```

### `GET /health` — état du service

```bash
curl http://localhost:5000/health
# {"database": "ok", "model": "ok", "status": "ok"}
```

## Réentraînement automatisé

Le modèle est réentraîné **chaque semaine** sur les données les plus récentes
de la table `tweets`. Le script `scripts/retrain.py` :

1. recharge l'intégralité de la table `tweets` (nouvelles annotations comprises) ;
2. réentraîne et réévalue le modèle ;
3. remplace `models/sentiment_model.joblib` — **l'API le recharge à chaud**,
   sans redémarrage ;
4. journalise le résultat dans `logs/retrain.log`.

Exécution manuelle :

```bash
python scripts/retrain.py
```

### Planification Linux (cron)

```bash
crontab -e
# puis ajouter (lundi 03h00), en adaptant le chemin :
0 3 * * 1 cd /chemin/vers/projet-final && ./venv/bin/python scripts/retrain.py >> logs/cron.log 2>&1
```

Voir `scripts/crontab.example`.

### Planification Windows (Planificateur de tâches)

```powershell
# PowerShell en administrateur, depuis la racine du projet :
.\scripts\setup_windows_task.ps1
```

Crée la tâche planifiée `SocialMetrics-Retrain` (tous les lundis à 03h00).

## Rapport d'évaluation

Le rapport PDF complet est disponible : **[report/rapport_evaluation.pdf](report/rapport_evaluation.pdf)**.

Il contient les deux matrices de confusion (labels `positive` et `negative`)
avec interprétation, les mesures de précision/rappel/F1-score par classe,
l'analyse des forces, faiblesses et biais du modèle, et les recommandations
d'amélioration. Pour le régénérer après un réentraînement :

```bash
python report/generate_report.py
```

## Fonctionnement du modèle

Deux classifieurs `LogisticRegression` (scikit-learn) partagent une même
vectorisation TF-IDF (n-grammes de caractères 2-5, choisis par validation
croisée) :

- le premier prédit le label **`positive`** de la table `tweets` ;
- le second prédit le label **`negative`**.

Le score renvoyé par l'API est `P(positif) − P(négatif)`, naturellement borné
dans `[-1, 1]`. Le seuil de décision de chaque classifieur (labels binaires du
rapport d'évaluation) est calibré à l'entraînement par validation croisée
(maximisation du F1-score de la classe 1), ce qui compense le déséquilibre des
classes et améliore le rappel des tweets polarisés.

Cette architecture à deux labels indépendants permet de gérer
les tweets neutres (`positive=0, negative=0`) comme les tweets mitigés
(`positive=1, negative=1`), et de produire les deux matrices de confusion du
rapport d'évaluation.
