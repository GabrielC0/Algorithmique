"""Génération du rapport d'évaluation du modèle (PDF).

Lit les métriques de validation produites par l'entraînement
(models/metrics.json), génère les deux matrices de confusion (labels
`positive` et `negative`) et assemble le rapport PDF final :
matrices, précision/rappel/F1 par classe, analyse et recommandations.

Usage :
    python report/generate_report.py
(nécessite un modèle entraîné : python -m src.train)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
METRICS_FILE = ROOT / "models" / "metrics.json"
FIGURES_DIR = ROOT / "report" / "figures"
PDF_FILE = ROOT / "report" / "rapport_evaluation.pdf"

# Teinte séquentielle unique (bleu) : l'intensité encode l'effectif.
CMAP = "Blues"
INK = "#1f2937"
MUTED = "#6b7280"


# ---------------------------------------------------------------- figures


def plot_confusion_matrix(cm, label: str, path: Path) -> None:
    """Matrice de confusion 2x2 annotée (teinte unique, axes discrets)."""
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=200)
    ax.imshow(cm, cmap=CMAP, vmin=0, vmax=max(max(row) for row in cm))

    names = [
        ["Vrais négatifs", "Faux positifs"],
        ["Faux négatifs", "Vrais positifs"],
    ]
    total = sum(sum(row) for row in cm)
    threshold = max(max(row) for row in cm) * 0.6
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i][j] >= threshold else INK
            ax.text(
                j, i - 0.08, str(cm[i][j]),
                ha="center", va="center", fontsize=20, fontweight="bold", color=color,
            )
            ax.text(
                j, i + 0.22,
                f"{names[i][j]}\n({cm[i][j] / total:.0%})",
                ha="center", va="center", fontsize=8.5, color=color,
            )

    ax.set_xticks([0, 1], ["0 (prédit non)", "1 (prédit oui)"], fontsize=9)
    ax.set_yticks([0, 1], ["0 (réel non)", "1 (réel oui)"], fontsize=9)
    ax.set_xlabel(f"Prédiction du label « {label} »", fontsize=10, color=MUTED)
    ax.set_ylabel("Valeur réelle", fontsize=10, color=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, colors=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- texte


def interpretation(label: str, m: dict) -> str:
    """Interprétation en français d'une matrice de confusion binaire."""
    (vn, fp), (fn, vp) = m["confusion_matrix"]
    cls = m["classes"]["1"]
    total = vn + fp + fn + vp
    # Formulation de la fiabilité indexée sur la valeur réelle de précision,
    # pour que le texte ne contredise jamais le chiffre.
    precision = cls["precision"]
    if precision >= 0.85:
        fiabilite = "il se trompe rarement"
    elif precision > 0:
        fiabilite = f"environ une prédiction sur {round(1 / (1 - precision))} est erronée"
    else:
        fiabilite = "aucune prédiction de ce label n'est correcte"
    return (
        f"Sur les {total} tweets du jeu de validation, le modèle identifie "
        f"correctement {vp} tweets {label}s (vrais positifs) et {vn} tweets "
        f"non {label}s (vrais négatifs), soit une exactitude globale de "
        f"{m['accuracy']:.1%}. Il commet {fp} faux positifs (tweets classés "
        f"« {label} » à tort) et {fn} faux négatifs (tweets « {label} » "
        f"manqués). La précision de la classe {label} est de "
        f"{cls['precision']:.1%} : lorsque le modèle prédit ce label, "
        f"{fiabilite}. Son rappel de {cls['recall']:.1%} indique la part "
        f"des tweets réellement {label}s qu'il parvient à retrouver ; le "
        f"F1-score de {cls['f1_score']:.1%} synthétise cet équilibre."
    )


# ---------------------------------------------------------------- PDF


class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 8, "SocialMetrics AI - Rapport d'évaluation du modèle",
                  align="R", new_x="LMARGIN", new_y="NEXT")

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def h1(self, text):
        self.set_font("helvetica", "B", 15)
        self.set_text_color(31, 41, 55)
        self.multi_cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def h2(self, text):
        self.ln(3)
        self.set_font("helvetica", "B", 12)
        self.set_text_color(31, 41, 55)
        self.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font("helvetica", "", 10)
        self.set_text_color(55, 65, 81)
        self.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def bullet(self, text):
        self.set_font("helvetica", "", 10)
        self.set_text_color(55, 65, 81)
        self.set_x(self.l_margin + 4)
        self.multi_cell(0, 5.5, "- " + text, new_x="LMARGIN", new_y="NEXT")

    def metrics_table(self, metrics):
        headers = ["Label", "Classe", "Précision", "Rappel", "F1-score", "Support"]
        widths = [32, 30, 30, 30, 30, 25]
        self.set_font("helvetica", "B", 9.5)
        self.set_fill_color(239, 246, 255)
        self.set_text_color(31, 41, 55)
        for h, w in zip(headers, widths):
            self.cell(w, 7, h, border=1, align="C", fill=True)
        self.ln()
        self.set_font("helvetica", "", 9.5)
        for label in ("positive", "negative"):
            for cls_key, cls_name in (("1", label), ("0", f"non {label}")):
                c = metrics[label]["classes"][cls_key]
                row = [
                    label, cls_name,
                    f"{c['precision']:.1%}", f"{c['recall']:.1%}",
                    f"{c['f1_score']:.1%}", str(c["support"]),
                ]
                for v, w in zip(row, widths):
                    self.cell(w, 7, v, border=1, align="C")
                self.ln()
        self.ln(3)


def build_pdf(metrics: dict) -> None:
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # --- Page de titre / contexte
    pdf.set_font("helvetica", "B", 20)
    pdf.set_text_color(31, 41, 55)
    pdf.ln(4)
    pdf.multi_cell(0, 10, "Rapport d'évaluation du modèle\nd'analyse de sentiments", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(107, 114, 128)
    trained_at = datetime.fromisoformat(metrics["trained_at"]).strftime("%d/%m/%Y")
    pdf.multi_cell(0, 6, f"SocialMetrics AI - client Daunale Treupe - {trained_at}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.h2("1. Contexte et méthodologie")
    pdf.body(
        "Le modèle évalué est une double régression logistique (scikit-learn, "
        "LogisticRegression) entraînée sur les tweets annotés de la table MySQL "
        "« tweets ». Chaque tweet est vectorisé par TF-IDF sur des n-grammes "
        "de caractères (2 à 5, bornés aux mots, accents normalisés), choisis "
        "par validation croisée : ils capturent la morphologie du français "
        "(« catastrophe » / « catastrophique ») et résistent aux fautes de "
        "frappe. Un premier classifieur prédit le label "
        "« positive », un second le label « negative » ; le score de sentiment "
        "renvoyé par l'API est la différence P(positif) - P(négatif), comprise "
        "entre -1 (très négatif) et 1 (très positif). Le seuil de décision de "
        "chaque classifieur est calibré par validation croisée sur le jeu "
        "d'entraînement (maximisation du F1-score de la classe 1"
        + (
            f" ; seuils retenus : {metrics['thresholds']['positive']:.2f} pour "
            f"« positive », {metrics['thresholds']['negative']:.2f} pour "
            "« negative »)"
            if "thresholds" in metrics
            else ")"
        )
        + ", afin de compenser le déséquilibre des classes."
    )
    pdf.body(
        f"Le dataset compte {metrics['dataset_size']} tweets annotés, séparés en "
        f"{metrics['train_size']} tweets d'entraînement (80 %) et "
        f"{metrics['validation_size']} tweets de validation (20 %), avec "
        "stratification sur le couple de labels afin de conserver la répartition "
        "des classes. Toutes les mesures ci-dessous sont calculées sur le jeu de "
        "validation, jamais vu pendant l'entraînement."
    )

    # --- Matrices de confusion
    for idx, label in enumerate(("positive", "negative"), start=2):
        pdf.add_page()
        pdf.h2(f"{idx}. Matrice de confusion - prédictions du label « {label} »")
        fig_path = FIGURES_DIR / f"confusion_{label}.png"
        pdf.image(str(fig_path), w=110, x=(210 - 110) / 2)
        pdf.ln(3)
        pdf.body(interpretation(label, metrics[label]))

    # --- Métriques
    pdf.add_page()
    pdf.h2("4. Précision, rappel et F1-score par classe")
    pdf.metrics_table(metrics)
    pdf.body(
        "La précision mesure la fiabilité des prédictions du modèle (part de "
        "vraies détections parmi les détections), le rappel sa couverture (part "
        "des cas réels retrouvés) et le F1-score leur moyenne harmonique. Les "
        "classes majoritaires (« non positive » et « non negative ») obtiennent "
        "logiquement les meilleurs scores : elles regroupent à la fois les tweets "
        "neutres et ceux de polarité opposée, plus nombreux et plus faciles à "
        "écarter. Les F1-scores des classes 1 restent élevés et équilibrés entre "
        "les deux labels, signe que le modèle ne favorise pas une polarité au "
        "détriment de l'autre."
    )

    # --- Analyse
    pdf.h2("5. Analyse des performances : forces, faiblesses et biais")
    pdf.set_font("helvetica", "B", 10.5)
    pdf.set_text_color(31, 41, 55)
    pdf.multi_cell(0, 6, "Forces", new_x="LMARGIN", new_y="NEXT")
    for point in (
        "Prédictions cohérentes sur le vocabulaire explicite : les tweets "
        "contenant des marqueurs clairs (« excellent », « catastrophique », "
        "« je recommande », « scandaleux ») sont classés de manière fiable.",
        "Bonne symétrie entre les deux labels : précision et rappel restent "
        "proches pour « positive » et « negative », le score final n'est donc "
        "pas biaisé vers une polarité.",
        "Les n-grammes de caractères capturent la morphologie française "
        "(« déçu » / « déception », « catastrophe » / « catastrophique ») et "
        "restent robustes aux fautes de frappe et à l'argot des tweets.",
    ):
        pdf.bullet(point)
    pdf.ln(2)
    pdf.set_font("helvetica", "B", 10.5)
    pdf.multi_cell(0, 6, "Faiblesses et biais identifiés", new_x="LMARGIN", new_y="NEXT")
    for point in (
        "Ironie et sarcasme : les tweets du type « Génial, encore une panne un "
        "lundi matin » emploient un lexique positif pour exprimer une opinion "
        "négative ; le sac de mots TF-IDF ne modélise pas ce renversement et "
        "produit des faux positifs.",
        "Négations : le sac de n-grammes ne modélise pas la portée d'une "
        "négation ; « je ne pensais pas que ce serait aussi bon » ou « pas "
        "terrible » peuvent être classés à contresens.",
        "Tweets mitigés (positifs ET négatifs, ex. « hôtel magnifique mais "
        "service désastreux ») : les deux classifieurs peuvent s'activer "
        "simultanément et le score proche de 0 masque la double polarité.",
        f"Biais de vocabulaire : le dataset ({metrics['dataset_size']} tweets) reste petit ; un mot "
        "absent de l'entraînement (argot, faute d'orthographe, emoji) est "
        "ignoré par le TF-IDF, ce qui pousse la prédiction vers la classe "
        "majoritaire (neutre).",
        "Déséquilibre de classes : environ un tiers seulement des tweets porte "
        "chaque label. La calibration des seuils de décision relève le rappel "
        "de la classe 1, mais au prix d'une précision moindre : le modèle "
        "détecte davantage de tweets polarisés en acceptant plus de fausses "
        "alertes.",
    ):
        pdf.bullet(point)
    pdf.ln(2)

    # --- Recommandations
    pdf.h2("6. Recommandations pour améliorer le modèle")
    for point in (
        "Agrandir le dataset : collecter et annoter en continu (plusieurs "
        "milliers de tweets) via l'API, en priorisant les cas difficiles "
        "(ironie, négations, tweets mitigés) - le réentraînement hebdomadaire "
        "en tirera automatiquement parti.",
        "Enrichir le prétraitement : gestion explicite des emojis et des "
        "élongations (« troooop bien »), lemmatisation française (spaCy), "
        "marquage des portées de négation (préfixer les mots suivant « ne... "
        "pas »).",
        "Passer à des représentations contextuelles : un modèle de type "
        "CamemBERT (fine-tuné ou utilisé comme encodeur devant la régression "
        "logistique) capture ironie et contexte bien mieux que le TF-IDF.",
        "Affiner la calibration : le rééquilibrage des classes "
        "(class_weight='balanced') et les seuils de décision par classe "
        "(calibrés par validation croisée, maximisation du F1) sont en place ; "
        "l'étape suivante est de choisir le compromis précision/rappel selon "
        "le besoin métier (courbes précision/rappel) et de généraliser "
        "l'évaluation finale par validation croisée plutôt que par un unique "
        "split.",
        "Suivre les performances dans le temps : historiser les métriques à "
        "chaque réentraînement hebdomadaire pour détecter toute dérive du "
        "modèle ou du vocabulaire des tweets.",
    ):
        pdf.bullet(point)

    pdf.output(str(PDF_FILE))
    print(f"Rapport généré : {PDF_FILE}")


def main() -> None:
    if not METRICS_FILE.exists():
        print(
            "models/metrics.json introuvable : entraînez d'abord le modèle "
            "(python -m src.train).",
            file=sys.stderr,
        )
        sys.exit(1)
    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for label in ("positive", "negative"):
        plot_confusion_matrix(
            metrics[label]["confusion_matrix"],
            label,
            FIGURES_DIR / f"confusion_{label}.png",
        )
    build_pdf(metrics)


if __name__ == "__main__":
    main()

