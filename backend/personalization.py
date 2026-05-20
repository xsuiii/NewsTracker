"""
Personalisierung der Nachrichten-Reihenfolge.

Priorisiert eine Liste von Artikeln anhand von Nutzer-Präferenzgewichten
pro Kategorie (z. B. {"finance": 0.9, "politics": 0.4}) und der
Glaubwürdigkeit aus der BERT-Klassifikation.

Score-Formel pro Artikel:
    score = preference[category] * credibility
mit credibility = confidence, wenn label == "Echt", sonst (1 - confidence).

Verwendung:
    # Demo mit Beispielartikeln und Default-Präferenzen
    python backend/personalization.py
"""

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional

# Default-Präferenzen, falls keine Datei vorhanden
DEFAULT_PREFERENCES: Dict[str, float] = {
    "finance":      0.9,
    "politics":     0.7,
    "technology":   0.5,
    "sports":       0.2,
    "entertainment": 0.1,
}

DEFAULT_PREF_FILE = os.path.join(
    os.path.dirname(__file__), "..", "data", "preferences.json"
)


# --------------------------------------------------------------------------- #
# Präferenzen laden / speichern
# --------------------------------------------------------------------------- #
def load_preferences(path: str = DEFAULT_PREF_FILE) -> Dict[str, float]:
    """Lädt Präferenzen aus JSON; fällt auf Defaults zurück."""
    if not os.path.isfile(path):
        return dict(DEFAULT_PREFERENCES)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_preferences(prefs: Dict[str, float], path: str = DEFAULT_PREF_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def _credibility(article: Dict) -> float:
    """Glaubwürdigkeit aus Label + Konfidenz ableiten (0..1)."""
    label = (article.get("label") or "").lower()
    conf = float(article.get("confidence", 0.5))
    return conf if label in ("echt", "real", "true") else 1.0 - conf


def score_articles(
    articles: Iterable[Dict],
    preferences: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """Berechnet einen Score je Artikel und gibt sie absteigend sortiert zurück.

    Artikel ohne bekannte Kategorie erhalten ein neutrales Gewicht von 0.3,
    damit sie nicht komplett verschwinden.
    """
    prefs = preferences if preferences is not None else load_preferences()
    scored: List[Dict] = []
    for a in articles:
        category = (a.get("category") or "").lower()
        weight = prefs.get(category, 0.3)
        score = weight * _credibility(a)
        scored.append({**a, "score": score})
    return sorted(scored, key=lambda x: x["score"], reverse=True)


def top_n(
    articles: Iterable[Dict],
    n: int = 5,
    preferences: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    return score_articles(articles, preferences)[:n]


# --------------------------------------------------------------------------- #
# CLI-Demo
# --------------------------------------------------------------------------- #
SAMPLE_ARTICLES: List[Dict] = [
    {"title": "Fed signals rate cut", "category": "finance",
     "label": "Echt", "confidence": 0.91},
    {"title": "Celebrity gossip explodes", "category": "entertainment",
     "label": "Fake", "confidence": 0.80},
    {"title": "New chip beats benchmarks", "category": "technology",
     "label": "Echt", "confidence": 0.74},
    {"title": "Senate passes budget bill", "category": "politics",
     "label": "Echt", "confidence": 0.88},
    {"title": "Match-fixing rumors", "category": "sports",
     "label": "Fake", "confidence": 0.65},
]


def main():
    parser = argparse.ArgumentParser(
        description="Personalisierte Nachrichten-Reihenfolge auf Basis von Präferenzen."
    )
    parser.add_argument(
        "--prefs",
        type=str,
        default=None,
        help="JSON-String mit Präferenzen, z. B. '{\"finance\": 0.9}'.",
    )
    parser.add_argument("--top", type=int, default=5, help="Anzahl Top-Artikel.")
    args = parser.parse_args()

    prefs = json.loads(args.prefs) if args.prefs else load_preferences()
    print("Verwendete Präferenzen:", prefs)

    ranked = top_n(SAMPLE_ARTICLES, n=args.top, preferences=prefs)
    print(f"\nTop {len(ranked)} Artikel:")
    for i, a in enumerate(ranked, 1):
        print(f"  [{i}] {a['title']}  "
              f"(Kategorie: {a['category']}, Label: {a['label']}, "
              f"Score: {a['score']:.3f})")


if __name__ == "__main__":
    main()
