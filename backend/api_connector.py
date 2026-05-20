"""
API-Connector für NewsTracker.

Holt aktuelle Nachrichtenartikel über NewsAPI (https://newsapi.org) und
klassifiziert sie direkt mit dem in `fake_news_detection.py` definierten
BERT-basierten Klassifikator als "Fake" oder "Echt".

Voraussetzungen:
    - Umgebungsvariable NEWS_API_KEY mit einem gültigen NewsAPI-Key
    - Optional: trainiertes Modell unter models/bert_fake_news/
      (sonst wird das vortrainierte BERT ohne Fine-Tuning verwendet)

Verwendung:
    export NEWS_API_KEY="dein_key"
    python backend/api_connector.py --query "climate change" --limit 5
"""

import argparse
import os
from typing import Dict, List

import requests

from fake_news_detection import predict

# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #
NEWS_API_URL = "https://newsapi.org/v2/everything"
DEFAULT_QUERY = "technology"
DEFAULT_LIMIT = 5
REQUEST_TIMEOUT = 10  # Sekunden


# --------------------------------------------------------------------------- #
# News-Abruf
# --------------------------------------------------------------------------- #
def fetch_news(query: str, limit: int = DEFAULT_LIMIT) -> List[Dict[str, str]]:
    """Holt Nachrichtenartikel zu `query` über die NewsAPI.

    Gibt eine Liste von Artikeln mit den Feldern `title`, `description`,
    `url` und `source` zurück.
    """
    api_key = os.environ.get("NEWS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Umgebungsvariable NEWS_API_KEY ist nicht gesetzt. "
            "Einen kostenlosen Key gibt es unter https://newsapi.org."
        )

    params = {
        "q": query,
        "pageSize": limit,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": api_key,
    }

    response = requests.get(NEWS_API_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    articles = []
    for item in payload.get("articles", []):
        articles.append({
            "title": item.get("title") or "",
            "description": item.get("description") or "",
            "url": item.get("url") or "",
            "source": (item.get("source") or {}).get("name") or "",
        })
    return articles


# --------------------------------------------------------------------------- #
# Klassifikation
# --------------------------------------------------------------------------- #
def classify_articles(articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Klassifiziert jeden Artikel mit BERT als 'Fake' oder 'Echt'.

    Als Eingabetext wird `title` + `description` verwendet, da das die
    aussagekräftigste, frei verfügbare Textgrundlage ist.
    """
    # Texte für die Batch-Vorhersage zusammenbauen
    texts = [
        f"{a['title']}. {a['description']}".strip()
        for a in articles
    ]
    predictions = predict(texts)

    # Vorhersagen mit den Originalartikeln zusammenführen
    enriched = []
    for article, pred in zip(articles, predictions):
        enriched.append({
            **article,
            "label": pred["label"],
            "confidence": pred["confidence"],
        })
    return enriched


# --------------------------------------------------------------------------- #
# CLI-Einstiegspunkt
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Lädt Nachrichten via NewsAPI und klassifiziert sie mit BERT."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Suchbegriff.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="Maximale Anzahl Artikel.")
    args = parser.parse_args()

    articles = fetch_news(args.query, args.limit)
    if not articles:
        print(f"Keine Artikel für '{args.query}' gefunden.")
        return

    results = classify_articles(articles)

    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r['title']}")
        print(f"    Quelle:    {r['source']}")
        print(f"    URL:       {r['url']}")
        print(f"    Bewertung: {r['label']} (Konfidenz: {r['confidence']:.2%})")


if __name__ == "__main__":
    main()
