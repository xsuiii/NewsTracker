"""
NewsTracker-Backend (Flask).

Endpunkte:
    GET  /                  Health-Check
    GET  /news?q=&limit=    NewsAPI-Abruf + BERT-Klassifikation + DB-Persistierung
    POST /classify          Body: {"text": "..."} -> Fake/Echt mit Konfidenz
    GET  /personalized      Liefert Top-N persistierte Artikel gemäß Präferenzen
"""

from flask import Flask, jsonify, request

from api_connector import classify_articles, fetch_news
from db import list_articles, save_articles
from fake_news_detection import predict
from personalization import load_preferences, top_n

app = Flask(__name__)


@app.route("/")
def home():
    return "Welcome to the NewsTracker API!"


@app.route("/news", methods=["GET"])
def news():
    """Holt Artikel zu `q`, klassifiziert sie und speichert sie in der DB."""
    query = request.args.get("q", "technology")
    limit = int(request.args.get("limit", 5))

    articles = fetch_news(query, limit)
    classified = classify_articles(articles)
    # Die Suchanfrage dient hier als grobe Kategorie
    save_articles(classified, category=query.lower())
    return jsonify({"query": query, "count": len(classified), "articles": classified})


@app.route("/classify", methods=["POST"])
def classify():
    """Klassifiziert einen einzelnen Text."""
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "").strip()
    if not text:
        return jsonify({"error": "Feld 'text' fehlt"}), 400
    return jsonify(predict([text])[0])


@app.route("/personalized", methods=["GET"])
def personalized():
    """Top-N persistierte Artikel nach Nutzerpräferenzen."""
    n = int(request.args.get("top", 5))
    prefs = load_preferences()
    # Aus DB alle aktuellen Artikel laden und im Speicher sortieren
    articles = list_articles(limit=200)
    return jsonify({
        "preferences": prefs,
        "articles": top_n(articles, n=n, preferences=prefs),
    })


if __name__ == "__main__":
    app.run(debug=True)
