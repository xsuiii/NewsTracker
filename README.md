# NewsTracker

NewsTracker ist ein Tool zur Verfolgung und Analyse wichtiger globaler
Nachrichten. Es konzentriert sich auf seriöse Nachrichtenquellen und bietet
Funktionen wie Fake-News-Erkennung (BERT, LIAR-Datensatz) und Personalisierung
der Nachrichten basierend auf Benutzerpräferenzen.

## Projektstruktur
```
NewsTracker/
├── backend/
│   ├── main.py                 # Flask-App (/news, /classify, /personalized)
│   ├── api_connector.py        # NewsAPI-Abruf + BERT-Klassifikation
│   ├── fake_news_detection.py  # BERT-Fine-Tuning auf LIAR + Inferenz
│   ├── personalization.py      # Präferenz-basiertes Ranking
│   ├── db.py                   # SQLite-Persistierung
├── data/                       # Laufzeitdaten (DB, Präferenzen) – nicht im Git
├── models/                     # Trainierte Modelle (nicht im Git)
├── requirements.txt
└── README.md
```

## Einrichtung

```bash
git clone https://github.com/xsuiii/NewsTracker.git
cd NewsTracker
pip install -r requirements.txt
```

Umgebungsvariablen:

```bash
export NEWS_API_KEY="dein_newsapi_key"             # https://newsapi.org
export NEWSTRACKER_DB="data/newstracker.sqlite3"   # optional
```

## Verwendung

### 1. Modell trainieren (einmalig, ~30–60 min auf GPU)

```bash
python backend/fake_news_detection.py --mode train
```

Lädt den LIAR-Datensatz von HuggingFace, fine-tuned `bert-base-uncased` und
speichert das Modell unter `models/bert_fake_news/`.

### 2. Einzelne Aussage klassifizieren

```bash
python backend/fake_news_detection.py --mode predict \
    --text "The earth is flat and NASA is hiding it."
```

### 3. Aktuelle Nachrichten abrufen und klassifizieren

```bash
python backend/api_connector.py --query "finance" --limit 5
```

### 4. Personalisierung testen

```bash
python backend/personalization.py \
    --prefs '{"finance": 0.9, "politics": 0.7, "sports": 0.2}' --top 5
```

### 5. Flask-Server starten

```bash
python backend/main.py
```

Verfügbare Endpunkte:

| Methode | Pfad | Zweck |
|---|---|---|
| GET  | `/`                           | Health-Check |
| GET  | `/news?q=finance&limit=5`     | Abruf + Klassifikation + DB-Save |
| POST | `/classify`                   | Body `{"text": "..."}` → Fake/Echt |
| GET  | `/personalized?top=5`         | Top-N persistierte Artikel nach Präferenzen |

Beispiel:

```bash
curl -X POST http://127.0.0.1:5000/classify \
    -H "Content-Type: application/json" \
    -d '{"text": "Breaking: aliens land in Berlin."}'
```

## Datenbank

Standardmäßig SQLite (`data/newstracker.sqlite3`). Das Schema ist 1:1 auf
PostgreSQL übertragbar:

```sql
CREATE TABLE news (
    id                SERIAL PRIMARY KEY,
    title             TEXT NOT NULL,
    url               TEXT UNIQUE,
    category          TEXT,
    credibility_score FLOAT,
    content           TEXT,
    description       TEXT,
    source            TEXT,
    label             TEXT,
    fetched_at        TIMESTAMPTZ NOT NULL
);
```

## Beitragende
Erstellt von xsuiii.
