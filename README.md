# NewsTracker

NewsTracker ist ein Tool zur Verfolgung und Analyse wichtiger globaler Nachrichten. Es konzentriert sich auf seriöse Nachrichtenquellen und bietet Funktionen wie Fake-News-Erkennung und Personalisierung von Nachrichten basierend auf Benutzerpräferenzen.

## Projektstruktur
```
news-tracker/
│
├── backend/                    # Python-Backend
│   ├── main.py                 # Einstiegspunkt für den Server
│   ├── api_connector.py        # API-Integration mit NewsAPI/Bloomberg
│   ├── fake_news_detection.py  # Fake-News-Erkennungslogik
│   ├── utils.py                # Hilfsfunktionen
│
├── data/                       # Enthält Daten oder Modelle
│   ├── liar_dataset/           # Training Datensatz (Fake News Dataset)
│
├── frontend/                   # React.js Frontend (Optional für UIs, falls gewünscht)
│
├── models/                     # Enthält trainierte ML-Modelle oder Checkpoints
│
├── tests/                      # Unittests für Projektmodule
│
├── .gitignore                  # Ignoriert sensible Dateien
├── requirements.txt            # Python-Abhängigkeiten
├── README.md                   # Dokumentiert das Projekt und die Einrichtung
```

## Einrichtung
1. Klone das Repository:
   ```bash
   git clone https://github.com/xsuiii/NewsTracker.git
   ```

2. Installiere die Abhängigkeiten:
   ```bash
   pip install -r requirements.txt
   ```

3. Starte die Anwendung:
   ```bash
   python backend/main.py
   ```

## Beitragende
Erstellt von xsuiii.