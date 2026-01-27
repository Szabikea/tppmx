# Tippmix AI Segéd

Megalapozott fogadási döntések API-Football adatok alapján.

## Funkciók

- **Statisztikai elemzés**: Szögletek, sárga lapok, xG az utolsó 5-10 meccs alapján
- **Ligák**: Premier League, La Liga, Serie A, Bundesliga, Ligue 1, NB1
- **Smart caching**: SQLite cache, max 100 API hívás/nap
- **Dark theme**: Mobilbarát glassmorphism design

## Telepítés

```bash
pip install flask python-dotenv requests
```

## API Kulcs

1. Regisztrálj: https://rapidapi.com/api-sports/api/api-football
2. Hozz létre `.env` fájlt:

```
RAPIDAPI_KEY=your_api_key_here
```

## Indítás

```bash
python run.py
```

Böngészőben: http://127.0.0.1:5000

## Tech Stack

- **Backend**: Python 3, Flask
- **Frontend**: HTML5, Jinja2, Bootstrap 5
- **Adatbázis**: SQLite (cache)
- **API**: API-Football (RapidAPI)
