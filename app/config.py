"""
Tippmix AI Segéd - Configuration Module
========================================
Környezeti változók és alkalmazás beállítások kezelése.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env fájl betöltése
load_dotenv()

# Alap útvonalak
BASE_DIR = Path(__file__).resolve().parent.parent
INSTANCE_DIR = BASE_DIR / "instance"
DATA_DIR = BASE_DIR / "data"

# Biztosítjuk, hogy a könyvtárak léteznek
INSTANCE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


class Config:
    """Alapértelmezett konfiguráció"""
    
    # Flask settings
    SECRET_KEY = os.getenv("SECRET_KEY", "tippmix-dev-secret-key-change-in-prod")
    
    # SQLite Database
    DATABASE_PATH = INSTANCE_DIR / "tippmix.db"
    
    # Scraped data storage
    SCRAPED_DATA_PATH = DATA_DIR / "scraped_data.json"
    
    # Scraping beállítások
    SCRAPE_INTERVAL_HOURS = 168  # Hetente egyszer (7 nap * 24 óra)
    SCRAPE_DELAY_SECONDS = 2  # Késleltetés kérések között
    
    # Céloldalak
    EREDMENYEK_BASE_URL = "https://www.eredmenyek.com"
    FOOTBALLDB_BASE_URL = "https://www.footballdatabase.eu"
    
    # Elemzéshez használt meccsek száma
    ANALYSIS_MATCH_COUNT = 10  # Utolsó 10 meccs
    
    # Cache TTL (time-to-live) órában (backup ha scraping nem elérhető)
    CACHE_TTL_HOURS = 168  # 1 hét
    
    # Támogatott ligák
    SUPPORTED_LEAGUES = {
        "premier-league": {"name": "Premier League", "country": "Anglia", "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿"},
        "la-liga": {"name": "La Liga", "country": "Spanyolország", "flag": "🇪🇸"},
        "serie-a": {"name": "Serie A", "country": "Olaszország", "flag": "🇮🇹"},
        "bundesliga": {"name": "Bundesliga", "country": "Németország", "flag": "🇩🇪"},
        "ligue-1": {"name": "Ligue 1", "country": "Franciaország", "flag": "🇫🇷"},
        "champions-league": {"name": "Champions League", "country": "Európa", "flag": "⭐"},
    }
    
    # Aktuális szezon
    CURRENT_SEASON = "2025-2026"


class DevelopmentConfig(Config):
    """Fejlesztői konfiguráció"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Produkciós konfiguráció"""
    DEBUG = False
    TESTING = False


# Környezet alapú konfiguráció választás
config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
}


def get_config():
    """Visszaadja az aktuális környezet konfigurációját"""
    env = os.getenv("FLASK_ENV", "development")
    return config_map.get(env, DevelopmentConfig)
