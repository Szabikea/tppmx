"""
Tippmix AI Segéd - Flask Application Factory
=============================================
Az alkalmazás inicializálása és konfigurálása.
"""

from flask import Flask
from .config import get_config
from .models import Database
from .scraper_client import FootballScraper
from .analytics import AnalyticsEngine


def create_app():
    """Flask alkalmazás factory"""
    
    # Flask app létrehozása
    app = Flask(__name__, 
                template_folder="../templates",
                static_folder="../static")
    
    # Konfiguráció betöltése
    config = get_config()
    app.config.from_object(config)
    
    # Database inicializálása
    db = Database(config.DATABASE_PATH)
    app.config["db"] = db
    
    # Scraper inicializálása (lecseréli az API client-et)
    scraper = FootballScraper(db)
    app.config["scraper"] = scraper
    
    # Analytics engine inicializálása
    analytics = AnalyticsEngine(config.ANALYSIS_MATCH_COUNT)
    app.config["analytics"] = analytics
    
    # Blueprint regisztrálása
    from .routes import main
    app.register_blueprint(main)
    
    # Jinja2 custom filters
    @app.template_filter("format_date")
    def format_date_filter(date_str):
        """Dátum formázása magyar stílusban"""
        if not date_str:
            return ""
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.strftime("%Y.%m.%d %H:%M")
        except:
            return date_str
    
    @app.template_filter("confidence_stars")
    def confidence_stars_filter(confidence):
        """Confidence érték csillagokká alakítása"""
        return "★" * confidence + "☆" * (5 - confidence)
    
    @app.template_filter("confidence_percent")
    def confidence_percent_filter(score):
        """Confidence score százalékos formázása"""
        if score is None:
            return "N/A"
        return f"{score:.0f}%"
    
    # Context processor - minden template-ben elérhető változók
    @app.context_processor
    def inject_globals():
        return {
            "app_name": "Tippmix AI Segéd",
            "app_version": "2.0.0"
        }
    
    return app
