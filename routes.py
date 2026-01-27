"""
Tippmix AI Segéd - Flask Routes
===============================
Web alkalmazás útvonalak és view-k (Advanced Analytics verzió).
"""

from flask import Blueprint, render_template, jsonify, request, current_app
from datetime import datetime
import random
import hashlib

from .scraper_client import FootballScraper
from .analytics import AnalyticsEngine
from .models import Database
from .config import get_config
from .advanced_analytics import get_analytics_engine, AdvancedAnalytics
from .ml_predictor import get_ml_predictor, MLPredictor
from .stat_engine import get_stat_engine, ProfessionalStatEngine


def generate_match_tips(fixture: dict) -> list:
    """
    Professzionális tippek generálása Poisson-modellezéssel
    Value Bet detektálás és konfidencia intervallum számítással
    """
    analytics = get_analytics_engine()
    league = fixture.get('league', 'Premier League')
    
    # Teljes elemzés futtatása
    analysis = analytics.analyze_match(fixture)
    
    tips = []
    prediction = analysis['prediction']
    value_bets = analysis['value_bets']
    
    # Value Bet-ek konvertálása tippekké
    for vb in value_bets:
        tip = {
            'bet_type': vb.bet_type,
            'description': vb.description,
            'probability': vb.our_probability,
            'confidence': vb.confidence,
            'odds_estimate': vb.recommended_odds,
            'edge': vb.edge,
            'is_value_bet': vb.is_value,
            'implied_prob': vb.implied_odds_prob
        }
        tips.append(tip)
    
    # Szöglet tipp hozzáadása
    corners = analysis['corners']
    if corners.get('best_bet'):
        cb = corners['best_bet']
        tips.append({
            'bet_type': 'corners',
            'description': f"Szögletek {cb['direction']} {cb['line']}",
            'probability': cb['probability'],
            'confidence': corners['reliability_score'],
            'odds_estimate': round(100 / cb['probability'], 2),
            'edge': cb['probability'] - 50,
            'is_value_bet': cb['probability'] > 60 or cb['probability'] < 40,
            'implied_prob': 50.0,
            'std_deviation': corners['standard_deviation']
        })
    
    # Lap tipp ha van szignál
    cards = analysis['cards']
    if cards.get('over_signal'):
        tips.append({
            'bet_type': 'cards',
            'description': cards['recommendation'] or 'Lapok over ajánlott',
            'probability': cards['over_probs'].get('over_4.5', 55),
            'confidence': 4 if cards['over_signal'] else 3,
            'odds_estimate': 1.85,
            'edge': cards['expected_total'] - 4.5,
            'is_value_bet': True,
            'implied_prob': 54.0
        })
    
    # Konfidencia intervallum és szint hozzáadása minden tipphez 
    for tip in tips:
        tip['confidence_level'] = prediction.confidence_level
        tip['confidence_interval'] = prediction.confidence_interval
    
    # Rendezés: Valószínűség szerint csökkenő sorrendben
    tips.sort(key=lambda x: x.get('probability', 0), reverse=True)
    
    return tips


def get_full_match_analysis(fixture: dict) -> dict:
    """Teljes meccs elemzés részletes adatokkal"""
    analytics = get_analytics_engine()
    return analytics.analyze_match(fixture)


# Blueprint létrehozása
main = Blueprint("main", __name__)



def get_db() -> Database:
    """Database instance lekérése"""
    return current_app.config["db"]


def get_scraper() -> FootballScraper:
    """Scraper instance lekérése"""
    return current_app.config["scraper"]


def get_analytics() -> AnalyticsEngine:
    """Analytics engine instance lekérése"""
    return current_app.config["analytics"]


# =============================================================================
# Főoldal
# =============================================================================

@main.route("/")
def index():
    """Főoldal - Liga választó és scraping státusz"""
    scraper = get_scraper()
    config = get_config()
    
    # Támogatott ligák a config-ból
    leagues = config.SUPPORTED_LEAGUES
    
    # Scraping státusz
    scraper_status = scraper.get_status()
    
    return render_template(
        "index.html",
        leagues=leagues,
        scraper_status=scraper_status,
        current_season=config.CURRENT_SEASON
    )


# =============================================================================
# Liga oldal
# =============================================================================

@main.route("/league/<league_id>")
def league(league_id: str):
    """Liga meccsek és elemzések"""
    scraper = get_scraper()
    config = get_config()
    
    # Liga info
    league_info = config.SUPPORTED_LEAGUES.get(league_id, {
        "name": f"Liga: {league_id}",
        "country": "Ismeretlen",
        "flag": "🏆"
    })
    
    # Liga név konvertálás a szűréshez
    league_name_map = {
        'premier-league': 'Premier League',
        'la-liga': 'La Liga',
        'bundesliga': 'Bundesliga',
        'serie-a': 'Serie A',
        'ligue-1': 'Ligue 1',
        'champions-league': 'Champions League'
    }
    
    target_league = league_name_map.get(league_id, league_id)
    
    # Meccsek lekérése a cache-ből
    all_fixtures = scraper.get_fixtures()
    
    # Szűrés liga alapján - pontos egyezés
    fixtures = [
        f for f in all_fixtures 
        if f.get('league', '').lower() == target_league.lower()
    ][:20]  # Max 20 meccs
    
    # Tippek generálása minden meccshez
    for fixture in fixtures:
        fixture['tips'] = generate_match_tips(fixture)
    
    # Scraping státusz
    scraper_status = scraper.get_status()
    
    return render_template(
        "league.html",
        league_id=league_id,
        league_info=league_info,
        fixtures=fixtures,
        scraper_status=scraper_status
    )


# =============================================================================
# Következő meccsek (összes verseny)
# =============================================================================

@main.route("/upcoming")
@main.route("/upcoming/<league_filter>")
def upcoming_matches(league_filter=None):
    """Következő meccsek - liga szűréssel"""
    scraper = get_scraper()
    config = get_config()
    
    # Liga név konvertálás
    league_name_map = {
        'premier-league': 'Premier League',
        'la-liga': 'La Liga',
        'bundesliga': 'Bundesliga',
        'serie-a': 'Serie A',
        'ligue-1': 'Ligue 1',
        'champions-league': 'Champions League'
    }
    
    # Meccsek lekérése cache-ből
    all_fixtures = scraper.get_fixtures(limit=120)
    
    # Szűrés liga alapján ha van
    if league_filter and league_filter in league_name_map:
        target_league = league_name_map[league_filter]
        fixtures = [
            f for f in all_fixtures
            if f.get('league', '').lower() == target_league.lower()
        ][:20]
    else:
        fixtures = all_fixtures[:30]
    
    # Tippek generálása minden meccshez
    for fixture in fixtures:
        fixture['tips'] = generate_match_tips(fixture)
    
    # Scraping státusz
    scraper_status = scraper.get_status()
    
    return render_template(
        "upcoming.html",
        fixtures=fixtures,
        scraper_status=scraper_status,
        current_league=league_filter,
        leagues=config.SUPPORTED_LEAGUES
    )


# =============================================================================
# Meccs elemzés
# =============================================================================

@main.route("/match/<path:match_id>")
def match_analysis(match_id: str):
    """Részletes meccs elemzés - teljes statisztikákkal"""
    scraper = get_scraper()
    analytics = get_analytics_engine()
    config = get_config()
    
    # Meccs keresése
    fixtures = scraper.get_fixtures()
    fixture = None
    
    # Keresés ID vagy csapatnevek alapján
    for f in fixtures:
        fid = str(f.get('id', ''))
        if fid == match_id:
            fixture = f
            break
        # Csapatnév alapú keresés
        slug = f"{f.get('home_team', '')}-vs-{f.get('away_team', '')}".lower().replace(' ', '-')
        if match_id.lower() == slug:
            fixture = f
            break
    
    if not fixture:
        return render_template("error.html", 
                              error_code=404, 
                              error_message="Meccs nem található"), 404
    
    home_team = fixture.get('home_team', 'Hazai')
    away_team = fixture.get('away_team', 'Vendég')
    league = fixture.get('league', 'Premier League')
    
    # Teljes advanced analytics elemzés
    full_analysis = analytics.analyze_match(fixture)
    
    # Csapat statisztikák kinyerése
    home_stats = full_analysis['home_stats']
    away_stats = full_analysis['away_stats']
    prediction = full_analysis['prediction']
    value_bets = full_analysis['value_bets']
    corners = full_analysis['corners']
    cards = full_analysis['cards']
    
    # Tippek generálása magyarázatokkal
    tips_with_explanation = []
    
    for vb in value_bets:
        tip = {
            'bet_type': vb.bet_type,
            'description': vb.description,
            'probability': vb.our_probability,
            'confidence': vb.confidence,
            'odds_estimate': vb.recommended_odds,
            'edge': vb.edge,
            'is_value_bet': vb.is_value,
            'implied_prob': vb.implied_odds_prob,
            'explanation': _generate_tip_explanation(vb, home_team, away_team, home_stats, away_stats)
        }
        tips_with_explanation.append(tip)
    
    # Szöglet tipp
    if corners.get('best_bet'):
        cb = corners['best_bet']
        tips_with_explanation.append({
            'bet_type': 'corners',
            'description': f"Szögletek {cb['direction']} {cb['line']}",
            'probability': cb['probability'],
            'confidence': corners['reliability_score'],
            'odds_estimate': round(100 / cb['probability'], 2),
            'edge': cb['probability'] - 50,
            'is_value_bet': cb['probability'] > 60 or cb['probability'] < 40,
            'explanation': f"Hazai átlag: {home_stats.avg_corners:.1f}, Vendég átlag: {away_stats.avg_corners:.1f}. Szórás: {corners['standard_deviation']:.1f} ({'alacsony - megbízható' if corners['standard_deviation'] < 2.5 else 'magas - bizonytalan'})"
        })
    
    # Lapok tipp
    if cards.get('over_signal'):
        tips_with_explanation.append({
            'bet_type': 'cards',
            'description': cards['recommendation'] or 'Lapok over ajánlott',
            'probability': cards['over_probs'].get('over_4.5', 55),
            'confidence': 4,
            'odds_estimate': 1.85,
            'edge': cards['expected_total'] - 4.5,
            'is_value_bet': True,
            'explanation': f"Hazai lapok átlag: {home_stats.avg_cards:.1f}, Vendég: {away_stats.avg_cards:.1f}. Várható: {cards['expected_total']:.1f} lap"
        })
    
    # Rendezés: Valószínűség szerint csökkenő sorrendben
    tips_with_explanation.sort(key=lambda x: x.get('probability', 0), reverse=True)
    
    # ML Prediction
    ml_predictor = get_ml_predictor()
    ml_prediction = ml_predictor.predict(home_stats, away_stats, home_team, away_team)
    
    # Poisson vs ML összehasonlítás
    combined_prediction = ml_predictor.compare_with_poisson(ml_prediction, prediction)
    
    # Professional Statistical Analysis
    stat_engine = get_stat_engine()
    full_stats = stat_engine.full_statistical_analysis(
        home_team, away_team, home_stats, away_stats, league
    )
    
    # Minden tipphez hozzáadjuk a részletes statisztikákat
    for tip in tips_with_explanation:
        # AI integráció: Használjuk a kombinált (ML + Poisson) valószínűségeket a számításokhoz
        # Így a 'motor' ténylegesen használja az AI-t is
        prob_to_use = tip['probability']
        
        if combined_prediction and combined_prediction.combined_probs:
            if 'Hazai' in tip['description'] and 'győzelem' in tip['description']:
                prob_to_use = combined_prediction.combined_probs.get('1', prob_to_use)
                # Frissítjük a tipp valószínűségét is a UI-hoz
                tip['probability'] = prob_to_use
                tip['explanation'] += " (AI-val korrigálva)"
            elif 'Döntetlen' in tip['description']:
                prob_to_use = combined_prediction.combined_probs.get('X', prob_to_use)
                tip['probability'] = prob_to_use
                tip['explanation'] += " (AI-val korrigálva)"
            elif 'Vendég' in tip['description'] and 'győzelem' in tip['description']:
                prob_to_use = combined_prediction.combined_probs.get('2', prob_to_use)
                tip['probability'] = prob_to_use
                tip['explanation'] += " (AI-val korrigálva)"
        
        bet_stats = stat_engine.calculate_bet_statistics(
            tip['description'],
            prob_to_use,
            tip.get('odds_estimate'),
            home_stats,
            away_stats
        )
        tip['fair_odds'] = bet_stats.fair_odds
        tip['ev'] = bet_stats.expected_value
        tip['kelly'] = bet_stats.kelly_fraction
        tip['ci'] = bet_stats.confidence_interval
        tip['std'] = bet_stats.standard_deviation
        tip['mc_prob'] = bet_stats.monte_carlo_prob
        tip['reliability'] = bet_stats.reliability_score
        tip['bet_rating'] = bet_stats.bet_rating
        tip['z_score'] = bet_stats.z_score
    
    # Scraping státusz
    scraper_status = scraper.get_status()
    
    return render_template(
        "analysis.html",
        fixture=fixture,
        home_team=home_team,
        away_team=away_team,
        league=league,
        home_stats=home_stats,
        away_stats=away_stats,
        prediction=prediction,
        tips=tips_with_explanation,
        corners=corners,
        cards=cards,
        confidence_level=prediction.confidence_level,
        confidence_interval=prediction.confidence_interval,
        ml_prediction=ml_prediction,
        combined=combined_prediction,
        full_stats=full_stats,
        scraper_status=scraper_status
    )


def _generate_tip_explanation(vb, home_team, away_team, home_stats, away_stats) -> str:
    """Tipp magyarázat generálása"""
    if 'Hazai' in vb.description or home_team in vb.description:
        if vb.is_value:
            return f"{home_team} támadóereje ({home_stats.attack_strength:.2f}x) magasabb a liga átlagnál. Forma-index: {home_stats.form_index:.0f}/100. A piaci odds alábecsüli az esélyeiket."
        else:
            return f"{home_team} támadóereje: {home_stats.attack_strength:.2f}x, védekezés: {home_stats.defense_strength:.2f}x. Hazai pálya előny 25%-kal növeli az esélyeket."
    
    elif 'Vendég' in vb.description or away_team in vb.description:
        if vb.is_value:
            return f"{away_team} védekezése ({away_stats.defense_strength:.2f}x) és támadása ({away_stats.attack_strength:.2f}x) jobb mint amit a piac áraz."
        else:
            return f"{away_team} idegenben 10%-kal gyengébb. Támadóerő: {away_stats.attack_strength:.2f}x, forma: {away_stats.form_index:.0f}/100."
    
    elif 'Döntetlen' in vb.description:
        return f"Mindkét csapat hasonló erősségű (hazai: {home_stats.attack_strength:.2f}x vs vendég: {away_stats.attack_strength:.2f}x). Poisson-modell {vb.our_probability:.1f}% döntetlent prediktál."
    
    elif 'gól' in vb.description.lower():
        if 'felett' in vb.description.lower():
            return f"Várható gólok: {home_stats.avg_goals_scored:.1f} + {away_stats.avg_goals_scored:.1f} = {home_stats.avg_goals_scored + away_stats.avg_goals_scored:.1f}. A Poisson-modell {vb.our_probability:.1f}% esélyt ad 2.5 feletti gólokra."
        else:
            return f"Mindkét csapat képes gólt szerezni. Hazai gólátlag: {home_stats.avg_goals_scored:.1f}, vendég: {away_stats.avg_goals_scored:.1f}."
    
    return "Matematikai modell alapján számított valószínűség."


# =============================================================================
# Statisztikák oldal
# =============================================================================

@main.route("/stats")
def stats():
    """Scraping és cache statisztikák"""
    scraper = get_scraper()
    db = get_db()
    
    scraper_status = scraper.get_status()
    cache_stats = db.get_cache_stats()
    
    return render_template(
        "stats.html",
        scraper_status=scraper_status,
        cache_stats=cache_stats
    )


# =============================================================================
# API Endpoints (AJAX)
# =============================================================================

@main.route("/api/status")
def api_status():
    """Scraper státusz JSON formátumban"""
    scraper = get_scraper()
    return jsonify(scraper.get_status())


@main.route("/api/refresh-data", methods=["POST"])
def refresh_data():
    """Manuális adatfrissítés indítása"""
    scraper = get_scraper()
    
    try:
        result = scraper.refresh_data_sync()
        
        return jsonify({
            "success": result.get('success', False),
            "message": f"Sikeresen frissítve! {result.get('fixtures_count', 0)} meccs találva.",
            "fixtures_count": result.get('fixtures_count', 0),
            "teams_count": result.get('teams_count', 0)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Hiba történt: {str(e)}"
        }), 500


@main.route("/api/quick-analysis/<home_team>/<away_team>")
def quick_analysis(home_team: str, away_team: str):
    """Gyors elemzés JSON formátumban (AJAX hívásokhoz)"""
    return jsonify({
        "analysis": f"{home_team} vs {away_team}",
        "tips": [],
        "confidence_score": 50
    })


@main.route("/api/refresh-cache", methods=["POST"])
def refresh_cache():
    """Cache törlése (lejárt bejegyzések)"""
    db = get_db()
    deleted = db.clear_expired_cache()
    
    return jsonify({
        "success": True,
        "message": f"{deleted} lejárt cache bejegyzés törölve"
    })


# =============================================================================
# Error Handlers
# =============================================================================

@main.errorhandler(404)
def not_found(error):
    """404 hiba oldal"""
    return render_template("error.html", 
                          error_code=404, 
                          error_message="Az oldal nem található"), 404


@main.errorhandler(500)
def server_error(error):
    """500 hiba oldal"""
    return render_template("error.html",
                          error_code=500,
                          error_message="Szerver hiba történt"), 500
