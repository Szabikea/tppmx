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
from .gemini_analyzer import get_gemini_analyzer, GeminiAnalyzer


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
    
    # Extra biztonsági tipp: Over 1.5 gól, ha a modell sok gólt vár
    exp_total_goals = prediction.expected_home_goals + prediction.expected_away_goals
    if exp_total_goals >= 2.2: 
        tips.append({
            'bet_type': 'goals',
            'description': 'Gólok 1.5 felett',
            'probability': 82.0, # Konzervatív becslés
            'confidence': 5 if prediction.confidence_level == 'high' else (4 if prediction.confidence_level == 'medium' else 2),
            'odds_estimate': 1.25,
            'edge': 0,
            'is_value_bet': False, 
            'implied_prob': 80.0
        })

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
    """Liga tabella és csapatok"""
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
    
    target_league = league_name_map.get(league_id, league_id)
    all_fixtures = scraper.get_fixtures()
    
    # Csak az adott liga meccsei
    league_fixtures = [f for f in all_fixtures if f.get('league', '').lower() == target_league.lower()]
    
    # Tabella generálás "on-the-fly"
    table = {}
    
    for f in league_fixtures:
        home = f.get('home_team')
        away = f.get('away_team')
        
        if home and home not in table:
            table[home] = {'name': home, 'p': 0, 'w': 0, 'd': 0, 'l': 0, 'gf': 0, 'ga': 0, 'pts': 0}
        if away and away not in table:
            table[away] = {'name': away, 'p': 0, 'w': 0, 'd': 0, 'l': 0, 'gf': 0, 'ga': 0, 'pts': 0}
            
        # Ha van eredmény, számoljuk
        if f.get('status') == 'Finished' and f.get('home_score') is not None:
            h_score = int(f['home_score'])
            a_score = int(f['away_score'])
            
            table[home]['p'] += 1
            table[away]['p'] += 1
            table[home]['gf'] += h_score
            table[home]['ga'] += a_score
            table[away]['gf'] += a_score
            table[away]['ga'] += h_score
            
            if h_score > a_score:
                table[home]['w'] += 1; table[home]['pts'] += 3
                table[away]['l'] += 1
            elif a_score > h_score:
                table[away]['w'] += 1; table[away]['pts'] += 3
                table[home]['l'] += 1
            else:
                table[home]['d'] += 1; table[home]['pts'] += 1
                table[away]['d'] += 1; table[away]['pts'] += 1
    
    # Rendezés pontszám (majd gólkülönbség) szerint
    standings = sorted(table.values(), key=lambda x: (x['pts'], x['gf']-x['ga']), reverse=True)
    
    # Ha üres a tabella (nincs meccs a ligából az adatbázisban), akkor is mutassunk valamit ha tudunk
    # De most csak az adatbázisból dolgozunk
    
    scraper_status = scraper.get_status()
    
    return render_template(
        "league.html",
        league_name=target_league,
        standings=standings,
        scraper_status=scraper_status
    )


@main.route("/team/<path:team_name>")
def team_details(team_name: str):
    """Csapat részletes statisztikái"""
    scraper = get_scraper()
    all_fixtures = scraper.get_fixtures()
    
    # Csapat meccsei
    team_fixtures = [
        f for f in all_fixtures 
        if f.get('home_team') == team_name or f.get('away_team') == team_name
    ]
    
    # Rendezés dátum szerint (legutóbbi elől)
    team_fixtures.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Statisztikák számolása
    stats = {
        'played': 0,
        'wins': 0,
        'draws': 0,
        'losses': 0,
        'goals_scored': 0,
        'goals_conceded': 0,
        'form': []
    }
    
    for f in team_fixtures:
        if f.get('status') == 'Finished' and f.get('home_score') is not None:
            stats['played'] += 1
            h_score = int(f['home_score'])
            a_score = int(f['away_score'])
            
            is_home = f['home_team'] == team_name
            team_score = h_score if is_home else a_score
            opp_score = a_score if is_home else h_score
            
            stats['goals_scored'] += team_score
            stats['goals_conceded'] += opp_score
            
            if team_score > opp_score:
                stats['wins'] += 1
                stats['form'].append('W')
            elif team_score < opp_score:
                stats['losses'] += 1
                stats['form'].append('L')
            else:
                stats['draws'] += 1
                stats['form'].append('D')
    
    # Csak az utolsó 5 meccs formája
    stats['form'] = stats['form'][:5]
    
    return render_template(
        "team_details.html",
        team_name=team_name,
        fixtures=team_fixtures,
        stats=stats
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
            'explanation': f"Hazai átlag: {home_stats.avg_corners:.1f}, Vendég átlag: {away_stats.avg_corners:.1f}. Szórás: {corners['standard_deviation']:.1f} ({"alacsony - megbízható" if corners['standard_deviation'] < 2.5 else "magas - bizonytalan"})"
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
    # Poisson vs ML összehasonlítás
    combined_prediction = ml_predictor.compare_with_poisson(ml_prediction, prediction)
    
    # Gemini AI Elemzés
    gemini_analyzer = get_gemini_analyzer()
    gemini_analysis = gemini_analyzer.analyze_match(
        home_team, away_team, 
        home_stats, away_stats,
        prediction
    )
    
    # További magabiztos tippek hozzáadása (Gólok, BTTS)
    if combined_prediction.confident_secondary_tips:
        for ct in combined_prediction.confident_secondary_tips:
            # Ellenőrizzük, hogy ez a tipp már létezik-e
            existing = next((t for t in tips_with_explanation if t['description'] == ct.description), None)
            
            if existing:
                # Frissítjük a meglévő tippet
                existing['confidence'] = 5  # Max confidence
                existing['is_value_bet'] = True
                existing['explanation'] = f"✅ MAGABIZTOS TIPP! {ct.reasoning}"
            else:
                # Új tipp hozzáadása
                tips_with_explanation.insert(0, {
                    'bet_type': ct.tip_type,
                    'description': ct.description,
                    'probability': ct.probability,
                    'confidence': 5,
                    'odds_estimate': 100 / ct.probability, # Becsült odds
                    'edge': 10.0, # Feltételezett edge
                    'is_value_bet': True,
                    'explanation': f"✅ MAGABIZTOS TIPP! {ct.reasoning}"
                })
    
    # Professional Statistical Analysis
    stat_engine = get_stat_engine()
    full_stats = stat_engine.full_statistical_analysis(
        home_team, away_team, home_stats, away_stats, league
    )
    
    # Minden tipphez hozzáadjuk a részletes statisztikákat
    for tip in tips_with_explanation:
        # AI integráció: Használjuk a kombinált (ML + Poisson) valószínűségeket a számításokhoz
        prob_to_use = tip['probability']
        
        # 1. ML Boost (1X2)
        if combined_prediction and combined_prediction.combined_probs:
            if 'Hazai' in tip['description'] and 'győzelem' in tip['description']:
                prob_to_use = combined_prediction.combined_probs.get('1', prob_to_use)
            elif 'Döntetlen' in tip['description']:
                prob_to_use = combined_prediction.combined_probs.get('X', prob_to_use)
            elif 'Vendég' in tip['description'] and 'győzelem' in tip['description']:
                prob_to_use = combined_prediction.combined_probs.get('2', prob_to_use)
        
        # 2. Gemini AI Boost
        if gemini_analysis:
            boost_data = gemini_analyzer.get_tip_confidence_boost(
                tip.get('bet_type', ''),
                prob_to_use,
                gemini_analysis
            )
            if boost_data['ai_boost'] > 0:
                prob_to_use = boost_data['boosted_probability']
                tip['explanation'] += f" | 🤖 {boost_data['reasoning']}"
                tip['confidence'] = min(5, tip['confidence'] + 1)
        
        # Frissítjük a valószínűséget
        tip['probability'] = prob_to_use
        
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
        gemini_analysis=gemini_analysis,
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
# Tuti Mix Generátor
# =============================================================================

@main.route("/smart-mix")
def smart_mix():
    """Tuti Mix generátor oldal szűréssel"""
    scraper = get_scraper()
    analytics = get_analytics_engine()
    
    # 1. Paraméterek beolvasása
    try:
        threshold = int(request.args.get('threshold', 65)) # Alapértelmezett: 65%
    except ValueError:
        threshold = 65
        
    selected_leagues = request.args.getlist('leagues')
    
    # 2. Minden meccs lekérése
    fixtures = scraper.get_fixtures()
    
    # 3. Elérhető ligák kigyűjtése a szűrőhöz
    all_leagues = sorted(list(set([f.get('league', 'Egyéb') for f in fixtures if f.get('league')])))
    
    # 4. Szűrés: Csak jövőbeli/függő meccsek
    upcoming = [f for f in fixtures if f.get('status') != 'Finished']
    
    # 5. Liga szűrés alkalmazása
    if selected_leagues:
        upcoming = [f for f in upcoming if f.get('league') in selected_leagues]
    
    # Ha nincs elég meccs a szűrés után, de nincs szűrés, vegyük az összeset
    # De ha van szűrés, akkor tiszteljük a user döntését, legfeljebb üres lesz
    
    # Rendezés idő szerint
    upcoming.sort(key=lambda x: x.get('timestamp', '')) 
    
    # Max 50 meccsre növeljük a limitet, hogy legyen miből válogatni
    target_fixtures = upcoming[:50] 
    
    safe_items = []
    value_items = []
    all_items = []
    
    for f in target_fixtures:
        # Elemzés futtatása
        try:
            analysis = analytics.analyze_match(f)
            tips = generate_match_tips(f) # Ez egyszerűsített tippeket ad
            
            if not tips: continue
            
            # Legjobb tipp kiválasztása
            best_tip = tips[0]
            
            # 1. Minden meccs mix (ide minden bekerül ami a szűrésnek megfelelt)
            all_items.append({
                'home_team': f['home_team'],
                'away_team': f['away_team'],
                'match_time': f.get('match_time', 'N/A'),
                'league': f.get('league', ''),
                'tip': best_tip['description'],
                'odds': best_tip.get('odds_estimate', 1.5),
                'confidence': best_tip.get('confidence', 3)
            })
            
            # 2. Biztonsági mix (User által megadott küszöb felett)
            prob = best_tip.get('probability', 0)
            if prob >= threshold:
                # Rövid indoklás generálása
                reason = f"Valószínűség: {prob}%"
                if 'bet_type' in best_tip and best_tip['bet_type'] == 'corners':
                    reason = f"Szöglet trendek ({prob}%)"
                elif best_tip.get('is_value_bet'):
                    reason = f"Value bet ({prob}%)"
                    
                safe_items.append({
                    'home_team': f['home_team'],
                    'away_team': f['away_team'],
                    'league': f.get('league', ''),
                    'tip': best_tip['description'],
                    'odds': best_tip.get('odds_estimate', 1.5),
                    'confidence': best_tip.get('confidence', 0),
                    'reason': reason
                })
                
            # 3. Value mix (Edge > 5%)
            for t in tips:
                if t.get('is_value_bet') and t.get('edge', 0) > 5:
                    value_items.append({
                        'home_team': f['home_team'],
                        'away_team': f['away_team'],
                        'tip': t['description'],
                        'edge': t.get('edge', 0),
                        'odds': t.get('odds_estimate', 2.0),
                        'reason': f"Edge: {t.get('edge', 0):.1f}%"
                    })
                    break # Egy meccsből csak egyet
                    
        except Exception as e:
            print(f"Error analyzing fixture {f.get('home_team')} vs {f.get('away_team')}: {e}")
            continue
    
    # Összesítések számolása
    def calc_stats(items):
        total_odds = 1.0
        confs = []
        for i in items:
            total_odds *= float(i['odds'])
            if 'confidence' in i:
                confs.append(i['confidence'])
        
        avg_conf = sum(confs) / len(confs) if confs else 0
        return {
            'bets': items,
            'total_odds': round(total_odds, 2),
            'avg_confidence': round(avg_conf, 1)
        }

    match_date_str = target_fixtures[0].get('match_date', 'Mai nap') if target_fixtures else "Nincs adat"

    return render_template(
        "smart_mix.html",
        match_date=match_date_str,
        total_matches=len(target_fixtures),
        safe_mix=calc_stats(safe_items),
        all_mix=calc_stats(all_items),
        value_mix=calc_stats(value_items),
        all_leagues=all_leagues,
        selected_leagues=selected_leagues,
        threshold=threshold
    )


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
