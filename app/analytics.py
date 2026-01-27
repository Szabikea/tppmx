"""
Tippmix AI Segéd - Analytics Engine
====================================
Fogadási elemző algoritmusok: szögletek, sárga lapok, xG számítások.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class BetType(Enum):
    """Fogadási típusok"""
    CORNERS_OVER = "corners_over"
    CORNERS_UNDER = "corners_under"
    CARDS_OVER = "cards_over"
    CARDS_UNDER = "cards_under"
    GOALS_OVER = "goals_over"
    GOALS_UNDER = "goals_under"
    BTTS_YES = "btts_yes"  # Both Teams To Score
    BTTS_NO = "btts_no"
    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"


@dataclass
class BettingTip:
    """Fogadási tipp adatstruktúra"""
    bet_type: BetType
    description: str
    line: float  # pl. 9.5 szöglet, 2.5 gól
    confidence: int  # 1-5 csillag
    reasoning: str
    probability: float  # 0-100%
    
    def to_dict(self) -> dict:
        return {
            "bet_type": self.bet_type.value,
            "description": self.description,
            "line": self.line,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "probability": round(self.probability, 1)
        }


class AnalyticsEngine:
    """
    Fogadási elemző motor.
    
    Az utolsó N meccs statisztikái alapján számol átlagokat
    és generál megalapozott fogadási tippeket.
    """
    
    def __init__(self, analysis_match_count: int = 5):
        self.match_count = analysis_match_count
    
    # =========================================================================
    # Szöglet Elemzés
    # =========================================================================
    
    def calculate_corner_stats(self, fixtures: List[dict], 
                                team_id: int) -> Dict[str, float]:
        """
        Szöglet statisztikák számítása egy csapat meccsei alapján.
        
        Args:
            fixtures: Meccsek listája statisztikákkal
            team_id: Csapat azonosító
            
        Returns:
            Szöglet átlagok és valószínűségek
        """
        corners_for = []
        corners_against = []
        total_corners = []
        
        for fixture in fixtures[:self.match_count]:
            stats = self._extract_fixture_stats(fixture, team_id)
            if stats:
                corners_for.append(stats.get("corners_for", 0))
                corners_against.append(stats.get("corners_against", 0))
                total_corners.append(stats.get("corners_for", 0) + stats.get("corners_against", 0))
        
        if not corners_for:
            return {"error": "Nincs elegendő adat"}
        
        avg_for = sum(corners_for) / len(corners_for)
        avg_against = sum(corners_against) / len(corners_against)
        avg_total = sum(total_corners) / len(total_corners)
        
        # Over/Under valószínűségek számítása
        over_9_5 = sum(1 for c in total_corners if c > 9.5) / len(total_corners) * 100
        over_10_5 = sum(1 for c in total_corners if c > 10.5) / len(total_corners) * 100
        over_11_5 = sum(1 for c in total_corners if c > 11.5) / len(total_corners) * 100
        
        return {
            "avg_corners_for": round(avg_for, 2),
            "avg_corners_against": round(avg_against, 2),
            "avg_total_corners": round(avg_total, 2),
            "over_9_5_probability": round(over_9_5, 1),
            "over_10_5_probability": round(over_10_5, 1),
            "over_11_5_probability": round(over_11_5, 1),
            "matches_analyzed": len(corners_for)
        }
    
    # =========================================================================
    # Sárga Lap Elemzés
    # =========================================================================
    
    def calculate_card_stats(self, fixtures: List[dict], 
                              team_id: int) -> Dict[str, float]:
        """
        Sárga lap statisztikák számítása.
        
        Args:
            fixtures: Meccsek listája
            team_id: Csapat azonosító
            
        Returns:
            Lap átlagok és valószínűségek
        """
        cards_for = []
        cards_against = []
        total_cards = []
        
        for fixture in fixtures[:self.match_count]:
            stats = self._extract_fixture_stats(fixture, team_id)
            if stats:
                cards_for.append(stats.get("yellow_cards_for", 0))
                cards_against.append(stats.get("yellow_cards_against", 0))
                total_cards.append(
                    stats.get("yellow_cards_for", 0) + 
                    stats.get("yellow_cards_against", 0)
                )
        
        if not cards_for:
            return {"error": "Nincs elegendő adat"}
        
        avg_for = sum(cards_for) / len(cards_for)
        avg_against = sum(cards_against) / len(cards_against)
        avg_total = sum(total_cards) / len(total_cards)
        
        # Over/Under valószínűségek
        over_3_5 = sum(1 for c in total_cards if c > 3.5) / len(total_cards) * 100
        over_4_5 = sum(1 for c in total_cards if c > 4.5) / len(total_cards) * 100
        over_5_5 = sum(1 for c in total_cards if c > 5.5) / len(total_cards) * 100
        
        return {
            "avg_cards_for": round(avg_for, 2),
            "avg_cards_against": round(avg_against, 2),
            "avg_total_cards": round(avg_total, 2),
            "over_3_5_probability": round(over_3_5, 1),
            "over_4_5_probability": round(over_4_5, 1),
            "over_5_5_probability": round(over_5_5, 1),
            "matches_analyzed": len(cards_for)
        }
    
    # =========================================================================
    # Várható Gólok (xG) Becslés
    # =========================================================================
    
    def estimate_expected_goals(self, home_fixtures: List[dict],
                                 away_fixtures: List[dict],
                                 home_team_id: int,
                                 away_team_id: int,
                                 h2h_fixtures: List[dict] = None) -> Dict[str, float]:
        """
        Várható gólok (xG) becslése a form és H2H alapján.
        
        Args:
            home_fixtures: Hazai csapat utolsó meccsei
            away_fixtures: Vendég csapat utolsó meccsei
            home_team_id: Hazai csapat ID
            away_team_id: Vendég csapat ID
            h2h_fixtures: Egymás elleni meccsek (opcionális)
            
        Returns:
            xG becslések és valószínűségek
        """
        # Hazai csapat form
        home_scored = []
        home_conceded = []
        for fixture in home_fixtures[:self.match_count]:
            goals = self._extract_goals(fixture, home_team_id)
            if goals:
                home_scored.append(goals["scored"])
                home_conceded.append(goals["conceded"])
        
        # Vendég csapat form
        away_scored = []
        away_conceded = []
        for fixture in away_fixtures[:self.match_count]:
            goals = self._extract_goals(fixture, away_team_id)
            if goals:
                away_scored.append(goals["scored"])
                away_conceded.append(goals["conceded"])
        
        if not home_scored or not away_scored:
            return {"error": "Nincs elegendő adat"}
        
        # Átlagok
        home_avg_scored = sum(home_scored) / len(home_scored)
        home_avg_conceded = sum(home_conceded) / len(home_conceded)
        away_avg_scored = sum(away_scored) / len(away_scored)
        away_avg_conceded = sum(away_conceded) / len(away_conceded)
        
        # xG számítás
        # Hazai xG = (Hazai támadóerő + Vendég védelmi gyengeség) / 2
        # + 10% hazai pálya előny korrekció
        home_xg = ((home_avg_scored + away_avg_conceded) / 2) * 1.1
        away_xg = (away_avg_scored + home_avg_conceded) / 2
        
        total_xg = home_xg + away_xg
        
        # H2H korrekció ha van adat
        h2h_avg = None
        if h2h_fixtures:
            h2h_goals = []
            for fixture in h2h_fixtures[:5]:
                home_goals = fixture.get("goals", {}).get("home", 0) or 0
                away_goals = fixture.get("goals", {}).get("away", 0) or 0
                h2h_goals.append(home_goals + away_goals)
            if h2h_goals:
                h2h_avg = sum(h2h_goals) / len(h2h_goals)
                # H2H súlyozás: 70% form, 30% H2H
                total_xg = (total_xg * 0.7) + (h2h_avg * 0.3)
        
        # Over/Under valószínűségek (Poisson-alapú egyszerűsítés)
        over_1_5 = self._poisson_over(total_xg, 1.5)
        over_2_5 = self._poisson_over(total_xg, 2.5)
        over_3_5 = self._poisson_over(total_xg, 3.5)
        
        # BTTS valószínűség
        btts_prob = self._calculate_btts_probability(home_xg, away_xg)
        
        return {
            "home_xg": round(home_xg, 2),
            "away_xg": round(away_xg, 2),
            "total_xg": round(total_xg, 2),
            "h2h_avg_goals": round(h2h_avg, 2) if h2h_avg else None,
            "over_1_5_probability": round(over_1_5, 1),
            "over_2_5_probability": round(over_2_5, 1),
            "over_3_5_probability": round(over_3_5, 1),
            "btts_probability": round(btts_prob, 1),
            "matches_analyzed": {
                "home": len(home_scored),
                "away": len(away_scored)
            }
        }
    
    # =========================================================================
    # Tipp Generálás
    # =========================================================================
    
    def generate_betting_tips(self, corner_stats_home: dict,
                               corner_stats_away: dict,
                               card_stats_home: dict,
                               card_stats_away: dict,
                               xg_analysis: dict) -> List[BettingTip]:
        """
        Összes statisztika alapján fogadási tippek generálása.
        
        Returns:
            Rangsorolt fogadási tippek listája
        """
        tips = []
        
        # ----- Szöglet tippek -----
        if "error" not in corner_stats_home and "error" not in corner_stats_away:
            combined_corners = (
                (corner_stats_home.get("avg_total_corners", 0) +
                 corner_stats_away.get("avg_total_corners", 0)) / 2
            )
            
            # Over 9.5 szöglet
            prob_9_5 = (corner_stats_home.get("over_9_5_probability", 50) +
                        corner_stats_away.get("over_9_5_probability", 50)) / 2
            
            if prob_9_5 >= 60:
                confidence = 5 if prob_9_5 >= 80 else (4 if prob_9_5 >= 70 else 3)
                tips.append(BettingTip(
                    bet_type=BetType.CORNERS_OVER,
                    description="Szögletek Over 9.5",
                    line=9.5,
                    confidence=confidence,
                    reasoning=f"Átlagos meccsenkénti szöglet: {combined_corners:.1f}",
                    probability=prob_9_5
                ))
            elif prob_9_5 <= 40:
                confidence = 5 if prob_9_5 <= 20 else (4 if prob_9_5 <= 30 else 3)
                tips.append(BettingTip(
                    bet_type=BetType.CORNERS_UNDER,
                    description="Szögletek Under 9.5",
                    line=9.5,
                    confidence=confidence,
                    reasoning=f"Alacsony szöglet átlag: {combined_corners:.1f}",
                    probability=100 - prob_9_5
                ))
        
        # ----- Sárga lap tippek -----
        if "error" not in card_stats_home and "error" not in card_stats_away:
            combined_cards = (
                (card_stats_home.get("avg_total_cards", 0) +
                 card_stats_away.get("avg_total_cards", 0)) / 2
            )
            
            prob_4_5 = (card_stats_home.get("over_4_5_probability", 50) +
                        card_stats_away.get("over_4_5_probability", 50)) / 2
            
            if prob_4_5 >= 60:
                confidence = 5 if prob_4_5 >= 80 else (4 if prob_4_5 >= 70 else 3)
                tips.append(BettingTip(
                    bet_type=BetType.CARDS_OVER,
                    description="Sárga lapok Over 4.5",
                    line=4.5,
                    confidence=confidence,
                    reasoning=f"Átlagos lapszám meccsenként: {combined_cards:.1f}",
                    probability=prob_4_5
                ))
        
        # ----- Gól tippek -----
        if "error" not in xg_analysis:
            total_xg = xg_analysis.get("total_xg", 2.5)
            over_2_5 = xg_analysis.get("over_2_5_probability", 50)
            btts = xg_analysis.get("btts_probability", 50)
            
            if over_2_5 >= 60:
                confidence = 5 if over_2_5 >= 80 else (4 if over_2_5 >= 70 else 3)
                tips.append(BettingTip(
                    bet_type=BetType.GOALS_OVER,
                    description="Gólok Over 2.5",
                    line=2.5,
                    confidence=confidence,
                    reasoning=f"Várható össz gól (xG): {total_xg:.2f}",
                    probability=over_2_5
                ))
            elif over_2_5 <= 35:
                confidence = 5 if over_2_5 <= 20 else (4 if over_2_5 <= 30 else 3)
                tips.append(BettingTip(
                    bet_type=BetType.GOALS_UNDER,
                    description="Gólok Under 2.5",
                    line=2.5,
                    confidence=confidence,
                    reasoning=f"Alacsony xG: {total_xg:.2f}",
                    probability=100 - over_2_5
                ))
            
            if btts >= 65:
                confidence = 5 if btts >= 80 else (4 if btts >= 70 else 3)
                tips.append(BettingTip(
                    bet_type=BetType.BTTS_YES,
                    description="Mindkét csapat szerez gólt (BTTS)",
                    line=0,
                    confidence=confidence,
                    reasoning=f"Hazai xG: {xg_analysis.get('home_xg', 0):.2f}, Vendég xG: {xg_analysis.get('away_xg', 0):.2f}",
                    probability=btts
                ))
        
        # Rendezés confidence szerint (csökkenő)
        tips.sort(key=lambda t: (t.confidence, t.probability), reverse=True)
        
        return tips
    
    # =========================================================================
    # Segéd Metódusok
    # =========================================================================
    
    def _extract_fixture_stats(self, fixture: dict, team_id: int) -> Optional[dict]:
        """Meccs statisztikák kinyerése egy csapat szempontjából"""
        stats = fixture.get("statistics", [])
        
        if not stats:
            return None
        
        # Meghatározzuk, hogy hazai vagy vendég a csapat
        home_id = fixture.get("teams", {}).get("home", {}).get("id")
        is_home = home_id == team_id
        
        team_stats = stats[0] if is_home else stats[1] if len(stats) > 1 else None
        opponent_stats = stats[1] if is_home else stats[0] if len(stats) > 1 else None
        
        if not team_stats:
            return None
        
        def get_stat_value(statistics: list, stat_type: str) -> int:
            for stat in statistics:
                if stat.get("type") == stat_type:
                    value = stat.get("value")
                    return int(value) if value else 0
            return 0
        
        team_statistics = team_stats.get("statistics", [])
        opponent_statistics = opponent_stats.get("statistics", []) if opponent_stats else []
        
        return {
            "corners_for": get_stat_value(team_statistics, "Corner Kicks"),
            "corners_against": get_stat_value(opponent_statistics, "Corner Kicks"),
            "yellow_cards_for": get_stat_value(team_statistics, "Yellow Cards"),
            "yellow_cards_against": get_stat_value(opponent_statistics, "Yellow Cards"),
            "shots_on_goal": get_stat_value(team_statistics, "Shots on Goal"),
            "possession": get_stat_value(team_statistics, "Ball Possession"),
        }
    
    def _extract_goals(self, fixture: dict, team_id: int) -> Optional[dict]:
        """Gólok kinyerése egy meccsből"""
        home_id = fixture.get("teams", {}).get("home", {}).get("id")
        goals = fixture.get("goals", {})
        
        if not goals:
            return None
        
        is_home = home_id == team_id
        
        return {
            "scored": goals.get("home", 0) if is_home else goals.get("away", 0),
            "conceded": goals.get("away", 0) if is_home else goals.get("home", 0)
        }
    
    def _poisson_over(self, expected: float, line: float) -> float:
        """
        Egyszerűsített Poisson valószínűség Over fogadáshoz.
        
        Normál eloszlást használunk közelítésként a gyorsaság érdekében.
        """
        import math
        
        # Poisson valószínűség: P(X > k) = 1 - sum(P(X = i) for i in 0..floor(k))
        prob_under = 0
        for i in range(int(line) + 1):
            prob_under += (math.exp(-expected) * (expected ** i)) / math.factorial(i)
        
        return (1 - prob_under) * 100
    
    def _calculate_btts_probability(self, home_xg: float, away_xg: float) -> float:
        """Both Teams To Score valószínűség számítása"""
        import math
        
        # P(Home scores) = 1 - P(Home scores 0)
        p_home_scores = 1 - math.exp(-home_xg)
        p_away_scores = 1 - math.exp(-away_xg)
        
        # P(BTTS) = P(Home scores) * P(Away scores)
        return p_home_scores * p_away_scores * 100
    
    def create_match_analysis(self, home_team: dict, away_team: dict,
                               home_fixtures: List[dict], away_fixtures: List[dict],
                               h2h_fixtures: List[dict] = None) -> dict:
        """
        Teljes meccs elemzés készítése.
        
        Returns:
            Összes statisztika és tipp egy dict-ben
        """
        home_id = home_team.get("id")
        away_id = away_team.get("id")
        
        corner_home = self.calculate_corner_stats(home_fixtures, home_id)
        corner_away = self.calculate_corner_stats(away_fixtures, away_id)
        
        card_home = self.calculate_card_stats(home_fixtures, home_id)
        card_away = self.calculate_card_stats(away_fixtures, away_id)
        
        xg = self.estimate_expected_goals(
            home_fixtures, away_fixtures,
            home_id, away_id, h2h_fixtures
        )
        
        tips = self.generate_betting_tips(
            corner_home, corner_away,
            card_home, card_away, xg
        )
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "corners": {
                "home": corner_home,
                "away": corner_away
            },
            "cards": {
                "home": card_home,
                "away": card_away
            },
            "expected_goals": xg,
            "tips": [t.to_dict() for t in tips],
            "analysis_date": __import__("datetime").datetime.now().isoformat()
        }
    
    # =========================================================================
    # Új módszerek scraped adatokhoz
    # =========================================================================
    
    def analyze_matchup(self, home_stats: Dict, away_stats: Dict) -> Dict:
        """
        Meccs elemzése scraped csapat statisztikák alapján.
        
        Args:
            home_stats: Hazai csapat statisztikái (avg_corners, avg_goals, stb.)
            away_stats: Vendég csapat statisztikái
            
        Returns:
            Elemzési eredmények dict-ként
        """
        analysis = {
            "corners": {},
            "cards": {},
            "goals": {},
            "summary": ""
        }
        
        # Szöglet elemzés
        home_corners = home_stats.get('avg_corners', 0) if home_stats else 0
        away_corners = away_stats.get('avg_corners', 0) if away_stats else 0
        combined_corners = home_corners + away_corners
        
        analysis["corners"] = {
            "home_avg": home_corners,
            "away_avg": away_corners,
            "expected_total": combined_corners,
            "over_9_5_likely": combined_corners > 9.5,
            "over_10_5_likely": combined_corners > 10.5
        }
        
        # Sárga lap elemzés
        home_cards = home_stats.get('avg_yellow_cards', 0) if home_stats else 0
        away_cards = away_stats.get('avg_yellow_cards', 0) if away_stats else 0
        combined_cards = home_cards + away_cards
        
        analysis["cards"] = {
            "home_avg": home_cards,
            "away_avg": away_cards,
            "expected_total": combined_cards,
            "over_4_5_likely": combined_cards > 4.5
        }
        
        # Gól elemzés
        home_goals = home_stats.get('avg_goals', 0) if home_stats else 0
        away_goals = away_stats.get('avg_goals', 0) if away_stats else 0
        combined_goals = home_goals + away_goals
        
        # Hazai pálya előny korrekció
        home_xg = home_goals * 1.1
        away_xg = away_goals * 0.9
        
        analysis["goals"] = {
            "home_xg": round(home_xg, 2),
            "away_xg": round(away_xg, 2),
            "expected_total": round(home_xg + away_xg, 2),
            "over_2_5_likely": (home_xg + away_xg) > 2.5,
            "btts_likely": home_xg > 0.8 and away_xg > 0.8
        }
        
        # Összefoglaló
        tips_count = sum([
            analysis["corners"]["over_9_5_likely"],
            analysis["cards"]["over_4_5_likely"],
            analysis["goals"]["over_2_5_likely"]
        ])
        
        if tips_count >= 2:
            analysis["summary"] = "Magas aktivitású meccs várható"
        elif tips_count == 1:
            analysis["summary"] = "Közepes intenzitású meccs"
        else:
            analysis["summary"] = "Alacsony eseményszámú meccs várható"
        
        return analysis
    
    def generate_tips(self, home_stats: Dict, away_stats: Dict) -> List[BettingTip]:
        """
        Tippek generálása scraped statisztikák alapján.
        
        Különösen az Over 9.5 Corners szabályt alkalmazza:
        Ha mindkét csapat átlaga > 5 szöglet/meccs → erős ajánlás
        
        Returns:
            Fogadási tippek listája
        """
        tips = []
        
        home_corners = home_stats.get('avg_corners', 0) if home_stats else 0
        away_corners = away_stats.get('avg_corners', 0) if away_stats else 0
        home_cards = home_stats.get('avg_yellow_cards', 0) if home_stats else 0
        away_cards = away_stats.get('avg_yellow_cards', 0) if away_stats else 0
        home_goals = home_stats.get('avg_goals', 0) if home_stats else 0
        away_goals = away_stats.get('avg_goals', 0) if away_stats else 0
        
        # ===== OVER 9.5 CORNERS szabály =====
        # Ha mindkét csapat átlaga >= 5 szöglet
        if home_corners >= 5 and away_corners >= 5:
            combined = home_corners + away_corners
            probability = min(95, 50 + (combined - 10) * 5)
            confidence = 5 if probability >= 80 else (4 if probability >= 65 else 3)
            
            tips.append(BettingTip(
                bet_type=BetType.CORNERS_OVER,
                description="Szögletek Over 9.5",
                line=9.5,
                confidence=confidence,
                reasoning=f"Hazai átlag: {home_corners:.1f}, Vendég átlag: {away_corners:.1f}",
                probability=probability
            ))
        
        # Over 10.5 szöglet
        combined_corners = home_corners + away_corners
        if combined_corners >= 11:
            probability = min(90, 40 + (combined_corners - 10) * 8)
            confidence = 4 if probability >= 70 else 3
            
            tips.append(BettingTip(
                bet_type=BetType.CORNERS_OVER,
                description="Szögletek Over 10.5",
                line=10.5,
                confidence=confidence,
                reasoning=f"Össz szöglet átlag: {combined_corners:.1f}",
                probability=probability
            ))
        
        # Sárga lapok
        combined_cards = home_cards + away_cards
        if combined_cards >= 5:
            probability = min(85, 45 + (combined_cards - 4) * 10)
            confidence = 4 if probability >= 70 else 3
            
            tips.append(BettingTip(
                bet_type=BetType.CARDS_OVER,
                description="Sárga lapok Over 4.5",
                line=4.5,
                confidence=confidence,
                reasoning=f"Össz lap átlag: {combined_cards:.1f}",
                probability=probability
            ))
        
        # Gólok Over 2.5
        combined_goals = (home_goals * 1.1) + (away_goals * 0.9)  # Hazai előny
        if combined_goals >= 2.8:
            probability = self._poisson_over(combined_goals, 2.5)
            confidence = 5 if probability >= 75 else (4 if probability >= 60 else 3)
            
            tips.append(BettingTip(
                bet_type=BetType.GOALS_OVER,
                description="Gólok Over 2.5",
                line=2.5,
                confidence=confidence,
                reasoning=f"Várható gólszám: {combined_goals:.2f}",
                probability=probability
            ))
        
        # BTTS
        if home_goals >= 1.2 and away_goals >= 1.0:
            btts_prob = self._calculate_btts_probability(home_goals, away_goals)
            if btts_prob >= 55:
                confidence = 4 if btts_prob >= 70 else 3
                
                tips.append(BettingTip(
                    bet_type=BetType.BTTS_YES,
                    description="Mindkét csapat szerez gólt",
                    line=0,
                    confidence=confidence,
                    reasoning=f"Hazai gól/meccs: {home_goals:.1f}, Vendég: {away_goals:.1f}",
                    probability=btts_prob
                ))
        
        # Rendezés
        tips.sort(key=lambda t: (t.confidence, t.probability), reverse=True)
        
        return tips
    
    def calculate_confidence_score(self, home_stats: Dict, away_stats: Dict) -> float:
        """
        Bizalmi szint számítása az utolsó 10 meccs stabilitása alapján.
        
        Algoritmus:
        1. Ellenőrzi van-e elegendő adat (10 meccs)
        2. Az adatmennyiség és stabilitás alapján 0-100% közötti értéket ad
        
        Returns:
            Confidence score 0-100 között
        """
        score = 0
        
        # Alap pontok az adatmennyiség alapján
        home_matches = home_stats.get('matches_analyzed', 0) if home_stats else 0
        away_matches = away_stats.get('matches_analyzed', 0) if away_stats else 0
        
        # Meccsszám alapján (max 40 pont)
        matches_score = min(40, ((home_matches + away_matches) / 20) * 40)
        score += matches_score
        
        # Stabilitás pontok (max 60 pont)
        # Ha vannak statisztikák, magasabb a bizalom
        if home_stats and away_stats:
            has_corners = home_stats.get('avg_corners', 0) > 0 and away_stats.get('avg_corners', 0) > 0
            has_cards = home_stats.get('avg_yellow_cards', 0) > 0 and away_stats.get('avg_yellow_cards', 0) > 0
            has_goals = home_stats.get('avg_goals', 0) > 0 and away_stats.get('avg_goals', 0) > 0
            
            stability_score = 0
            if has_corners:
                stability_score += 20
            if has_cards:
                stability_score += 20
            if has_goals:
                stability_score += 20
            
            score += stability_score
        elif home_stats or away_stats:
            # Csak egyik csapatról van adat
            score += 20
        
        return min(100, max(0, score))
