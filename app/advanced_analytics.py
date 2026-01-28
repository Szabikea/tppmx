"""
Advanced Betting Analytics Engine
==================================
Professzionális sportfogadási matematikai modell:
- Súlyozott Forma-index
- Poisson-modellezés
- Value Bet detektálás
- Szöglet/Lap predikció szórással
- xG integráció
- Kiugró értékek szűrése
- Konfidencia intervallum
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from statistics import mean, stdev
import hashlib
import random


@dataclass
class TeamStats:
    """Csapat statisztikák strukturálva"""
    name: str
    attack_strength: float = 1.0
    defense_strength: float = 1.0
    avg_goals_scored: float = 1.5
    avg_goals_conceded: float = 1.2
    avg_corners: float = 5.0
    corners_std: float = 2.0
    avg_cards: float = 2.0
    cards_std: float = 1.0
    xg_avg: Optional[float] = None
    form_index: float = 50.0
    matches_played: int = 0
    has_outliers: bool = False


@dataclass
class PoissonPrediction:
    """Poisson-modell előrejelzés"""
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    expected_home_goals: float
    expected_away_goals: float
    over_25_prob: float
    btts_prob: float
    confidence_interval: Tuple[float, float]
    confidence_level: str  # 'high', 'medium', 'low'


@dataclass
class ValueBet:
    """Értékes fogadás jelzése"""
    bet_type: str
    description: str
    our_probability: float
    implied_odds_prob: float
    edge: float  # (our_prob - implied_prob)
    recommended_odds: float
    is_value: bool
    confidence: int  # 1-5


class AdvancedAnalytics:
    """Professzionális fogadási elemző motor"""
    
    # Liga átlagok referenciához
    LEAGUE_AVERAGES = {
        'Premier League': {'goals': 2.77, 'corners': 10.5, 'cards': 3.8},
        'La Liga': {'goals': 2.51, 'corners': 9.8, 'cards': 4.2},
        'Bundesliga': {'goals': 3.12, 'corners': 10.2, 'cards': 3.5},
        'Serie A': {'goals': 2.68, 'corners': 10.0, 'cards': 4.5},
        'Ligue 1': {'goals': 2.55, 'corners': 9.5, 'cards': 3.9},
        'Champions League': {'goals': 2.85, 'corners': 10.8, 'cards': 3.2},
    }
    
    # Súlyok az utolsó meccsekhez
    MATCH_WEIGHTS = {
        0: 2.0, 1: 2.0, 2: 2.0,  # Utolsó 3 meccs 2x súly
        3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0  # Korábbi 7 meccs 1x
    }
    
    def __init__(self):
        self.team_cache: Dict[str, TeamStats] = {}
    
    def calculate_weighted_form_index(self, 
                                       recent_results: List[Dict],
                                       is_home: bool = True) -> Tuple[float, TeamStats]:
        """
        Súlyozott Forma-index számítás
        Az utolsó 10 meccs alapján, az utolsó 3 meccs 2x súllyal
        
        Returns:
            (form_index 0-100, TeamStats objektum)
        """
        if not recent_results:
            return 50.0, TeamStats(name="Unknown")
        
        # Statisztikák gyűjtése
        goals_scored = []
        goals_conceded = []
        corners = []
        cards = []
        xgs = []
        weights = []
        points = 0
        max_points = 0
        
        for i, match in enumerate(recent_results[:10]):
            weight = self.MATCH_WEIGHTS.get(i, 1.0)
            weights.append(weight)
            
            # Gólok
            gs = match.get('goals_scored', match.get('home_score', 0)) or 0
            gc = match.get('goals_conceded', match.get('away_score', 0)) or 0
            goals_scored.append(gs * weight)
            goals_conceded.append(gc * weight)
            
            # Szögletek (ha van adat)
            if 'corners' in match:
                corners.append(match['corners'] * weight)
            
            # Lapok
            if 'cards' in match:
                cards.append(match['cards'] * weight)
            
            # xG
            if 'xg' in match:
                xgs.append(match['xg'] * weight)
            
            # Pontok
            if gs > gc:
                points += 3 * weight
            elif gs == gc:
                points += 1 * weight
            max_points += 3 * weight
        
        total_weight = sum(weights)
        
        # Átlagok számítása
        avg_scored = sum(goals_scored) / total_weight if total_weight > 0 else 1.5
        avg_conceded = sum(goals_conceded) / total_weight if total_weight > 0 else 1.2
        
        # Forma-index (0-100 skála)
        form_index = (points / max_points * 100) if max_points > 0 else 50.0
        
        # TeamStats összeállítása
        stats = TeamStats(
            name=recent_results[0].get('team', 'Unknown') if recent_results else 'Unknown',
            avg_goals_scored=avg_scored,
            avg_goals_conceded=avg_conceded,
            attack_strength=avg_scored / 1.5,  # Liga átlaghoz viszonyított
            defense_strength=avg_conceded / 1.2,
            form_index=form_index,
            matches_played=len(recent_results),
            avg_corners=sum(corners) / total_weight if corners else 5.0,
            corners_std=self._safe_stdev([c / w for c, w in zip(corners, weights[:len(corners)])]) if len(corners) > 1 else 2.0,
            avg_cards=sum(cards) / total_weight if cards else 2.0,
            cards_std=self._safe_stdev([c / w for c, w in zip(cards, weights[:len(cards)])]) if len(cards) > 1 else 1.0,
            xg_avg=sum(xgs) / total_weight if xgs else None
        )
        
        return form_index, stats
    
    def _safe_stdev(self, data: List[float]) -> float:
        """Biztonságos szórás számítás"""
        if len(data) < 2:
            return 0.0
        try:
            return stdev(data)
        except:
            return 0.0
    
    def poisson_probability(self, lam: float, k: int) -> float:
        """
        Poisson valószínűség: P(X=k) = (λ^k * e^(-λ)) / k!
        """
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return (math.pow(lam, k) * math.exp(-lam)) / math.factorial(k)
    
    def calculate_poisson_predictions(self,
                                       home_stats: TeamStats,
                                       away_stats: TeamStats,
                                       league: str = 'Premier League') -> PoissonPrediction:
        """
        Poisson-modellezés a H/D/V esélyek számításához
        
        A modell:
        - Expected Home Goals = Home Attack * Away Defense * League Avg * Home Advantage
        - Expected Away Goals = Away Attack * Home Defense * League Avg
        """
        league_avg = self.LEAGUE_AVERAGES.get(league, {'goals': 2.7})['goals']
        avg_per_team = league_avg / 2
        
        # Home advantage faktor (kb. 1.2-1.4)
        home_advantage = 1.25
        
        # Várható gólok
        # Ha van xG adat, azt használjuk
        if home_stats.xg_avg is not None:
            expected_home = home_stats.xg_avg * home_advantage
        else:
            expected_home = (home_stats.attack_strength * 
                           away_stats.defense_strength * 
                           avg_per_team * home_advantage)
        
        if away_stats.xg_avg is not None:
            expected_away = away_stats.xg_avg * 0.9  # Vendég hátrány
        else:
            expected_away = (away_stats.attack_strength * 
                           home_stats.defense_strength * 
                           avg_per_team * 0.9)
        
        # Poisson eloszlás mátrix (0-6 gól)
        max_goals = 7
        home_probs = [self.poisson_probability(expected_home, i) for i in range(max_goals)]
        away_probs = [self.poisson_probability(expected_away, i) for i in range(max_goals)]
        
        # H/D/V valószínűségek
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        over_25 = 0.0
        btts = 0.0
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob = home_probs[i] * away_probs[j]
                
                if i > j:
                    home_win += prob
                elif i == j:
                    draw += prob
                else:
                    away_win += prob
                
                if i + j > 2:
                    over_25 += prob
                
                if i > 0 and j > 0:
                    btts += prob
        
        # Normalizálás
        total = home_win + draw + away_win
        if total > 0:
            home_win /= total
            draw /= total
            away_win /= total
        
        # Konfidencia intervallum és szint
        confidence_interval, confidence_level = self._calculate_confidence(
            home_stats, away_stats, expected_home, expected_away
        )
        
        return PoissonPrediction(
            home_win_prob=round(home_win * 100, 1),
            draw_prob=round(draw * 100, 1),
            away_win_prob=round(away_win * 100, 1),
            expected_home_goals=round(expected_home, 2),
            expected_away_goals=round(expected_away, 2),
            over_25_prob=round(over_25 * 100, 1),
            btts_prob=round(btts * 100, 1),
            confidence_interval=confidence_interval,
            confidence_level=confidence_level
        )
    
    def _calculate_confidence(self,
                              home_stats: TeamStats,
                              away_stats: TeamStats,
                              exp_home: float,
                              exp_away: float) -> Tuple[Tuple[float, float], str]:
        """
        Konfidencia intervallum számítása
        A variancia a Poisson eloszlásnál = λ
        95% CI: λ ± 1.96 * sqrt(λ)
        """
        # Össz gól variancia
        total_lambda = exp_home + exp_away
        variance = math.sqrt(total_lambda)
        
        # 95% konfidencia intervallum
        ci_lower = max(0, total_lambda - 1.96 * variance)
        ci_upper = total_lambda + 1.96 * variance
        
        # Ha nincs valós adat (matches_played == 0), azonnal LOW confidence
        if home_stats.matches_played == 0 or away_stats.matches_played == 0:
            return (round(ci_lower, 1), round(ci_upper, 1)), 'low'
        
        # Konfidencia szint az adatok alapján
        data_quality_score = 0
        
        # Több meccs = magasabb bizalom
        if home_stats.matches_played >= 8 and away_stats.matches_played >= 8:
            data_quality_score += 2
        elif home_stats.matches_played >= 5 and away_stats.matches_played >= 5:
            data_quality_score += 1
        
        # xG adat = magasabb bizalom
        if home_stats.xg_avg is not None and away_stats.xg_avg is not None:
            data_quality_score += 2
        
        # Alacsony szórás = stabilabb eredmények
        if home_stats.corners_std < 2.0 and away_stats.corners_std < 2.0:
            data_quality_score += 1
        
        # Nincs kiugró érték
        if not home_stats.has_outliers and not away_stats.has_outliers:
            data_quality_score += 1
        
        if data_quality_score >= 5:
            level = 'high'
        elif data_quality_score >= 3:
            level = 'medium'
        else:
            level = 'low'
        
        return (round(ci_lower, 1), round(ci_upper, 1)), level
    
    def detect_value_bets(self,
                          prediction: PoissonPrediction,
                          market_odds: Dict[str, float] = None) -> List[ValueBet]:
        """
        Value Bet detektálás
        Összehasonlítja a mi valószínűségünket a piaci odds-szal
        Value Bet = Ha a mi prob > implied prob (edge > 0)
        """
        value_bets = []
        
        # Ha nincs piaci odds, generáljunk becsült odds-ot
        if market_odds is None:
            market_odds = {
                'home': self._prob_to_odds(prediction.home_win_prob / 100) * 0.95,  # 5% margin
                'draw': self._prob_to_odds(prediction.draw_prob / 100) * 0.95,
                'away': self._prob_to_odds(prediction.away_win_prob / 100) * 0.95,
                'over25': self._prob_to_odds(prediction.over_25_prob / 100) * 0.95,
                'btts': self._prob_to_odds(prediction.btts_prob / 100) * 0.95
            }
        
        # Hazai győzelem
        implied_home = 100 / market_odds.get('home', 2.0)
        edge_home = prediction.home_win_prob - implied_home
        value_bets.append(ValueBet(
            bet_type='1X2',
            description='Hazai győzelem (1)',
            our_probability=prediction.home_win_prob,
            implied_odds_prob=round(implied_home, 1),
            edge=round(edge_home, 1),
            recommended_odds=market_odds.get('home', 2.0),
            is_value=edge_home > 3.0,  # 3% felett value
            confidence=self._edge_to_confidence(edge_home, prediction.confidence_level)
        ))
        
        # Döntetlen
        implied_draw = 100 / market_odds.get('draw', 3.5)
        edge_draw = prediction.draw_prob - implied_draw
        value_bets.append(ValueBet(
            bet_type='1X2',
            description='Döntetlen (X)',
            our_probability=prediction.draw_prob,
            implied_odds_prob=round(implied_draw, 1),
            edge=round(edge_draw, 1),
            recommended_odds=market_odds.get('draw', 3.5),
            is_value=edge_draw > 3.0,
            confidence=self._edge_to_confidence(edge_draw, prediction.confidence_level)
        ))
        
        # Vendég győzelem
        implied_away = 100 / market_odds.get('away', 3.0)
        edge_away = prediction.away_win_prob - implied_away
        value_bets.append(ValueBet(
            bet_type='1X2',
            description='Vendég győzelem (2)',
            our_probability=prediction.away_win_prob,
            implied_odds_prob=round(implied_away, 1),
            edge=round(edge_away, 1),
            recommended_odds=market_odds.get('away', 3.0),
            is_value=edge_away > 3.0,
            confidence=self._edge_to_confidence(edge_away, prediction.confidence_level)
        ))
        
        # Over 2.5
        implied_over = 100 / market_odds.get('over25', 1.9)
        edge_over = prediction.over_25_prob - implied_over
        value_bets.append(ValueBet(
            bet_type='goals',
            description='2.5 gól felett',
            our_probability=prediction.over_25_prob,
            implied_odds_prob=round(implied_over, 1),
            edge=round(edge_over, 1),
            recommended_odds=market_odds.get('over25', 1.9),
            is_value=edge_over > 3.0,
            confidence=self._edge_to_confidence(edge_over, prediction.confidence_level)
        ))
        
        # BTTS
        implied_btts = 100 / market_odds.get('btts', 1.85)
        edge_btts = prediction.btts_prob - implied_btts
        value_bets.append(ValueBet(
            bet_type='goals',
            description='Mindkét csapat szerez gólt',
            our_probability=prediction.btts_prob,
            implied_odds_prob=round(implied_btts, 1),
            edge=round(edge_btts, 1),
            recommended_odds=market_odds.get('btts', 1.85),
            is_value=edge_btts > 3.0,
            confidence=self._edge_to_confidence(edge_btts, prediction.confidence_level)
        ))
        
        return value_bets
    
    def _prob_to_odds(self, prob: float) -> float:
        """Valószínűség -> odds konverzió"""
        if prob <= 0:
            return 100.0
        return round(1 / prob, 2)
    
    def _edge_to_confidence(self, edge: float, level: str) -> int:
        """Edge és adatminőség -> konfidencia (1-5)"""
        base = 3
        if level == 'high':
            base = 4
        elif level == 'low':
            base = 2
        
        if edge > 10:
            return min(5, base + 1)
        elif edge > 5:
            return base
        elif edge > 0:
            return max(1, base - 1)
        else:
            return max(1, base - 2)
    
    def predict_corners(self,
                        home_stats: TeamStats,
                        away_stats: TeamStats,
                        league: str = 'Premier League') -> Dict:
        """
        Szöglet predikció szórással
        Ha a szórás alacsony -> megbízhatóbb tipp
        """
        league_avg = self.LEAGUE_AVERAGES.get(league, {'corners': 10.0})['corners']
        
        expected_corners = home_stats.avg_corners + away_stats.avg_corners
        combined_std = math.sqrt(home_stats.corners_std**2 + away_stats.corners_std**2)
        
        # Reliability score (alacsony szórás = magas megbízhatóság)
        if combined_std < 2.5:
            reliability = 'high'
            reliability_score = 5
        elif combined_std < 4.0:
            reliability = 'medium'
            reliability_score = 3
        else:
            reliability = 'low'
            reliability_score = 2
        
        # 95% konfidencia intervallum
        ci_lower = max(0, expected_corners - 1.96 * combined_std)
        ci_upper = expected_corners + 1.96 * combined_std
        
        # Over/Under vonalak
        lines = [8.5, 9.5, 10.5, 11.5, 12.5]
        over_probs = {}
        for line in lines:
            # Z-score számítás
            z = (line - expected_corners) / combined_std if combined_std > 0 else 0
            # Normál eloszlás CDF közelítés
            over_prob = 1 - self._normal_cdf(z)
            over_probs[f'over_{line}'] = round(over_prob * 100, 1)
        
        return {
            'expected_total': round(expected_corners, 1),
            'confidence_interval': (round(ci_lower, 1), round(ci_upper, 1)),
            'standard_deviation': round(combined_std, 2),
            'reliability': reliability,
            'reliability_score': reliability_score,
            'over_probs': over_probs,
            'best_bet': self._find_best_corner_bet(over_probs, expected_corners)
        }
    
    def predict_cards(self,
                      home_stats: TeamStats,
                      away_stats: TeamStats,
                      referee_strictness: float = 1.0) -> Dict:
        """
        Sárga lap predikció
        Ha összeadott átlag > 4.5 és szigorú bíró -> Over ajánlás
        """
        expected_cards = (home_stats.avg_cards + away_stats.avg_cards) * referee_strictness
        combined_std = math.sqrt(home_stats.cards_std**2 + away_stats.cards_std**2)
        
        # Over 4.5 lap szignál
        is_over_signal = expected_cards > 4.5 and referee_strictness >= 1.0
        
        # Vonalak
        lines = [3.5, 4.5, 5.5, 6.5]
        over_probs = {}
        for line in lines:
            z = (line - expected_cards) / combined_std if combined_std > 0 else 0
            over_prob = 1 - self._normal_cdf(z)
            over_probs[f'over_{line}'] = round(over_prob * 100, 1)
        
        return {
            'expected_total': round(expected_cards, 1),
            'confidence_interval': (
                round(max(0, expected_cards - 1.96 * combined_std), 1),
                round(expected_cards + 1.96 * combined_std, 1)
            ),
            'standard_deviation': round(combined_std, 2),
            'over_signal': is_over_signal,
            'referee_factor': referee_strictness,
            'over_probs': over_probs,
            'recommendation': 'Over 4.5 lapok ajánlott' if is_over_signal else None
        }
    
    def _normal_cdf(self, z: float) -> float:
        """Standard normál eloszlás CDF közelítése"""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    
    def _find_best_corner_bet(self, over_probs: Dict, expected: float) -> Optional[Dict]:
        """Legjobb szöglet fogadás keresése"""
        best = None
        best_value = 0
        
        for line_key, prob in over_probs.items():
            line = float(line_key.split('_')[1])
            # Value = |prob - 50| (minél távolabb 50%-tól, annál jobb)
            value = abs(prob - 50)
            
            if value > best_value and value > 15:  # Min 15% edge
                best_value = value
                if prob > 55:
                    best = {'line': line, 'direction': 'over', 'probability': prob}
                elif prob < 45:
                    best = {'line': line, 'direction': 'under', 'probability': 100 - prob}
        
        return best
    
    def filter_outliers(self, matches: List[Dict]) -> List[Dict]:
        """
        Kiugró értékek szűrése
        Kiszűri a torzító meccseket (korai piros lap, extrém eredmények)
        """
        if not matches:
            return []
        
        filtered = []
        
        for match in matches:
            is_outlier = False
            
            # Korai piros lap ellenőrzés
            red_cards = match.get('red_cards', 0)
            red_card_minute = match.get('red_card_minute', 90)
            if red_cards > 0 and red_card_minute < 30:
                is_outlier = True
            
            # Extrém gólkülönbség (5+)
            goals_scored = match.get('goals_scored', 0) or 0
            goals_conceded = match.get('goals_conceded', 0) or 0
            if abs(goals_scored - goals_conceded) >= 5:
                is_outlier = True
            
            # Extrém szöglet szám (20+)
            corners = match.get('corners', 0)
            if corners >= 20:
                is_outlier = True
            
            if not is_outlier:
                filtered.append(match)
            else:
                match['is_outlier'] = True
        
        return filtered
    
    def analyze_match(self,
                      fixture: Dict,
                      home_history: List[Dict] = None,
                      away_history: List[Dict] = None,
                      market_odds: Dict = None) -> Dict:
        """
        Teljes meccs elemzés egy helyen
        Ez a fő belépési pont az analytics-hez
        """
        league = fixture.get('league', 'Premier League')
        
        # Csapat statisztikák (ha van történeti adat, azt használjuk)
        if home_history:
            home_history = self.filter_outliers(home_history)
            _, home_stats = self.calculate_weighted_form_index(home_history, is_home=True)
        else:
            # Alapértelmezett statisztikák generálása
            home_stats = self._generate_team_stats(fixture.get('home_team', 'Home'), league)
        
        if away_history:
            away_history = self.filter_outliers(away_history)
            _, away_stats = self.calculate_weighted_form_index(away_history, is_home=False)
        else:
            away_stats = self._generate_team_stats(fixture.get('away_team', 'Away'), league)
        
        # Poisson predikció
        prediction = self.calculate_poisson_predictions(home_stats, away_stats, league)
        
        # Value Bet-ek
        value_bets = self.detect_value_bets(prediction, market_odds)
        
        # Szöglet és lap predikciók
        corners = self.predict_corners(home_stats, away_stats, league)
        cards = self.predict_cards(home_stats, away_stats)
        
        return {
            'fixture': fixture,
            'home_stats': home_stats,
            'away_stats': away_stats,
            'prediction': prediction,
            'value_bets': value_bets,
            'corners': corners,
            'cards': cards,
            'confidence_level': prediction.confidence_level,
            'confidence_interval': prediction.confidence_interval,
            'has_value_bet': any(vb.is_value for vb in value_bets),
            'best_value_bet': max(value_bets, key=lambda x: x.edge) if value_bets else None
        }
    
    def _generate_team_stats(self, team_name: str, league: str) -> TeamStats:
        """
        Csapat statisztikák generálása ha nincs történeti adat
        Determinisztikus a csapatnév alapján
        """
        # Determinisztikus seed
        seed = int(hashlib.md5(team_name.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        
        league_data = self.LEAGUE_AVERAGES.get(league, {'goals': 2.7, 'corners': 10.0, 'cards': 4.0})
        
        # Konzervatívabb variancia (random adatoknál ne legyenek extrém eltérések)
        strength_var = rng.uniform(0.9, 1.1)
        
        return TeamStats(
            name=team_name,
            attack_strength=strength_var,
            defense_strength=1 / strength_var,
            avg_goals_scored=league_data['goals'] / 2 * strength_var,
            avg_goals_conceded=league_data['goals'] / 2 / strength_var,
            avg_corners=league_data['corners'] / 2 * rng.uniform(0.9, 1.1),
            corners_std=rng.uniform(3.0, 5.0), # Magasabb szórás a bizonytalanság miatt
            avg_cards=league_data['cards'] / 2 * rng.uniform(0.9, 1.1),
            cards_std=rng.uniform(1.5, 2.5),
            form_index=50.0, # Semleges forma
            matches_played=0, # Jelezzük, hogy ez generált adat
            xg_avg=None
        )


# Singleton instance
_analytics_engine = None

def get_analytics_engine() -> AdvancedAnalytics:
    """Analytics engine singleton"""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = AdvancedAnalytics()
    return _analytics_engine
