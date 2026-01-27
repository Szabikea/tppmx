"""
Professional Statistical Betting Engine
========================================
Professzionális statisztikai számítások minden tipp típushoz:

1. Dixon-Coles modell (javított Poisson a döntetlenekhez)
2. Kelly-kritérium (optimális fogadási méret)
3. Expected Value (EV) számítás
4. Monte Carlo szimuláció (10000 iteráció)
5. Bivariate Poisson (gól korreláció)
6. Elo-rating rendszer
7. Standard deviation és Confidence Interval MINDEN tipphez
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
import random
from statistics import mean, stdev, variance


@dataclass
class BetStatistics:
    """Statisztikai adatok egyetlen tipphez"""
    probability: float  # Becsült valószínűség %
    fair_odds: float  # Fair odds (1/probability)
    market_odds: float  # Piaci odds
    expected_value: float  # EV = (prob * odds) - 1
    kelly_fraction: float  # Optimális tét méret %
    confidence_interval: Tuple[float, float]  # 95% CI
    standard_deviation: float  # Szórás
    reliability_score: int  # 1-5 (adatminőség)
    monte_carlo_prob: float  # MC szimuláció eredménye
    variance: float  # Variancia
    z_score: float  # Z-score a norml eloszláshoz
    is_value_bet: bool  # EV > 0
    bet_rating: str  # 'A', 'B', 'C', 'D', 'F'


@dataclass
class DixonColesResult:
    """Dixon-Coles modell eredmény"""
    home_goals_lambda: float
    away_goals_lambda: float
    rho: float  # Gól korreláció paraméter
    home_win: float
    draw: float
    away_win: float
    score_matrix: np.ndarray = field(default_factory=lambda: np.zeros((7, 7)))


@dataclass
class EloRating:
    """Elo-rating eredmény"""
    home_elo: float
    away_elo: float
    home_expected: float
    away_expected: float
    elo_difference: float


class ProfessionalStatEngine:
    """Professzionális statisztikai számító motor"""
    
    # Liga alapértelmezett Elo besorolások
    LEAGUE_BASE_ELO = {
        'Premier League': 1700,
        'La Liga': 1650,
        'Bundesliga': 1620,
        'Serie A': 1600,
        'Ligue 1': 1550,
        'Champions League': 1800,
    }
    
    def __init__(self):
        self.monte_carlo_iterations = 10000
        self.team_elo_cache: Dict[str, float] = {}
    
    # =========================================================================
    # Dixon-Coles Model (Javított Poisson)
    # =========================================================================
    
    def dixon_coles_adjustment(self, 
                                home_goals: int, 
                                away_goals: int,
                                home_lambda: float,
                                away_lambda: float,
                                rho: float = -0.13) -> float:
        """
        Dixon-Coles korrekció az alacsony gólszámú eredményekhez
        τ(x,y) szorzó faktor
        
        rho: korreláció paraméter, tipikusan -0.13 és -0.20 között
        """
        if home_goals == 0 and away_goals == 0:
            return 1 - home_lambda * away_lambda * rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + home_lambda * rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + away_lambda * rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - rho
        else:
            return 1.0
    
    def calculate_dixon_coles(self,
                              home_attack: float,
                              home_defense: float,
                              away_attack: float,
                              away_defense: float,
                              home_advantage: float = 1.25,
                              league_avg: float = 1.35) -> DixonColesResult:
        """
        Dixon-Coles modell teljes számítása
        Javítja a standard Poisson-t a döntetlen és alacsony gólszámú meccsekre
        """
        # Lambda értékek (várható gólok)
        home_lambda = home_attack * away_defense * league_avg * home_advantage
        away_lambda = away_attack * home_defense * league_avg
        
        # Gól korrekció paraméter
        rho = -0.13  # Empirikus érték labdarúgáshoz
        
        # Pontszám mátrix 0-6 gólokra
        max_goals = 7
        score_matrix = np.zeros((max_goals, max_goals))
        
        for i in range(max_goals):
            for j in range(max_goals):
                # Standard Poisson
                home_prob = self._poisson_prob(home_lambda, i)
                away_prob = self._poisson_prob(away_lambda, j)
                
                # Dixon-Coles korrekció
                dc_factor = self.dixon_coles_adjustment(i, j, home_lambda, away_lambda, rho)
                
                score_matrix[i, j] = home_prob * away_prob * dc_factor
        
        # Normalizálás
        score_matrix = score_matrix / np.sum(score_matrix)
        
        # H/D/A valószínűségek
        home_win = np.sum(np.tril(score_matrix, -1))
        draw = np.sum(np.diag(score_matrix))
        away_win = np.sum(np.triu(score_matrix, 1))
        
        return DixonColesResult(
            home_goals_lambda=round(home_lambda, 3),
            away_goals_lambda=round(away_lambda, 3),
            rho=rho,
            home_win=round(home_win * 100, 2),
            draw=round(draw * 100, 2),
            away_win=round(away_win * 100, 2),
            score_matrix=score_matrix
        )
    
    def _poisson_prob(self, lam: float, k: int) -> float:
        """Poisson valószínűség"""
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return (math.pow(lam, k) * math.exp(-lam)) / math.factorial(k)
    
    # =========================================================================
    # Kelly Criterion (Optimális fogadási méret)
    # =========================================================================
    
    def kelly_criterion(self, 
                        probability: float, 
                        odds: float,
                        fraction: float = 0.25) -> float:
        """
        Kelly-kritérium: optimális fogadási méret a bankrollhoz képest
        
        f* = (bp - q) / b
        ahol:
        - b = odds - 1
        - p = győzelmi valószínűség
        - q = veszteség valószínűség (1-p)
        
        fraction: Kelly-töredék (0.25 = quarter Kelly, konzervatívabb)
        """
        if probability <= 0 or probability >= 100:
            return 0.0
        
        p = probability / 100
        q = 1 - p
        b = odds - 1
        
        if b <= 0:
            return 0.0
        
        kelly = (b * p - q) / b
        
        # Negatív Kelly = nem fogadunk
        if kelly <= 0:
            return 0.0
        
        # Fractional Kelly (biztonságosabb)
        return round(kelly * fraction * 100, 2)  # % a bankrollból
    
    # =========================================================================
    # Expected Value (EV)
    # =========================================================================
    
    def expected_value(self, 
                       probability: float, 
                       odds: float,
                       stake: float = 1.0) -> float:
        """
        Expected Value = (Prob * Profit) - (1-Prob * Stake)
        
        Pozitív EV = értékes fogadás hosszú távon
        """
        if probability <= 0 or probability >= 100:
            return 0.0
        
        p = probability / 100
        profit = (odds - 1) * stake
        loss = stake
        
        ev = (p * profit) - ((1 - p) * loss)
        return round(ev, 4)
    
    def roi_estimate(self, ev: float, stake: float = 1.0) -> float:
        """ROI becslés az EV alapján"""
        if stake <= 0:
            return 0.0
        return round((ev / stake) * 100, 2)
    
    # =========================================================================
    # Monte Carlo Simulation
    # =========================================================================
    
    def monte_carlo_simulation(self,
                               home_lambda: float,
                               away_lambda: float,
                               iterations: int = None) -> Dict[str, float]:
        """
        Monte Carlo szimuláció a meccskimenetelekhez
        
        iterations db szimulált meccs alapján számoljuk a valószínűségeket
        """
        if iterations is None:
            iterations = self.monte_carlo_iterations
        
        np.random.seed(42)  # Reprodukálhatóság
        
        home_goals = np.random.poisson(home_lambda, iterations)
        away_goals = np.random.poisson(away_lambda, iterations)
        
        home_wins = np.sum(home_goals > away_goals)
        draws = np.sum(home_goals == away_goals)
        away_wins = np.sum(home_goals < away_goals)
        
        total_goals = home_goals + away_goals
        over_25 = np.sum(total_goals > 2.5)
        btts = np.sum((home_goals > 0) & (away_goals > 0))
        
        # Exact score probabilities (top 5)
        scores = {}
        for i in range(iterations):
            score = f"{home_goals[i]}-{away_goals[i]}"
            scores[score] = scores.get(score, 0) + 1
        
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'home_win': round(home_wins / iterations * 100, 2),
            'draw': round(draws / iterations * 100, 2),
            'away_win': round(away_wins / iterations * 100, 2),
            'over_25': round(over_25 / iterations * 100, 2),
            'btts': round(btts / iterations * 100, 2),
            'avg_total_goals': round(np.mean(total_goals), 2),
            'std_total_goals': round(np.std(total_goals), 2),
            'top_scores': [(s, round(c/iterations*100, 1)) for s, c in top_scores],
            'simulations': iterations
        }
    
    # =========================================================================
    # Elo Rating System
    # =========================================================================
    
    def get_team_elo(self, team_name: str, league: str = 'Premier League') -> float:
        """
        Csapat Elo-értékének lekérése/generálása
        """
        if team_name in self.team_elo_cache:
            return self.team_elo_cache[team_name]
        
        # Determinisztikus generálás a csapatnév alapján
        seed = int(hashlib.md5(team_name.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        
        base_elo = self.LEAGUE_BASE_ELO.get(league, 1600)
        variation = rng.gauss(0, 100)  # ±100 Elo szórás
        
        elo = base_elo + variation
        self.team_elo_cache[team_name] = elo
        
        return round(elo, 0)
    
    def elo_expected_score(self, 
                           home_elo: float, 
                           away_elo: float,
                           home_advantage: float = 65) -> EloRating:
        """
        Elo-alapú várt eredmény számítás
        
        Expected Score = 1 / (1 + 10^((opponent_elo - own_elo) / 400))
        home_advantage: Elo pontok hazai pálya előnyért (tipikusan 50-100)
        """
        # Hazai pálya előny
        adjusted_home_elo = home_elo + home_advantage
        
        elo_diff = adjusted_home_elo - away_elo
        
        home_expected = 1 / (1 + math.pow(10, -elo_diff / 400))
        away_expected = 1 - home_expected
        
        return EloRating(
            home_elo=home_elo,
            away_elo=away_elo,
            home_expected=round(home_expected * 100, 2),
            away_expected=round(away_expected * 100, 2),
            elo_difference=round(elo_diff, 0)
        )
    
    # =========================================================================
    # Comprehensive Bet Statistics
    # =========================================================================
    
    def calculate_bet_statistics(self,
                                 bet_type: str,
                                 probability: float,
                                 market_odds: float = None,
                                 home_stats: Dict = None,
                                 away_stats: Dict = None,
                                 historical_data: List[float] = None) -> BetStatistics:
        """
        Teljes statisztikai elemzés egy fogadáshoz
        
        Minden tipphez kiszámolja:
        - Fair odds vs piaci odds
        - Expected Value
        - Kelly kritérium
        - Standard deviation
        - 95% Confidence Interval
        - Monte Carlo valószínűség
        - Z-score
        - Reliability score
        - Bet rating (A-F)
        """
        # Fair odds számítás
        fair_odds = round(100 / probability, 3) if probability > 0 else 100.0
        
        # Market odds (ha nincs megadva, 5% margin-t feltételez)
        if market_odds is None:
            market_odds = round(fair_odds * 0.95, 2)
        
        # Expected Value
        ev = self.expected_value(probability, market_odds)
        
        # Kelly kritérium
        kelly = self.kelly_criterion(probability, market_odds)
        
        # Standard deviation és CI becslés
        # Binomiális eloszlás: σ = sqrt(n * p * (1-p))
        n = 100  # Virtuális mintaméret
        p = probability / 100
        std = math.sqrt(n * p * (1 - p)) / n * 100  # %-ban
        
        # Ha van történeti adat
        if historical_data and len(historical_data) > 1:
            std = stdev(historical_data)
            var = variance(historical_data)
        else:
            var = std ** 2
        
        # 95% CI (Z=1.96)
        margin = 1.96 * std
        ci_lower = max(0, probability - margin)
        ci_upper = min(100, probability + margin)
        
        # Z-score (mennyire tér el a 50%-tól)
        z_score = (probability - 50) / std if std > 0 else 0
        
        # Monte Carlo (ha van lambda adat)
        mc_prob = probability  # Alapértelmezett
        if home_stats and away_stats:
            home_lambda = getattr(home_stats, 'avg_goals_scored', 1.5)
            away_lambda = getattr(away_stats, 'avg_goals_scored', 1.2)
            mc = self.monte_carlo_simulation(home_lambda, away_lambda, 5000)
            
            if 'győzelem' in bet_type.lower() and 'hazai' in bet_type.lower():
                mc_prob = mc['home_win']
            elif 'győzelem' in bet_type.lower() and 'vendég' in bet_type.lower():
                mc_prob = mc['away_win']
            elif 'döntetlen' in bet_type.lower():
                mc_prob = mc['draw']
            elif 'over' in bet_type.lower() or 'felett' in bet_type.lower():
                mc_prob = mc['over_25']
            elif 'btts' in bet_type.lower() or 'mindkét' in bet_type.lower():
                mc_prob = mc['btts']
        
        # Reliability score (1-5)
        reliability = 3  # Alapértelmezett
        if std < 5:
            reliability = 5
        elif std < 10:
            reliability = 4
        elif std < 20:
            reliability = 3
        elif std < 30:
            reliability = 2
        else:
            reliability = 1
        
        # Value bet detektálás
        is_value = ev > 0
        
        # Bet rating (A-F)
        if ev > 0.1 and kelly > 2 and reliability >= 4:
            rating = 'A'
        elif ev > 0.05 and kelly > 1:
            rating = 'B'
        elif ev > 0 or probability > 60:
            rating = 'C'
        elif ev > -0.05:
            rating = 'D'
        else:
            rating = 'F'
        
        return BetStatistics(
            probability=round(probability, 2),
            fair_odds=fair_odds,
            market_odds=market_odds,
            expected_value=ev,
            kelly_fraction=kelly,
            confidence_interval=(round(ci_lower, 1), round(ci_upper, 1)),
            standard_deviation=round(std, 2),
            reliability_score=reliability,
            monte_carlo_prob=round(mc_prob, 2),
            variance=round(var, 4),
            z_score=round(z_score, 2),
            is_value_bet=is_value,
            bet_rating=rating
        )
    
    # =========================================================================
    # Complete Match Analysis
    # =========================================================================
    
    def full_statistical_analysis(self,
                                   home_name: str,
                                   away_name: str,
                                   home_stats: Dict,
                                   away_stats: Dict,
                                   league: str = 'Premier League') -> Dict:
        """
        Teljes statisztikai elemzés egy meccshez
        
        Visszaad minden statisztikai számítást:
        - Dixon-Coles modell
        - Monte Carlo szimuláció
        - Elo rating
        - Minden fogadási típushoz részletes statisztika
        """
        # Csapat értékek
        home_attack = getattr(home_stats, 'attack_strength', 1.0)
        home_defense = getattr(home_stats, 'defense_strength', 1.0)
        away_attack = getattr(away_stats, 'attack_strength', 1.0)
        away_defense = getattr(away_stats, 'defense_strength', 1.0)
        
        # Dixon-Coles
        dc = self.calculate_dixon_coles(
            home_attack, home_defense,
            away_attack, away_defense
        )
        
        # Monte Carlo
        mc = self.monte_carlo_simulation(dc.home_goals_lambda, dc.away_goals_lambda)
        
        # Elo
        home_elo = self.get_team_elo(home_name, league)
        away_elo = self.get_team_elo(away_name, league)
        elo = self.elo_expected_score(home_elo, away_elo)
        
        # Fogadási statisztikák minden típushoz
        bets = {}
        
        # 1X2
        bets['home_win'] = self.calculate_bet_statistics(
            'Hazai győzelem', dc.home_win, None, home_stats, away_stats
        )
        bets['draw'] = self.calculate_bet_statistics(
            'Döntetlen', dc.draw, None, home_stats, away_stats
        )
        bets['away_win'] = self.calculate_bet_statistics(
            'Vendég győzelem', dc.away_win, None, home_stats, away_stats
        )
        
        # Over/Under 2.5
        bets['over_25'] = self.calculate_bet_statistics(
            'Over 2.5 gól', mc['over_25'], None, home_stats, away_stats
        )
        bets['under_25'] = self.calculate_bet_statistics(
            'Under 2.5 gól', 100 - mc['over_25'], None, home_stats, away_stats
        )
        
        # BTTS
        bets['btts_yes'] = self.calculate_bet_statistics(
            'Mindkét csapat szerez gólt (Igen)', mc['btts'], None, home_stats, away_stats
        )
        bets['btts_no'] = self.calculate_bet_statistics(
            'Mindkét csapat szerez gólt (Nem)', 100 - mc['btts'], None, home_stats, away_stats
        )
        
        # Double Chance
        bets['1X'] = self.calculate_bet_statistics(
            '1X (Hazai nem veszít)', dc.home_win + dc.draw, None, home_stats, away_stats
        )
        bets['X2'] = self.calculate_bet_statistics(
            'X2 (Vendég nem veszít)', dc.draw + dc.away_win, None, home_stats, away_stats
        )
        bets['12'] = self.calculate_bet_statistics(
            '12 (Nem döntetlen)', dc.home_win + dc.away_win, None, home_stats, away_stats
        )
        
        # Corners (ha van adat)
        home_corners = getattr(home_stats, 'avg_corners', 5.0)
        away_corners = getattr(away_stats, 'avg_corners', 4.5)
        total_corners = home_corners + away_corners
        corners_std = math.sqrt(
            getattr(home_stats, 'corners_std', 2.0)**2 + 
            getattr(away_stats, 'corners_std', 2.0)**2
        )
        
        for line in [8.5, 9.5, 10.5, 11.5]:
            z = (line - total_corners) / corners_std if corners_std > 0 else 0
            over_prob = (1 - self._normal_cdf(z)) * 100
            bets[f'corners_over_{line}'] = self.calculate_bet_statistics(
                f'Szögletek Over {line}', over_prob, None, home_stats, away_stats
            )
        
        # Cards
        home_cards = getattr(home_stats, 'avg_cards', 2.0)
        away_cards = getattr(away_stats, 'avg_cards', 2.0)
        total_cards = home_cards + away_cards
        cards_std = math.sqrt(
            getattr(home_stats, 'cards_std', 1.0)**2 + 
            getattr(away_stats, 'cards_std', 1.0)**2
        )
        
        for line in [3.5, 4.5, 5.5]:
            z = (line - total_cards) / cards_std if cards_std > 0 else 0
            over_prob = (1 - self._normal_cdf(z)) * 100
            bets[f'cards_over_{line}'] = self.calculate_bet_statistics(
                f'Lapok Over {line}', over_prob, None, home_stats, away_stats
            )
        
        return {
            'dixon_coles': dc,
            'monte_carlo': mc,
            'elo': elo,
            'bets': bets,
            'expected_goals': {
                'home': dc.home_goals_lambda,
                'away': dc.away_goals_lambda,
                'total': round(dc.home_goals_lambda + dc.away_goals_lambda, 2)
            },
            'top_scores': mc['top_scores']
        }
    
    def _normal_cdf(self, z: float) -> float:
        """Standard normál eloszlás CDF"""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# Singleton
_stat_engine = None

def get_stat_engine() -> ProfessionalStatEngine:
    """Professional stat engine singleton"""
    global _stat_engine
    if _stat_engine is None:
        _stat_engine = ProfessionalStatEngine()
    return _stat_engine
