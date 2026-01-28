"""
Gemini AI Analyzer for Tippmix AI Assistant
============================================
Google Gemini API integráció a mérkőzések elemzéséhez.
Természetes nyelvi elemzés és intelligens tipp generálás.
"""

import os
from typing import Dict, Optional, List
from dataclasses import dataclass

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[Gemini] Warning: google-generativeai not installed. Run: pip install google-generativeai")


@dataclass
class GeminiAnalysis:
    """Gemini AI elemzés eredménye"""
    prediction: str  # '1', 'X', '2'
    confidence: int  # 1-5
    key_factors: List[str]
    risks: List[str]
    recommendation: str
    corners_tip: Optional[str] = None
    goals_tip: Optional[str] = None
    cards_tip: Optional[str] = None
    raw_response: str = ""


class GeminiAnalyzer:
    """
    Google Gemini API alapú elemző motor.
    
    Használathoz szükséges:
    1. pip install google-generativeai
    2. GEMINI_API_KEY környezeti változó beállítása
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model = None
        self.is_configured = False
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.is_configured = True
                print("[Gemini] Successfully configured")
            except Exception as e:
                print(f"[Gemini] Configuration error: {e}")
        elif not GEMINI_AVAILABLE:
            print("[Gemini] Library not installed")
        else:
            print("[Gemini] No API key provided")
    
    def analyze_match(self, 
                      home_team: str, 
                      away_team: str,
                      home_stats: Dict = None,
                      away_stats: Dict = None,
                      prediction_data: Dict = None,
                      league: str = "Premier League") -> Optional[GeminiAnalysis]:
        """
        AI elemzés a mérkőzéshez Gemini API-val.
        
        Args:
            home_team: Hazai csapat neve
            away_team: Vendég csapat neve
            home_stats: Hazai csapat statisztikái
            away_stats: Vendég csapat statisztikái
            prediction_data: Poisson/ML predikciós adatok
            league: Liga neve
            
        Returns:
            GeminiAnalysis objektum vagy None ha nincs API
        """
        if not self.is_configured:
            return self._fallback_analysis(home_team, away_team, home_stats, away_stats)
        
        # Prompt összeállítása
        prompt = self._build_prompt(
            home_team, away_team, 
            home_stats, away_stats, 
            prediction_data, league
        )
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_response(response.text, home_team, away_team)
        except Exception as e:
            print(f"[Gemini] API error: {e}")
            return self._fallback_analysis(home_team, away_team, home_stats, away_stats)
    
    def _build_prompt(self, 
                      home_team: str, 
                      away_team: str,
                      home_stats: Dict,
                      away_stats: Dict,
                      prediction_data: Dict,
                      league: str) -> str:
        """Prompt összeállítása az AI-nak"""
        
        # Statisztikák kinyerése
        home_attack = getattr(home_stats, 'attack_strength', 1.0) if home_stats else 1.0
        home_defense = getattr(home_stats, 'defense_strength', 1.0) if home_stats else 1.0
        home_form = getattr(home_stats, 'form_index', 50) if home_stats else 50
        home_corners = getattr(home_stats, 'avg_corners', 5.0) if home_stats else 5.0
        
        away_attack = getattr(away_stats, 'attack_strength', 1.0) if away_stats else 1.0
        away_defense = getattr(away_stats, 'defense_strength', 1.0) if away_stats else 1.0
        away_form = getattr(away_stats, 'form_index', 50) if away_stats else 50
        away_corners = getattr(away_stats, 'avg_corners', 5.0) if away_stats else 5.0
        
        # Predikciós adatok
        home_prob = prediction_data.home_win_prob if prediction_data else 33
        draw_prob = prediction_data.draw_prob if prediction_data else 33
        away_prob = prediction_data.away_win_prob if prediction_data else 33
        
        prompt = f"""
Te egy professzionális sportfogadási elemző vagy. Elemezd a következő labdarúgó mérkőzést magyarul.

**Mérkőzés:** {home_team} vs {away_team}
**Liga:** {league}

**Hazai csapat ({home_team}) statisztikái:**
- Támadóerő: {home_attack:.2f}x (1.0 = liga átlag)
- Védekezés: {home_defense:.2f}x
- Forma-index: {home_form:.0f}/100
- Átlag szögletek: {home_corners:.1f}/meccs

**Vendég csapat ({away_team}) statisztikái:**
- Támadóerő: {away_attack:.2f}x
- Védekezés: {away_defense:.2f}x
- Forma-index: {away_form:.0f}/100
- Átlag szögletek: {away_corners:.1f}/meccs

**Matematikai modell predikció:**
- Hazai győzelem: {home_prob}%
- Döntetlen: {draw_prob}%
- Vendég győzelem: {away_prob}%

Kérlek válaszolj a következő formátumban:

PREDIKCIÓ: [1/X/2]
BIZALOM: [1-5]
KULCS_TÉNYEZŐK:
- [tényező 1]
- [tényező 2]
- [tényező 3]
KOCKÁZATOK:
- [kockázat 1]
- [kockázat 2]
SZÖGLETEK_TIPP: [Over/Under X.5 - rövid indoklás]
GÓLOK_TIPP: [Over/Under 2.5 - rövid indoklás]
LAPOK_TIPP: [Over/Under 4.5 - rövid indoklás vagy "Nincs erős jel"]
ÖSSZEFOGLALÓ: [1-2 mondat összefoglaló ajánlás]
"""
        return prompt
    
    def _parse_response(self, 
                        response_text: str, 
                        home_team: str,
                        away_team: str) -> GeminiAnalysis:
        """AI válasz feldolgozása strukturált formátumba"""
        
        lines = response_text.strip().split('\n')
        
        prediction = '1'
        confidence = 3
        key_factors = []
        risks = []
        corners_tip = None
        goals_tip = None
        cards_tip = None
        recommendation = ""
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('PREDIKCIÓ:'):
                pred = line.replace('PREDIKCIÓ:', '').strip()
                if pred in ['1', 'X', '2']:
                    prediction = pred
                elif '1' in pred:
                    prediction = '1'
                elif 'X' in pred or 'döntetlen' in pred.lower():
                    prediction = 'X'
                elif '2' in pred:
                    prediction = '2'
                    
            elif line.startswith('BIZALOM:'):
                try:
                    conf = int(line.replace('BIZALOM:', '').strip()[0])
                    confidence = max(1, min(5, conf))
                except:
                    pass
                    
            elif line.startswith('KULCS_TÉNYEZŐK:'):
                current_section = 'factors'
                
            elif line.startswith('KOCKÁZATOK:'):
                current_section = 'risks'
                
            elif line.startswith('SZÖGLETEK_TIPP:'):
                corners_tip = line.replace('SZÖGLETEK_TIPP:', '').strip()
                current_section = None
                
            elif line.startswith('GÓLOK_TIPP:'):
                goals_tip = line.replace('GÓLOK_TIPP:', '').strip()
                current_section = None
                
            elif line.startswith('LAPOK_TIPP:'):
                cards_tip = line.replace('LAPOK_TIPP:', '').strip()
                current_section = None
                
            elif line.startswith('ÖSSZEFOGLALÓ:'):
                recommendation = line.replace('ÖSSZEFOGLALÓ:', '').strip()
                current_section = None
                
            elif line.startswith('- ') and current_section:
                item = line[2:].strip()
                if current_section == 'factors':
                    key_factors.append(item)
                elif current_section == 'risks':
                    risks.append(item)
        
        return GeminiAnalysis(
            prediction=prediction,
            confidence=confidence,
            key_factors=key_factors or ["Nincs részletes elemzés"],
            risks=risks or ["Általános fogadási kockázat"],
            recommendation=recommendation or f"Ajánlott tipp: {prediction}",
            corners_tip=corners_tip,
            goals_tip=goals_tip,
            cards_tip=cards_tip,
            raw_response=response_text
        )
    
    def _fallback_analysis(self,
                           home_team: str,
                           away_team: str,
                           home_stats: Dict,
                           away_stats: Dict) -> GeminiAnalysis:
        """Fallback elemzés ha nincs Gemini API"""
        
        # Egyszerű logika az adatok alapján
        home_strength = getattr(home_stats, 'attack_strength', 1.0) if home_stats else 1.0
        away_strength = getattr(away_stats, 'attack_strength', 1.0) if away_stats else 1.0
        
        if home_strength > away_strength * 1.15:
            prediction = '1'
            confidence = 4
        elif away_strength > home_strength * 1.05:
            prediction = '2'
            confidence = 3
        else:
            prediction = 'X'
            confidence = 2
        
        return GeminiAnalysis(
            prediction=prediction,
            confidence=confidence,
            key_factors=[
                f"{home_team} hazai pálya előnye",
                "Forma és statisztikák alapján",
                "Matematikai modell predikció"
            ],
            risks=[
                "Gemini API nem elérhető - korlátozott elemzés",
                "Valós idejű információk hiánya"
            ],
            recommendation=f"Alapvető statisztikai elemzés alapján: {prediction}",
            corners_tip=None,
            goals_tip=None,
            cards_tip=None,
            raw_response="[Fallback analysis - no Gemini API]"
        )
    
    def get_tip_confidence_boost(self, 
                                  tip_type: str,
                                  base_probability: float,
                                  gemini_analysis: GeminiAnalysis) -> Dict:
        """
        Kombinálja a matematikai modell eredményét a Gemini elemzéssel.
        
        Returns:
            Dict with boosted probability and reasoning
        """
        if not gemini_analysis:
            return {
                'boosted_probability': base_probability,
                'ai_boost': 0,
                'reasoning': "Nincs AI elemzés"
            }
        
        # Ha Gemini egyetért, növeljük a konfidenciát
        ai_confidence = gemini_analysis.confidence / 5  # 0-1 skála
        boost = 0
        reasoning = ""
        
        if tip_type == 'corners' and gemini_analysis.corners_tip:
            if 'Over' in (gemini_analysis.corners_tip or ''):
                boost = 5 * ai_confidence
                reasoning = f"AI megerősíti: {gemini_analysis.corners_tip}"
        elif tip_type == 'goals' and gemini_analysis.goals_tip:
            if 'Over' in (gemini_analysis.goals_tip or ''):
                boost = 5 * ai_confidence
                reasoning = f"AI megerősíti: {gemini_analysis.goals_tip}"
        elif tip_type == '1X2':
            boost = 3 * ai_confidence
            reasoning = f"AI predikció: {gemini_analysis.prediction} ({gemini_analysis.confidence}/5)"
        
        return {
            'boosted_probability': min(95, base_probability + boost),
            'ai_boost': boost,
            'reasoning': reasoning
        }


# Singleton instance
_gemini_analyzer = None

def get_gemini_analyzer() -> GeminiAnalyzer:
    """Gemini analyzer singleton"""
    global _gemini_analyzer
    if _gemini_analyzer is None:
        _gemini_analyzer = GeminiAnalyzer()
    return _gemini_analyzer
