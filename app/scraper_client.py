"""
Football Data Scraper - Stealth Web Scraping alapú
eredmenyek.com és footballdatabase.eu használatával
Anti-detection technikák Cloudflare bypass-hoz
"""
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

from .config import get_config

# Try to import stealth scraper
try:
    from .stealth_scraper import StealthScraper, refresh_data_sync, PLAYWRIGHT_AVAILABLE
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    StealthScraper = None


class FootballScraper:
    """Labdarúgás adatgyűjtő - Web Scraping stealth technikákkal"""
    
    def __init__(self, db):
        self.config = get_config()
        self.db = db
        self.data_path = self.config.SCRAPED_DATA_PATH
        self._ensure_data_dir()
        
        # Initialize stealth scraper if available
        if PLAYWRIGHT_AVAILABLE and StealthScraper:
            self.stealth_scraper = StealthScraper(self.config)
        else:
            self.stealth_scraper = None
            print("[SCRAPER] Warning: Playwright not available, stealth scraping disabled")
    
    def _ensure_data_dir(self):
        """Adat könyvtár létrehozása"""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
    
    def should_refresh_data(self) -> bool:
        """Ellenőrizi, szükséges-e az adatfrissítés"""
        if not self.data_path.exists():
            return True
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            last_scrape = data.get('last_scrape_at')
            if not last_scrape:
                return True
            
            last_scrape_dt = datetime.fromisoformat(last_scrape)
            hours_since = (datetime.now() - last_scrape_dt).total_seconds() / 3600
            
            return hours_since >= self.config.SCRAPE_INTERVAL_HOURS
            
        except (json.JSONDecodeError, KeyError, ValueError):
            return True
    
    def get_cached_data(self) -> Optional[Dict]:
        """Cache-elt adatok lekérése"""
        if self.data_path.exists():
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def refresh_data(self) -> Dict:
        """Adatok frissítése stealth web scraping-gel"""
        if not self.stealth_scraper:
            return {
                'success': False,
                'message': 'Stealth scraper not available. Install playwright: pip install playwright && playwright install chromium',
                'fixtures': []
            }
        
        try:
            # Use async event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.stealth_scraper.refresh_all_data())
            loop.close()
            return result
        except Exception as e:
            return {
                'success': False,
                'message': f'Scraping failed: {str(e)}',
                'fixtures': []
            }
    
    def get_fixtures(self, league: str = None, limit: int = None) -> List[Dict]:
        """Meccsek lekérése a cache-ből"""
        data = self.get_cached_data()
        if not data:
            return []
        
        fixtures = data.get('fixtures', [])
        
        # Filter by league if specified
        if league:
            league_lower = league.lower().replace('-', ' ')
            fixtures = [
                f for f in fixtures 
                if league_lower in f.get('league', '').lower() or
                   league.lower() in f.get('league', '').lower().replace(' ', '-')
            ]
        
        # Apply limit
        if limit and limit > 0:
            fixtures = fixtures[:limit]
        
        return fixtures
    
    def get_upcoming_fixtures(self, limit: int = 20) -> List[Dict]:
        """Közelgő meccsek lekérése"""
        data = self.get_cached_data()
        if not data:
            return []
        
        fixtures = data.get('fixtures', [])
        
        # Filter upcoming (no score yet)
        upcoming = [f for f in fixtures if f.get('home_score') is None]
        
        # Sort by date
        upcoming.sort(key=lambda x: x.get('match_date', ''))
        
        return upcoming[:limit]
    
    def get_live_fixtures(self) -> List[Dict]:
        """Élő meccsek lekérése"""
        data = self.get_cached_data()
        if not data:
            return []
        
        fixtures = data.get('fixtures', [])
        
        # Filter live matches
        live = [f for f in fixtures if f.get('status') in ['1H', '2H', 'HT', 'Live', 'LIVE']]
        
        return live
    
    def get_league_table(self, league: str) -> List[Dict]:
        """Liga tabella lekérése (ha elérhető)"""
        data = self.get_cached_data()
        if not data:
            return []
        
        # Tables stored separately if scraped
        tables = data.get('tables', {})
        return tables.get(league, [])
    
    def get_teams(self, league: str = None) -> List[Dict]:
        """Csapatok lekérése"""
        data = self.get_cached_data()
        if not data:
            return []
        
        # Build team list from fixtures
        fixtures = data.get('fixtures', [])
        teams_set = set()
        
        for f in fixtures:
            if not league or league.lower() in f.get('league', '').lower():
                teams_set.add(f.get('home_team', ''))
                teams_set.add(f.get('away_team', ''))
        
        teams_set.discard('')
        
        return [{'name': t, 'league': league or 'Unknown'} for t in sorted(teams_set)]
    
    def get_status(self) -> Dict:
        """Scraper státusz lekérése"""
        data = self.get_cached_data()
        
        if not data:
            return {
                'status': 'inactive',
                'last_scrape_at': None,
                'fixture_count': 0,
                'team_count': 0,
                'needs_refresh': True,
                'next_refresh_hours': 0,
                'error_message': None,
                'scraper_available': PLAYWRIGHT_AVAILABLE
            }
        
        last_scrape = data.get('last_scrape_at')
        stats = data.get('stats', {})
        
        # Calculate next refresh time
        next_refresh_hours = self.config.SCRAPE_INTERVAL_HOURS
        if last_scrape:
            try:
                last_scrape_dt = datetime.fromisoformat(last_scrape)
                hours_since = (datetime.now() - last_scrape_dt).total_seconds() / 3600
                next_refresh_hours = max(0, self.config.SCRAPE_INTERVAL_HOURS - hours_since)
            except:
                pass
        
        return {
            'status': 'active',
            'last_scrape_at': last_scrape,
            'fixture_count': stats.get('total_fixtures', 0),
            'team_count': stats.get('total_teams', 0),
            'leagues_count': stats.get('total_leagues', 0),
            'needs_refresh': self.should_refresh_data(),
            'next_refresh_hours': next_refresh_hours,
            'error_message': None,
            'scraper_available': PLAYWRIGHT_AVAILABLE
        }
    
    def get_match_by_id(self, match_id: str) -> Optional[Dict]:
        """Egy meccs lekérése ID alapján"""
        data = self.get_cached_data()
        if not data:
            return None
        
        fixtures = data.get('fixtures', [])
        for f in fixtures:
            if str(f.get('id')) == str(match_id):
                return f
        
        return None
    
    def search_fixtures(self, query: str) -> List[Dict]:
        """Meccsek keresése csapatnév alapján"""
        data = self.get_cached_data()
        if not data:
            return []
        
        fixtures = data.get('fixtures', [])
        query_lower = query.lower()
        
        return [
            f for f in fixtures
            if query_lower in f.get('home_team', '').lower() or
               query_lower in f.get('away_team', '').lower()
        ]


# Factory function for Flask app
def create_scraper(db=None):
    """Create FootballScraper instance"""
    return FootballScraper(db)
