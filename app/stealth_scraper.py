"""
Stealth Web Scraper - Cloudflare Bypass
eredmenyek.com és footballdatabase.eu adatforrásokkal
Anti-detection technikák: random delays, real browser, headful mode
"""
import asyncio
import random
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
from bs4 import BeautifulSoup
import re

# Playwright lazy import (may not be installed)
try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("[SCRAPER] Playwright not available, install with: pip install playwright && playwright install chromium")


class StealthScraper:
    """Stealth web scraper with Cloudflare bypass techniques"""
    
    # Liga URL-ek az eredmenyek.com-on
    EREDMENYEK_LEAGUES = {
        'premier-league': 'https://www.eredmenyek.com/labdarugas/anglia/premier-league/',
        'la-liga': 'https://www.eredmenyek.com/labdarugas/spanyolorszag/laliga/',
        'bundesliga': 'https://www.eredmenyek.com/labdarugas/nemetorszag/bundesliga/',
        'serie-a': 'https://www.eredmenyek.com/labdarugas/olaszorszag/serie-a/',
        'ligue-1': 'https://www.eredmenyek.com/labdarugas/franciaorszag/ligue-1/',
        'champions-league': 'https://www.eredmenyek.com/labdarugas/europa/bajnokok-ligaja/',
    }
    
    # Alternative: footballdatabase.eu
    FOOTBALLDB_LEAGUES = {
        'premier-league': 'https://www.footballdatabase.eu/en/championship/england-premier_league',
        'la-liga': 'https://www.footballdatabase.eu/en/championship/spain-la_liga',
        'bundesliga': 'https://www.footballdatabase.eu/en/championship/germany-bundesliga',
        'serie-a': 'https://www.footballdatabase.eu/en/championship/italy-serie_a',
        'ligue-1': 'https://www.footballdatabase.eu/en/championship/france-ligue_1',
        'champions-league': 'https://www.footballdatabase.eu/en/championship/europe-uefa_champions_league',
    }
    
    # Realistic User Agents (Chrome on Windows)
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]
    
    def __init__(self, config):
        self.config = config
        self.data_path = config.SCRAPED_DATA_PATH
        self._ensure_data_dir()
        
    def _ensure_data_dir(self):
        """Create data directory if not exists"""
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _random_delay(self, min_sec: float = 1.0, max_sec: float = 3.0) -> float:
        """Generate random delay to appear human"""
        return random.uniform(min_sec, max_sec)
    
    def _get_random_ua(self) -> str:
        """Get random user agent"""
        return random.choice(self.USER_AGENTS)
    
    async def _create_stealth_browser(self, playwright):
        """Create browser with stealth settings"""
        browser = await playwright.chromium.launch(
            headless=False,  # Headful mode bypasses more detection
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-infobars',
                '--disable-extensions',
                '--window-size=1920,1080',
            ]
        )
        
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent=self._get_random_ua(),
            locale='hu-HU',
            timezone_id='Europe/Budapest',
        )
        
        # Add stealth scripts to avoid detection
        await context.add_init_script("""
            // Override navigator.webdriver
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            // Override plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            
            // Override languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['hu-HU', 'hu', 'en-US', 'en'],
            });
            
            // Chrome specific
            window.chrome = {
                runtime: {},
            };
        """)
        
        return browser, context

    async def scrape_eredmenyek(self) -> List[Dict]:
        """Scrape matches from eredmenyek.com with stealth"""
        if not PLAYWRIGHT_AVAILABLE:
            print("[SCRAPER] Playwright not available")
            return []
        
        all_fixtures = []
        
        async with async_playwright() as p:
            browser, context = await self._create_stealth_browser(p)
            page = await context.new_page()
            
            try:
                for league_key, url in self.EREDMENYEK_LEAGUES.items():
                    print(f"[SCRAPER] Scraping {league_key} from eredmenyek.com...")
                    
                    # Random delay before each request
                    await asyncio.sleep(self._random_delay(2, 5))
                    
                    try:
                        # Navigate with extended timeout
                        await page.goto(url, wait_until='networkidle', timeout=30000)
                        
                        # Wait for dynamic content
                        await asyncio.sleep(self._random_delay(1, 2))
                        
                        # Scroll to trigger lazy loading
                        await page.evaluate("window.scrollBy(0, 500)")
                        await asyncio.sleep(0.5)
                        
                        # Get page content
                        content = await page.content()
                        fixtures = self._parse_eredmenyek_html(content, league_key)
                        all_fixtures.extend(fixtures)
                        
                        print(f"[SCRAPER] Found {len(fixtures)} matches for {league_key}")
                        
                    except PlaywrightTimeout:
                        print(f"[SCRAPER] Timeout for {league_key}, trying next...")
                        continue
                    except Exception as e:
                        print(f"[SCRAPER] Error scraping {league_key}: {e}")
                        continue
                        
            finally:
                await browser.close()
        
        return all_fixtures
    
    def _parse_eredmenyek_html(self, html: str, league_key: str) -> List[Dict]:
        """Parse eredmenyek.com HTML to extract match data"""
        fixtures = []
        soup = BeautifulSoup(html, 'html.parser')
        
        # League name mapping
        league_names = {
            'premier-league': 'Premier League',
            'la-liga': 'La Liga',
            'bundesliga': 'Bundesliga',
            'serie-a': 'Serie A',
            'ligue-1': 'Ligue 1',
            'champions-league': 'Champions League',
        }
        
        # Find match rows - eredmenyek.com uses specific class patterns
        # Try multiple selectors as the site structure may vary
        selectors = [
            'div.event__match',
            'div[class*="sportName soccer"]',
            'div.event__match--scheduled',
            'div.event__match--live',
            'div.event__match--finished'
        ]
        
        matches_found = []
        for selector in selectors:
            matches_found = soup.select(selector)
            if matches_found:
                break
        
        # Fallback: look for any element with match-like structure
        if not matches_found:
            # Look for table rows with team names
            matches_found = soup.find_all('div', class_=re.compile(r'event'))
        
        for match_el in matches_found[:20]:  # Limit to 20 per league
            try:
                # Extract home team
                home_el = match_el.select_one('.event__participant--home, .event__homeParticipant')
                away_el = match_el.select_one('.event__participant--away, .event__awayParticipant')
                
                if not home_el or not away_el:
                    continue
                
                home_team = home_el.get_text(strip=True)
                away_team = away_el.get_text(strip=True)
                
                # Extract scores if available
                home_score = None
                away_score = None
                score_els = match_el.select('.event__score--home, .event__score--away')
                if len(score_els) >= 2:
                    try:
                        home_score = int(score_els[0].get_text(strip=True))
                        away_score = int(score_els[1].get_text(strip=True))
                    except ValueError:
                        pass
                
                # Extract time
                time_el = match_el.select_one('.event__time')
                match_time = time_el.get_text(strip=True) if time_el else '00:00'
                
                # Create fixture object
                fixture = {
                    'id': f"ered_{league_key}_{hash(home_team + away_team) % 100000}",
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'match_time': match_time[:5] if len(match_time) >= 5 else match_time,
                    'match_date': datetime.now().strftime('%Y-%m-%d'),
                    'timestamp': datetime.now().isoformat(),
                    'league': league_names.get(league_key, league_key),
                    'country': 'Unknown',
                    'status': 'Scheduled' if home_score is None else 'Finished',
                    'source': 'eredmenyek.com'
                }
                
                fixtures.append(fixture)
                
            except Exception as e:
                print(f"[PARSE] Error parsing match: {e}")
                continue
        
        return fixtures

    async def scrape_footballdb(self) -> List[Dict]:
        """Fallback: scrape from footballdatabase.eu"""
        if not PLAYWRIGHT_AVAILABLE:
            return []
        
        all_fixtures = []
        
        async with async_playwright() as p:
            browser, context = await self._create_stealth_browser(p)
            page = await context.new_page()
            
            try:
                for league_key, url in self.FOOTBALLDB_LEAGUES.items():
                    print(f"[SCRAPER] Scraping {league_key} from footballdatabase.eu...")
                    
                    await asyncio.sleep(self._random_delay(2, 4))
                    
                    try:
                        await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                        await asyncio.sleep(self._random_delay(1, 2))
                        
                        content = await page.content()
                        fixtures = self._parse_footballdb_html(content, league_key)
                        all_fixtures.extend(fixtures)
                        
                        print(f"[SCRAPER] Found {len(fixtures)} matches for {league_key}")
                        
                    except Exception as e:
                        print(f"[SCRAPER] Error: {e}")
                        continue
                        
            finally:
                await browser.close()
        
        return all_fixtures
    
    def _parse_footballdb_html(self, html: str, league_key: str) -> List[Dict]:
        """Parse footballdatabase.eu HTML"""
        fixtures = []
        soup = BeautifulSoup(html, 'html.parser')
        
        league_names = {
            'premier-league': 'Premier League',
            'la-liga': 'La Liga', 
            'bundesliga': 'Bundesliga',
            'serie-a': 'Serie A',
            'ligue-1': 'Ligue 1',
            'champions-league': 'Champions League',
        }
        
        # footballdatabase.eu uses tables for matches
        match_rows = soup.select('table.matchs tr, div.match-item')
        
        for row in match_rows[:20]:
            try:
                # Try to find team names
                team_els = row.select('td.team, a.team, span.team-name')
                if len(team_els) >= 2:
                    home_team = team_els[0].get_text(strip=True)
                    away_team = team_els[1].get_text(strip=True)
                    
                    fixture = {
                        'id': f"fdb_{league_key}_{hash(home_team + away_team) % 100000}",
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': None,
                        'away_score': None,
                        'match_time': '15:00',
                        'match_date': datetime.now().strftime('%Y-%m-%d'),
                        'timestamp': datetime.now().isoformat(),
                        'league': league_names.get(league_key, league_key),
                        'country': 'Unknown',
                        'status': 'Scheduled',
                        'source': 'footballdatabase.eu'
                    }
                    fixtures.append(fixture)
                    
            except Exception as e:
                continue
        
        return fixtures

    async def refresh_all_data(self) -> Dict:
        """Main method: try eredmenyek.com first, fallback to footballdb"""
        result = {
            'success': False,
            'fixtures': [],
            'message': '',
            'timestamp': datetime.now().isoformat()
        }
        
        print("[SCRAPER] Starting stealth scrape...")
        
        # Try eredmenyek.com first
        fixtures = await self.scrape_eredmenyek()
        
        # If failed, try footballdatabase.eu
        if not fixtures:
            print("[SCRAPER] Eredmenyek.com failed, trying footballdatabase.eu...")
            fixtures = await self.scrape_footballdb()
        
        if fixtures:
            # Save to cache
            data = {
                'last_scrape_at': datetime.now().isoformat(),
                'fixtures': fixtures,
                'teams': [],
                'stats': {
                    'total_fixtures': len(fixtures),
                    'total_teams': len(set(f['home_team'] for f in fixtures) | set(f['away_team'] for f in fixtures)),
                    'total_leagues': len(set(f['league'] for f in fixtures))
                }
            }
            
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            result['success'] = True
            result['fixtures'] = fixtures
            result['message'] = f"Scraped {len(fixtures)} fixtures successfully"
        else:
            result['message'] = "Failed to scrape from all sources"
        
        return result
    
    def get_cached_data(self) -> Optional[Dict]:
        """Get cached scraped data"""
        if self.data_path.exists():
            try:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def get_fixtures(self, league: str = None) -> List[Dict]:
        """Get fixtures, optionally filtered by league"""
        data = self.get_cached_data()
        if not data:
            return []
        
        fixtures = data.get('fixtures', [])
        
        if league:
            fixtures = [f for f in fixtures if league.lower() in f.get('league', '').lower()]
        
        return fixtures
    
    def get_status(self) -> Dict:
        """Get scraper status"""
        data = self.get_cached_data()
        
        if not data:
            return {
                'status': 'inactive',
                'last_scrape_at': None,
                'fixture_count': 0,
                'team_count': 0,
                'needs_refresh': True,
                'next_refresh_hours': 0,
                'error_message': None
            }
        
        stats = data.get('stats', {})
        
        return {
            'status': 'active',
            'last_scrape_at': data.get('last_scrape_at'),
            'fixture_count': stats.get('total_fixtures', 0),
            'team_count': stats.get('total_teams', 0),
            'leagues_count': stats.get('total_leagues', 0),
            'needs_refresh': False,
            'next_refresh_hours': 24,
            'error_message': None
        }


# Sync wrappers for Flask
def get_stealth_scraper(config):
    """Factory function to create stealth scraper"""
    return StealthScraper(config)


def refresh_data_sync(scraper: StealthScraper) -> Dict:
    """Sync wrapper for refresh"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(scraper.refresh_all_data())
        loop.close()
        return result
    except Exception as e:
        return {'success': False, 'message': str(e)}
