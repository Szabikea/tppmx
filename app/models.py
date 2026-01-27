"""
Tippmix AI Segéd - SQLite Models
================================
Adatbázis modellek a cache és scraping adatok tárolásához.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any, Dict


class Database:
    """SQLite adatbázis kezelő"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Kapcsolat létrehozása az adatbázishoz"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Táblák létrehozása ha nem léteznek"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Cache tábla - Scraped adatok tárolása
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cached_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                params_hash TEXT NOT NULL,
                response_json TEXT NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                UNIQUE(endpoint, params_hash)
            )
        """)
        
        # Scraping státusz nyomon követése
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scrape_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                last_scrape_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,
                error_message TEXT,
                fixtures_count INTEGER DEFAULT 0,
                teams_count INTEGER DEFAULT 0
            )
        """)
        
        # Scraped csapat statisztikák
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS team_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT NOT NULL,
                avg_corners REAL DEFAULT 0,
                avg_yellow_cards REAL DEFAULT 0,
                avg_shots_on_target REAL DEFAULT 0,
                avg_goals REAL DEFAULT 0,
                matches_analyzed INTEGER DEFAULT 0,
                confidence_score REAL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(team_name)
            )
        """)
        
        # Scraped meccsek
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scraped_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                match_date TEXT,
                match_time TEXT,
                league TEXT,
                match_url TEXT,
                corners_home INTEGER DEFAULT 0,
                corners_away INTEGER DEFAULT 0,
                yellow_cards_home INTEGER DEFAULT 0,
                yellow_cards_away INTEGER DEFAULT 0,
                shots_on_target_home INTEGER DEFAULT 0,
                shots_on_target_away INTEGER DEFAULT 0,
                goals_home INTEGER DEFAULT 0,
                goals_away INTEGER DEFAULT 0,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index a gyorsabb kereséshez
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_cache_lookup 
            ON cached_responses(endpoint, params_hash, expires_at)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_team_name 
            ON team_stats(team_name)
        """)
        
        conn.commit()
        conn.close()
    
    # =========================================================================
    # Cache műveletek
    # =========================================================================
    
    def get_cached_response(self, endpoint: str, params_hash: str) -> Optional[dict]:
        """Cache-elt válasz lekérése, ha még érvényes."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT response_json FROM cached_responses
            WHERE endpoint = ? AND params_hash = ? AND expires_at > datetime('now')
        """, (endpoint, params_hash))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row["response_json"])
        return None
    
    def save_cached_response(self, endpoint: str, params_hash: str, 
                            response: dict, ttl_hours: int = 168):
        """Válasz mentése cache-be (alapértelmezett: 1 hét)."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        cursor.execute("""
            INSERT OR REPLACE INTO cached_responses 
            (endpoint, params_hash, response_json, cached_at, expires_at)
            VALUES (?, ?, ?, datetime('now'), ?)
        """, (endpoint, params_hash, json.dumps(response), expires_at.isoformat()))
        
        conn.commit()
        conn.close()
    
    def clear_expired_cache(self) -> int:
        """Lejárt cache bejegyzések törlése"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM cached_responses WHERE expires_at < datetime('now')
        """)
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted
    
    # =========================================================================
    # Scraping státusz
    # =========================================================================
    
    def update_scrape_status(self, status: str, error_message: str = None,
                            fixtures_count: int = 0, teams_count: int = 0):
        """Scraping státusz frissítése"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO scrape_status (status, error_message, fixtures_count, teams_count)
            VALUES (?, ?, ?, ?)
        """, (status, error_message, fixtures_count, teams_count))
        
        conn.commit()
        conn.close()
    
    def get_scrape_status(self) -> Dict:
        """Legutóbbi scraping státusz lekérése"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM scrape_status ORDER BY id DESC LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'last_scrape_at': row['last_scrape_at'],
                'status': row['status'],
                'error_message': row['error_message'],
                'fixtures_count': row['fixtures_count'],
                'teams_count': row['teams_count']
            }
        
        return {
            'last_scrape_at': None,
            'status': 'never_run',
            'error_message': None,
            'fixtures_count': 0,
            'teams_count': 0
        }
    
    # =========================================================================
    # Csapat statisztikák
    # =========================================================================
    
    def save_team_stats(self, team_name: str, stats: Dict):
        """Csapat statisztikák mentése/frissítése"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO team_stats 
            (team_name, avg_corners, avg_yellow_cards, avg_shots_on_target, 
             avg_goals, matches_analyzed, confidence_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            team_name,
            stats.get('avg_corners', 0),
            stats.get('avg_yellow_cards', 0),
            stats.get('avg_shots_on_target', 0),
            stats.get('avg_goals', 0),
            stats.get('matches_analyzed', 0),
            stats.get('confidence_score', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def get_team_stats(self, team_name: str) -> Optional[Dict]:
        """Csapat statisztikák lekérése"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM team_stats WHERE team_name = ?
        """, (team_name,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'team_name': row['team_name'],
                'avg_corners': row['avg_corners'],
                'avg_yellow_cards': row['avg_yellow_cards'],
                'avg_shots_on_target': row['avg_shots_on_target'],
                'avg_goals': row['avg_goals'],
                'matches_analyzed': row['matches_analyzed'],
                'confidence_score': row['confidence_score'],
                'updated_at': row['updated_at']
            }
        return None
    
    # =========================================================================
    # Scraped meccsek
    # =========================================================================
    
    def save_scraped_match(self, match: Dict):
        """Scraped meccs mentése"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO scraped_matches 
            (home_team, away_team, match_date, match_time, league, match_url,
             corners_home, corners_away, yellow_cards_home, yellow_cards_away,
             shots_on_target_home, shots_on_target_away, goals_home, goals_away)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            match.get('home_team'),
            match.get('away_team'),
            match.get('match_date'),
            match.get('match_time'),
            match.get('league'),
            match.get('match_url'),
            match.get('corners_home', 0),
            match.get('corners_away', 0),
            match.get('yellow_cards_home', 0),
            match.get('yellow_cards_away', 0),
            match.get('shots_on_target_home', 0),
            match.get('shots_on_target_away', 0),
            match.get('goals_home', 0),
            match.get('goals_away', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def get_cache_stats(self) -> Dict:
        """Cache statisztikák lekérése"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Összes cache bejegyzés
        cursor.execute("SELECT COUNT(*) as total FROM cached_responses")
        total = cursor.fetchone()["total"]
        
        # Érvényes cache bejegyzések
        cursor.execute("""
            SELECT COUNT(*) as valid FROM cached_responses 
            WHERE expires_at > datetime('now')
        """)
        valid = cursor.fetchone()["valid"]
        
        # Team stats count
        cursor.execute("SELECT COUNT(*) as teams FROM team_stats")
        teams = cursor.fetchone()["teams"]
        
        conn.close()
        
        return {
            "total_cached": total,
            "valid_cached": valid,
            "expired_cached": total - valid,
            "teams_tracked": teams
        }
