"""Test script for stealth web scraper"""
import asyncio
import sys
sys.path.insert(0, '.')

from app.config import get_config
from app.stealth_scraper import StealthScraper, PLAYWRIGHT_AVAILABLE

async def test_scraper():
    print(f"Playwright available: {PLAYWRIGHT_AVAILABLE}")
    
    if not PLAYWRIGHT_AVAILABLE:
        print("ERROR: Playwright not installed!")
        return
    
    config = get_config()
    scraper = StealthScraper(config)
    
    print("\n--- Starting stealth scrape test ---\n")
    result = await scraper.refresh_all_data()
    
    print(f"\nSuccess: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Fixtures found: {len(result.get('fixtures', []))}")
    
    # Show first 5 fixtures
    for f in result.get('fixtures', [])[:5]:
        print(f"  {f['league']}: {f['home_team']} vs {f['away_team']}")

if __name__ == "__main__":
    asyncio.run(test_scraper())
