"""Test scraper API connection"""
import aiohttp
import asyncio

async def test_api():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.sofascore.com/'
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        url = 'https://www.sofascore.com/api/v1/sport/football/scheduled-events/2026-01-28'
        print(f"Fetching: {url}")
        
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            print(f"Status: {resp.status}")
            
            if resp.status == 200:
                data = await resp.json()
                events = data.get('events', [])
                print(f"Events count: {len(events)}")
                
                if events:
                    first = events[0]
                    print(f"First match: {first.get('homeTeam', {}).get('name')} vs {first.get('awayTeam', {}).get('name')}")
            else:
                text = await resp.text()
                print(f"Error: {text[:500]}")

if __name__ == "__main__":
    asyncio.run(test_api())
