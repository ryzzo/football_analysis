import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.football-data.org/v4"

def main() -> int:
    token = os.getenv("FOOTBALL_DATA_TOKEN", "").strip()
    comp = os.getenv("COMPETITION_CODE", "PL").strip()

    if not token:
        print("FOOTBALL_DATA_TOKEN")
        return 1
    
    r = requests.get(
        f"{BASE}/competitions/{comp}",
        headers={"X-Auth-Token": token},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()

    seasons = data.get("seasons", [])
    # seasons contain startDate/endDate - take the start year
    years = []
    for s in seasons:
        start = (s.get("startDate") or "")
        if len(start) >= 4 and start[:4].isdigit():
            years.append(int(start[:4]))

    years = sorted(set(years))
    print("Available start years:", years)
    if years:
        print("Min:", min(years), "Max:", max(years))
    return 0

if __name__ == "__main__":
    sys.exit(main())