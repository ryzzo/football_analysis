import os
import sys
import time
import random
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

BASE = "https://api.football-data.org/v4"
OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def fetch_matches(session: requests.Session, token: str, comp: str, season: int) -> list[dict]:
    headers = {"X-Auth-Token": token}
    url = f"{BASE}/competitions/{comp}/matches"

    for attempt in range(1, 6):
        try:
            r = session.get(url, headers=headers, params={"season": season}, timeout=60)
            if r.status_code != 200:
                print("HTTP", r.status_code, "URL:", r.url)
                print("Body:", r.text[:300])
                r.raise_for_status()
            return r.json().get("matches", [])
        except requests.exceptions.SSLError as e:
            wait = (2 ** attempt) + random.random()
            print(f"⚠️ SSL error season={season} attempt={attempt}: {e}. retrying in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Failed after SSL retries for season {season}")


def flatten_match(m: dict, season: int) -> dict:
    score = (m.get("score") or {})
    ft = (score.get("fullTime") or {})
    return {
        "season_start_year": season,
        "match_id": m.get("id"),
        "utc_date": m.get("utcDate"),
        "status": m.get("status"),
        "matchday": m.get("matchday"),
        "stage": m.get("stage"),
        "home_team_id": (m.get("homeTeam") or {}).get("id"),
        "home_team": (m.get("homeTeam") or {}).get("name"),
        "away_team_id": (m.get("awayTeam") or {}).get("id"),
        "away_team": (m.get("awayTeam") or {}).get("name"),
        "home_goals": ft.get("home"),
        "away_goals": ft.get("away"),
        "winner": score.get("winner"),
    }


def main() -> int:
    token = os.getenv("FOOTBALL_DATA_TOKEN", "").strip()
    comp = os.getenv("COMPETITION_CODE", "PL").strip()
    start = int(os.getenv("SEASON_START", "2018"))
    end = int(os.getenv("SEASON_END", "2024"))

    if not token:
        print("❌ FOOTBALL_DATA_TOKEN missing in .env")
        return 1

    session = make_session()

    rows = []
    for season in range(start, end + 1):
        print(f"⬇️ Fetching {comp} season {season} ...")
        matches = fetch_matches(session, token, comp, season)
        print(f"  got {len(matches)} matches")
        for m in matches:
            rows.append(flatten_match(m, season))
        time.sleep(2)  # rate-limit friendly

    df = pd.DataFrame(rows)
    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True, errors="coerce")

    out_path = OUT_DIR / f"matches_{comp}_all_{start}_{end}.parquet"
    df.to_parquet(out_path, index=False)
    print("✅ Saved:", out_path, "| rows:", len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())