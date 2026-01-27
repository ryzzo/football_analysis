import sys
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/matches_PL_all_2023_2025.parquet")
OUT = Path("data/processed/team_events.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

def outcome_points(home_goals, away_goals):
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None, None
    if home_goals > away_goals:
        return 3, 0
    if home_goals < away_goals:
        return 0, 3
    return 1, 1

def main() -> int:
    if not RAW.exists():
        print("Missing:", RAW)
        return 1
    
    df = pd.read_parquet(RAW).copy()
    df = df.dropna(subset=["utc_date", "home_team_id", "away_team_id"])
    df = df.sort_values("utc_date")

    rows = []
    for _, m in df.iterrows():
        hp, ap = outcome_points(m["home_goals"], m["away_goals"])

        # Home event
        rows.append({
            "team_id": int(m["home_team_id"]),
            "event_timestamp": pd.to_datetime(m["utc_date"], utc=True),
            "points": hp if hp is not None else 0,
            "gf": 0 if pd.isna(m["home_goals"]) else int(m["home_goals"]),
            "ga": 0 if pd.isna(m["away_goals"]) else int(m["away_goals"]),
        })

        # Away event
        rows.append({
            "team_id": int(m["away_team_id"]),
            "event_timestamp": pd.to_datetime(m["utc_date"], utc=True),
            "points": ap if ap is not None else 0,
            "gf": 0 if pd.isna(m["away_goals"]) else int(m["away_goals"]),
            "ga": 0 if pd.isna(m["home_goals"]) else int(m["home_goals"]),
        })

    events = pd.DataFrame(rows).sort_values(["team_id", "event_timestamp"])

    # Rolling last 5 matches
    events["points_last_5"] = (
        events.groupby("team_id")["points"]
        .transform(lambda s: s.shift(1).rolling(5).sum())
        .fillna(0.0)
    )
    events["gf_last_5"] = (
        events.groupby("team_id")["gf"]
        .transform(lambda s: s.shift(1).rolling(5).sum())
        .fillna(0.0)
    )
    events["ga_last_5"] = (
        events.groupby("team_id")["ga"]
        .transform(lambda s: s.shift(1).rolling(5).sum())
        .fillna(0.0)
    )

    events.to_parquet(OUT, index=False)
    print("Saved:", OUT, "| rows:", len(events))
    return 0

if __name__ == "__main__":
    sys.exit(main())