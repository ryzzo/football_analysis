import sys
from pathlib import Path
import pandas as pd

from feast import FeatureStore

RAW_MATCHES = Path("data/raw/matches_PL_all_2023_2025.parquet")
OUT = Path("data/processed/match_training.parquet")

FEATURES = [
    "team_form:points_last_5",
    "team_form:gf_last_5",
    "team_form:ga_last_5",
]

def outcome_label(home_goals, away_goals):
    if pd.isna(home_goals) or pd.isna(away_goals):
        return None
    if home_goals > away_goals:
        return 0 # HOME_WIN
    if home_goals < away_goals:
        return 2 # AWAY_WIN
    return 1     # DRAW

def main() -> int:
    if not RAW_MATCHES.exists():
        print("Missing:", RAW_MATCHES)
        return 1
    
    df = pd.read_parquet(RAW_MATCHES).copy()
    df["event_timestamp"] = pd.to_datetime(df["utc_date"], utc=True, errors="coerce")

    # Use only finished matches for training
    df = df[df["status"] == "FINISHED"].copy()

    df["label"] = df.apply(lambda r: outcome_label(r["home_goals"], r["away_goals"]), axis=1)
    df = df.dropna(subset=["event_timestamp", "home_team_id", "away_team_id", "label"])

    # Build the entity dataframe for home and away separately
    home_entities = pd.DataFrame({
        "team_id": df["home_team_id"].astype(int),
        "event_timestamp": df["event_timestamp"],
        "match_id": df["match_id"].astype(int),
    })
    away_entities = pd.DataFrame({
        "team_id": df["away_team_id"].astype(int),
        "event_timestamp": df["event_timestamp"],
        "match_id": df["match_id"].astype(int),
    })

    store = FeatureStore(repo_path="feature_repo")

    home_feats = store.get_historical_features(
        entity_df = home_entities,
        features = FEATURES
    ).to_df()
    away_feats = store.get_historical_features(
        entity_df = away_entities,
        features = FEATURES
    ).to_df()

    # Rename feature columns to distinguish home vs away
    home_feats = home_feats.rename(columns={
        "points_last_5": "home_points_last_5",
        "gf_last_5": "home_gf_last_5",
        "ga_last_5": "home_ga_last_5",
    })
    away_feats = away_feats.rename(columns={
        "points_last_5": "away_points_last_5",
        "gf_last_5": "away_gf_last_5",
        "ga_last_5": "away_ga_last_5",
    })

    # Join features back to matches by match_id + timestamp
    base = df[[
        "match_id", "season_start_year", "event_timestamp",
        "home_team_id", "away_team_id",
        "home_team", "away_team",
        "home_goals", "away_goals",
        "label"
    ]].copy()

    base["match_id"] = base["match_id"].astype(int)

    # Join on match_id and event_timestamp
    out = base.merge(
        home_feats.drop(columns=["team_id"]),
        on=["match_id", "event_timestamp"],
        how="left",
    ).merge(
        away_feats.drop(columns=["team_id"]),
        on=["match_id", "event_timestamp"],
        how="left"
    )

    # Fill missing features
    for c in ["home_points_last_5", "home_gf_last_5", "home_ga_last_5", "away_points_last_5", "away_gf_last_5", "away_ga_last_5"]:
        out[c] = out[c].fillna(0.0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    print("Saved:", OUT, "| rows:", len(out))
    print("Label distribution:", out["label"].value_counts().to_dict())
    return 0

if __name__ == "__main__":
    sys.exit(main())