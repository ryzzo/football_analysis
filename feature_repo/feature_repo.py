from datetime import timedelta
import pandas as pd

from feast import Entity, FeatureView, Field
from feast.types import Int64, Float64
from feast.infra.offline_stores.file_source import FileSource

MATCHES_PATH = "/app/data/processed/team_events.parquet"

matches_source = FileSource(
    path=MATCHES_PATH,
    event_timestamp_column="event_timestamp",
)

team = Entity(
    name="team_id",
    join_keys=["team_id"],
    description="Football team id"
)

team_form = FeatureView(
    name="team_form",
    entities=[team],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="points_last_5", dtype=Float64),
        Field(name="gf_last_5", dtype=Float64),
        Field(name="ga_last_5", dtype=Float64),
    ],
    source=matches_source,
)