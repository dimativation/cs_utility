"""Lineup catalog: repeated throws "from the same spot to the same spot".

For every (type, side) group:
  1. HDBSCAN on a random sample over (throw_x, throw_y, land_x, land_y);
  2. every throw of the group gets the label of its nearest clustered sample throw,
     if that throw is closer than ASSIGN_RADIUS; otherwise -1 (situational throw).
Jump / crouch / aim are not clustering features: they describe the technique of a lineup.

Build once:  python src/lineups.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import HDBSCAN

DATA = Path(__file__).resolve().parent.parent / "data"
FEATURES = ["throw_x", "throw_y", "land_x", "land_y"]
CENTER_COLS = {
    "throw_x": "throw_cx", "throw_y": "throw_cy",
    "land_x": "land_cx", "land_y": "land_cy",
}

SAMPLE = 60_000
MIN_CLUSTER_SIZE = 30  # in the sample; ~x5 in the full data
MIN_SAMPLES = 10
METHOD = "leaf"  # "eom" merges neighbouring lineups into huge clusters
ASSIGN_RADIUS = 50  # game units in 4D (throw + land)


def cluster_group(throws, sample=SAMPLE, min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
                  method=METHOD, radius=ASSIGN_RADIUS, seed=0):
    """Returns lineup label (-1 = no lineup) for every row of throws."""
    sampled = throws.sample(min(sample, len(throws)), random_state=seed)
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples,
        cluster_selection_method=method, copy=True,
    ).fit_predict(sampled[FEATURES].to_numpy())

    clustered = labels >= 0
    dist, idx = cKDTree(sampled.loc[clustered, FEATURES].to_numpy()).query(throws[FEATURES].to_numpy())
    return np.where(dist <= radius, labels[clustered][idx], -1)


def _circular_mean_deg(angles):
    """Mean of angles in degrees (yaw wraps at 360°)."""
    r = np.radians(angles)
    return np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))


def build_catalog(throws, labels):
    """throws: throws table; labels: lineup label per row. Returns (catalog, throw_lineup)."""
    clustered = throws.assign(label=labels)
    clustered = clustered[clustered.label >= 0]
    clustered["key"] = (
        clustered.type.astype(str) + "-" + clustered.side.astype(str) + "-" + clustered.label.astype(str)
    )

    by_key = clustered.groupby("key")
    catalog = by_key[FEATURES].median().rename(columns=CENTER_COLS)
    catalog["type"] = by_key.type.first().astype(str)
    catalog["side"] = by_key.side.first().astype(str)
    catalog["n"] = by_key.size()
    catalog["players"] = by_key.thrower_id.nunique()
    catalog["airborne"] = by_key.airborne.mean()
    catalog["crouch"] = by_key.crouch.mean()
    catalog["view_yaw"] = by_key.view_yaw.apply(_circular_mean_deg)
    catalog["view_pitch"] = by_key.view_pitch.median()

    clustered = clustered.join(catalog[list(CENTER_COLS.values())], on="key")
    clustered["throw_dev"] = np.hypot(clustered.throw_x - clustered.throw_cx, clustered.throw_y - clustered.throw_cy)
    clustered["land_dev"] = np.hypot(clustered.land_x - clustered.land_cx, clustered.land_y - clustered.land_cy)
    catalog["throw_spread"] = clustered.groupby("key").throw_dev.median()
    catalog["land_spread"] = clustered.groupby("key").land_dev.median()

    # readable id: type-side-rank by popularity, e.g. smoke-T-001
    catalog = catalog.sort_values(["type", "side", "n"], ascending=[True, True, False])
    rank = catalog.groupby(["type", "side"]).cumcount() + 1
    catalog["lineup_id"] = catalog.type + "-" + catalog.side + "-" + rank.map("{:03d}".format)
    clustered["lineup_id"] = clustered.key.map(catalog.lineup_id)
    catalog = catalog.set_index("lineup_id")

    group_size = throws.groupby(["type", "side"], observed=True).size()
    catalog["group_share"] = catalog.n / group_size.reindex(list(zip(catalog.type, catalog.side))).to_numpy()
    return catalog, clustered[["grenade_id", "lineup_id", "throw_dev", "land_dev"]]


def build_all(throws, **params):
    labels = pd.Series(-1, index=throws.index)
    for (grenade_type, side), group in throws.groupby(["type", "side"], observed=True):
        labels.loc[group.index] = cluster_group(group, **params)
        assigned = labels.loc[group.index]
        n_lineups = assigned.max() + 1
        print(f"{grenade_type:5s} {side:2s}: {n_lineups:4d} lineups, {(assigned >= 0).mean():.0%} of throws in lineups")
    return build_catalog(throws, labels.to_numpy())


if __name__ == "__main__":
    throws = pd.read_parquet(DATA / "throws.parquet")
    catalog, throw_lineup = build_all(throws)
    catalog.to_parquet(DATA / "lineups.parquet")
    throw_lineup.to_parquet(DATA / "throw_lineup.parquet", index=False)
    print(f"{len(catalog)} lineups, {len(throw_lineup):,} throws assigned ({len(throw_lineup) / len(throws):.0%})")
