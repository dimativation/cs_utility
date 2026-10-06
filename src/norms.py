"""Norms: what a lineup normally gives, and which lineup a throw was meant to be.

match_intent  - throw -> intended lineup by where the player stood and aimed (not where it landed),
                so missed throws are kept and marked hit=False.
lineup_norms  - per lineup: hit rate, technique, outcome (flash result / damage).
tier_norms    - per Premier rating tier: averages of the same metrics, to place a player on a scale.

Build once (after lineups.py):  python src/norms.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from metrics import load_throws

DATA = Path(__file__).resolve().parent.parent / "data"

THROW_R = 64  # stood within max(THROW_R, 2 * lineup throw spread) of the lineup throw spot
AIM_TOL = {"flash": 10.0}  # degrees, yaw and pitch; flashes are often thrown on the move
AIM_TOL_DEFAULT = 5.0
HIT_R = 100  # landed within max(HIT_R, 2 * lineup land spread) of the lineup landing spot
K = 8  # candidate lineups per throw


def _angle_diff(a, b):
    """Smallest difference between angles in degrees."""
    return np.abs((a - b + 180) % 360 - 180)


def _match_group(throws, lineups, aim_tol):
    """Match throws of one (type, side) to K nearest lineup throw-spots."""
    throw_r = np.maximum(THROW_R, 2 * lineups.throw_spread.to_numpy())
    hit_r = np.maximum(HIT_R, 2 * lineups.land_spread.to_numpy())

    dist, idx = cKDTree(lineups[["throw_cx", "throw_cy"]].to_numpy()).query(
        throws[["throw_x", "throw_y"]].to_numpy(),
        k=min(K, len(lineups)),
        distance_upper_bound=throw_r.max(),
    )
    dist, idx = np.atleast_2d(dist), np.atleast_2d(idx)
    found = idx < len(lineups)
    idx = np.where(found, idx, 0)  # dummy index so later slices are valid on misses

    yaw_dev = _angle_diff(throws.view_yaw.to_numpy()[:, None], lineups.view_yaw.to_numpy()[idx])
    pitch_dev = np.abs(throws.view_pitch.to_numpy()[:, None] - lineups.view_pitch.to_numpy()[idx])
    land_dev = np.hypot(
        throws.land_x.to_numpy()[:, None] - lineups.land_cx.to_numpy()[idx],
        throws.land_y.to_numpy()[:, None] - lineups.land_cy.to_numpy()[idx],
    )

    close = found & (dist <= throw_r[idx]) & (yaw_dev <= aim_tol) & (pitch_dev <= aim_tol)
    hit = close & (land_dev <= hit_r[idx])

    # a lineup it actually landed in wins; otherwise closest by position + aim
    score = dist / THROW_R + (yaw_dev + pitch_dev) / aim_tol - 100 * hit
    score = np.where(close, score, np.inf)
    best = score.argmin(axis=1)
    matched = np.flatnonzero(np.isfinite(score[np.arange(len(throws)), best]))
    pick = best[matched]

    return pd.DataFrame({
        "grenade_id": throws.grenade_id.to_numpy()[matched],
        "intent_id": lineups.index.to_numpy()[idx[matched, pick]],
        "hit": hit[matched, pick],
        "land_dev": land_dev[matched, pick],
        "throw_dist": dist[matched, pick],
        "aim_dev": np.maximum(yaw_dev[matched, pick], pitch_dev[matched, pick]),
    })


def match_intent(throws, catalog):
    """One row per matched throw: grenade_id, intent_id, hit, land_dev, throw_dist, aim_dev."""
    parts = []
    for (grenade_type, side), group in throws.groupby(["type", "side"], observed=True):
        lineups = catalog[(catalog.type == grenade_type) & (catalog.side == side)]
        if lineups.empty:
            continue
        parts.append(_match_group(group, lineups, AIM_TOL.get(grenade_type, AIM_TOL_DEFAULT)))
    return pd.concat(parts, ignore_index=True)


def lineup_norms(throws):
    """throws: throws (with outcomes) joined with intent. One row per lineup."""
    by_lineup = throws.groupby("intent_id")
    norms = pd.DataFrame({
        "attempts": by_lineup.size(),
        "hit_rate": by_lineup.hit.mean(),
        "land_dev_med": by_lineup.land_dev.median(),
        "airborne_hit": throws[throws.hit].groupby("intent_id").airborne.mean(),  # technique of hits
        "enemy_blind": by_lineup.enemy_blind.mean(),
        "own_blind": (throws.team_blind + throws.self_blind).groupby(throws.intent_id).mean(),
        "flash_kill_100": by_lineup.flash_kill.mean() * 100,
        "enemy_dmg": by_lineup.enemy_dmg.mean(),
        "no_dmg": (throws.enemy_dmg == 0).where(throws.enemy_dmg.notna()).groupby(throws.intent_id).mean().astype(float),
    })
    flash_share = throws.groupby("intent_id").flash_result.value_counts(normalize=True).unstack()
    return norms.join(flash_share.add_prefix("flash_")).rename_axis("lineup_id")


def tier_norms(throws):
    """Per rating tier averages. Unmatched throws have NaN intent/hit."""
    rated = throws[throws.rating_tier.notna()]
    by_tier = rated.groupby("rating_tier", observed=True)
    flashes = rated[rated.type == "flash"].groupby("rating_tier", observed=True)
    return pd.DataFrame({
        "throws": by_tier.size(),
        "hit_rate": by_tier.hit.mean(),
        "flash_useful": flashes.flash_result.apply(lambda s: (s == "полезная").mean()),
        "flash_harmful": flashes.flash_result.apply(lambda s: (s == "вредная").mean()),
        "enemy_blind": flashes.enemy_blind.mean(),
        "he_dmg": rated[rated.type == "he"].groupby("rating_tier", observed=True).enemy_dmg.mean(),
        "fire_dmg": rated[rated.type == "fire"].groupby("rating_tier", observed=True).enemy_dmg.mean(),
    })


def build(throws=None, catalog=None):
    throws = load_throws() if throws is None else throws
    catalog = pd.read_parquet(DATA / "lineups.parquet") if catalog is None else catalog
    intent = match_intent(throws, catalog)
    all_throws = throws.merge(intent, on="grenade_id", how="left")
    all_throws["hit"] = all_throws.hit.astype("boolean")
    matched = all_throws[all_throws.intent_id.notna()]
    return intent, lineup_norms(matched), tier_norms(all_throws)


if __name__ == "__main__":
    intent, lineup, tier = build()
    intent.to_parquet(DATA / "throw_intent.parquet", index=False)
    lineup.to_parquet(DATA / "lineup_norms.parquet")
    tier.to_parquet(DATA / "tier_norms.parquet")
    print(f"intent: {len(intent):,} throws matched, hit rate {intent.hit.mean():.0%}")
    print(tier.round(3).to_string())
