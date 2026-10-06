"""Per-grenade outcome and context features.

load_throws() returns the throws table enriched with:
  outcome  - what the grenade did (blind, damage, kills); NaN where not applicable
  context  - pistol round, time since first grenade of the round, team grenades nearby,
             thrower's Premier rating and rating tier
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

TICKRATE = 64
BLIND_SEC = 1.5  # flash longer than this actually takes the player out of the fight
BURST_SEC = 5  # grenades of one side within +-5s are treated as one coordinated push
PISTOL_ROUNDS = (0, 12)  # MR12
GROUP_OFFSET = 10**8  # bigger than any start_tick, so nearby-windows never cross groups

# flash result by who was blinded longer than BLIND_SEC
FLASH_RESULTS = ["полезная", "размен", "вредная", "пустая"]  # enemies only / enemies+own / own only / nobody
FLASH_COLS = ["enemy_blind", "team_blind", "self_blind", "enemy_blind_sec", "flash_kill", "team_flash_death"]
DMG_COLS = ["enemy_dmg", "enemy_kill", "enemies_hit"]

# Premier rating colour tiers
RATING_BINS = [0, 5000, 10000, 15000, 20000, 25000, np.inf]
RATING_LABELS = ["<5k", "5-10k", "10-15k", "15-20k", "20-25k", "25k+"]


def flash_outcomes(flashes):
    """One row per flash grenade_id."""
    long = flashes.duration > BLIND_SEC
    f = flashes.assign(
        enemy_blind=long & (flashes.victim == "enemy"),
        team_blind=long & (flashes.victim == "team"),
        self_blind=long & (flashes.victim == "self"),
        enemy_blind_sec=flashes.duration.where(flashes.victim == "enemy", 0.0),
        flash_kill=flashes.killed & (flashes.victim == "enemy"),
        team_flash_death=flashes.killed & (flashes.victim != "enemy"),
    )
    return f.groupby("grenade_id")[FLASH_COLS].sum()


def damage_outcomes(damage):
    """One row per HE / fire grenade_id. The dataset only records damage to enemies."""
    g = damage.groupby("grenade_id")
    out = pd.concat([g.hp.sum(), g.killed.sum(), g.player_id.nunique()], axis=1)
    out.columns = DMG_COLS
    return out


def add_outcomes(throws, flashes, damage):
    out = throws.join(flash_outcomes(flashes), on="grenade_id").join(damage_outcomes(damage), on="grenade_id")
    is_flash = out.type == "flash"
    is_dmg = out.type.isin(["he", "fire"])
    # right type with no victims -> 0; other types -> NaN
    out[FLASH_COLS] = out[FLASH_COLS].fillna(0).where(is_flash)
    out[DMG_COLS] = out[DMG_COLS].fillna(0).where(is_dmg)

    enemy = out.enemy_blind > 0
    own = (out.team_blind + out.self_blind) > 0
    # полезную / размен / вредную — by who was blinded; пустую if nobody
    result = np.select([enemy & ~own, enemy & own, ~enemy & own], FLASH_RESULTS[:3], FLASH_RESULTS[3])
    out["flash_result"] = pd.Categorical(np.where(is_flash, result, None), categories=FLASH_RESULTS)
    return out


def team_nades_nearby(throws):
    """Other grenades of the same match/round/side within ±BURST_SEC.

    All groups share one sorted array; GROUP_OFFSET keeps windows from crossing borders.
    """
    grp = throws.groupby(["match_id", "round", "side"], observed=True).ngroup().to_numpy(np.int64)
    key = grp * GROUP_OFFSET + throws.start_tick.to_numpy(np.int64)
    s = np.sort(key)
    w = BURST_SEC * TICKRATE
    return np.searchsorted(s, key + w, "right") - np.searchsorted(s, key - w, "left") - 1


def add_context(throws, players):
    throws = throws.copy()
    throws["pistol"] = throws["round"].isin(PISTOL_ROUNDS)
    first = throws.groupby(["match_id", "round"]).start_tick.transform("min")
    throws["sec_from_first"] = (throws.start_tick - first) / TICKRATE
    throws["team_nades_nearby"] = team_nades_nearby(throws)

    rated = players.dropna(subset=["rating"]).drop_duplicates(["match_id", "player_id"])
    throws = throws.merge(
        rated.rename(columns={"player_id": "thrower_id"}),
        on=["match_id", "thrower_id"],
        how="left",
    )
    throws["rating_tier"] = pd.cut(
        throws.rating.astype(float), RATING_BINS, labels=RATING_LABELS, right=False,
    )
    return throws


def load_tables():
    names = ["throws", "flash_victims", "damage", "players"]
    return [pd.read_parquet(DATA / f"{n}.parquet") for n in names]


def load_throws():
    throws, flashes, damage, players = load_tables()
    return add_context(add_outcomes(throws, flashes, damage), players)
