"""Flatten the grenades pickle into parquet tables.

Usage: python src/prepare.py [input.list] [output_dir]

Tables:
  throws         one row per grenade (smoke / flash / he / fire)
  flash_victims  one row per flashed player
  damage         one row per damage tick from HE or fire
  players        one row per player in a match (with Premier rating if known)
"""
import gc
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

GRENADE_TYPES = {"Smokes": "smoke", "Flashbangs": "flash", "HEs": "he", "Fires": "fire"}
SIDES = {2: "T", 3: "CT"}


def load_matches(path):
    gc.disable()  # millions of small dicts: gc passes make loading several times slower
    try:
        with open(path, "rb") as f:
            return pickle.load(f)  # needs pymongo: full pickle stores bson Int64 / ObjectId
    finally:
        gc.enable()


def victim_kind(player_id, team, thrower_id, thrower_team):
    if player_id == thrower_id:
        return "self"
    return "team" if team == thrower_team else "enemy"


def add(table, **row):
    """Append one row to a column-oriented dict-of-lists."""
    for col, val in row.items():
        table[col].append(val)


def add_throw(throws, grenade_id, match_id, grenade_type, info, throw):
    pos, vel, land = throw["Position"], throw["Velocity"], info["PosTo"]
    add(
        throws,
        grenade_id=grenade_id,
        match_id=match_id,
        round=info["RoundIndexMatch"],
        type=grenade_type,
        thrower_id=info["ThrowerID"],
        side=SIDES[info["Team"]],
        start_tick=info["StartTick"],
        end_tick=info["EndTick"],
        throw_x=pos["X"], throw_y=pos["Y"], throw_z=pos["Z"],
        view_yaw=throw["ViewX"], view_pitch=throw["ViewY"],
        crouch=throw["IsCrouching"], airborne=throw["IsAirborne"],
        vel_x=vel["X"], vel_y=vel["Y"], vel_z=vel["Z"],
        land_x=land["X"], land_y=land["Y"], land_z=land["Z"],
    )


def add_flash_victims(victims, grenade, grenade_id, match_id, round_idx, thrower_id, thrower_team):
    for flashed in grenade.get("FlashedPlayers", []):
        kill = flashed.get("WasKilled") or {}  # empty if the player survived the flash
        pos, view = flashed["Position"], flashed["ViewAngle"]
        add(
            victims,
            grenade_id=grenade_id,
            match_id=match_id,
            round=round_idx,
            player_id=flashed["PlayerID"],
            side=SIDES[flashed["Team"]],
            victim=victim_kind(flashed["PlayerID"], flashed["Team"], thrower_id, thrower_team),
            duration=flashed["FlashDuration"],
            existing_duration=flashed["ExistingFlashDuration"],
            pos_x=pos["X"], pos_y=pos["Y"], pos_z=pos["Z"],
            view_yaw=view["X"], view_pitch=view["Y"],
            killed=bool(kill),
            killed_by=kill.get("By"),
            kill_tick=kill.get("TickNumber"),
            open_kill=kill.get("OpenKill", False),
            remaining_duration=kill.get("RemainingFlashDuration"),
        )


def add_damage(damage, grenade, grenade_id, match_id, round_idx, grenade_type, thrower_id, thrower_team):
    for damaged in grenade.get("DamagedPlayers", []):
        kind = victim_kind(damaged["PlayerID"], damaged["Team"], thrower_id, thrower_team)
        for hit in damaged["Damage"]:
            pos = hit["Position"]
            add(
                damage,
                grenade_id=grenade_id,
                match_id=match_id,
                round=round_idx,
                type=grenade_type,
                player_id=damaged["PlayerID"],
                side=SIDES[damaged["Team"]],
                victim=kind,
                tick=hit["TickNumber"],
                hp=hit["HealthDamage"],
                armor=hit["ArmorDamage"],
                pos_x=pos["X"], pos_y=pos["Y"], pos_z=pos["Z"],
                killed=hit.get("WasKilled", False),
            )


def flatten(matches):
    throws, victims, damage, players = (defaultdict(list) for _ in range(4))
    grenade_id = 0

    for i, match in enumerate(matches):
        match_id = int(match["MatchID"])

        for player in match["Players"]:
            add(
                players,
                match_id=match_id,
                player_id=player["StaticInfo"]["PlayerID"],
                rating=player.get("PremierRatingNew"),
            )

        for pickle_key, grenade_type in GRENADE_TYPES.items():
            for grenade in match[pickle_key]:
                info = grenade["GeneralInfo"]
                add_throw(throws, grenade_id, match_id, grenade_type, info, info["ThrowPoint"])
                add_flash_victims(
                    victims, grenade, grenade_id, match_id,
                    info["RoundIndexMatch"], info["ThrowerID"], info["Team"],
                )
                add_damage(
                    damage, grenade, grenade_id, match_id,
                    info["RoundIndexMatch"], grenade_type, info["ThrowerID"], info["Team"],
                )
                grenade_id += 1

        matches[i] = None  # free the match dict once its rows are copied out

    return {
        "throws": pd.DataFrame(throws),
        "flash_victims": pd.DataFrame(victims),
        "damage": pd.DataFrame(damage),
        "players": pd.DataFrame(players),
    }


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "grenades_by_match_0726.list")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "data")
    out.mkdir(exist_ok=True)

    matches = load_matches(src)
    print(f"loaded {len(matches)} matches")
    tables = flatten(matches)
    del matches
    gc.collect()

    for name, df in tables.items():
        for col in ("type", "side", "victim"):
            if col in df:
                df[col] = df[col].astype("category")
        for col in ("rating", "killed_by", "kill_tick"):
            if col in df:
                df[col] = df[col].astype("Int64")
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name}: {len(df):,} rows")


if __name__ == "__main__":
    main()
