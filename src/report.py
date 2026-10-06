"""Match report: for each of the 10 players, what went wrong with their grenades.

d = load_data()                      # once
summary, problems = match_report(match_id, d)

Norms of lineups are recomputed without the analysed match; the lineup catalog stays global
(one match is 0.01% of the data). No rating-level estimate here: ~10 throws per player is far too
few (hit-rate error +-13 pp vs a 9 pp spread across all rating tiers).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import load_tables, add_outcomes, add_context, BLIND_SEC
from norms import lineup_norms

DATA = Path(__file__).resolve().parent.parent / "data"

JUMP_LINEUP, NO_JUMP_LINEUP = 0.8, 0.2  # share of jump throws among hits
MIN_LINEUP_HIT = 0.6  # a miss is reported only for lineups that are normally hit
ALT_RADIUS = 150  # alternative lineup: thrown from within this distance of the player's spot
ALT_LAND_RADIUS = 400  # ... and landing in the same area as the player's lineup
ALT_MIN_ATTEMPTS = 200
ALT_MARGIN = {"flash": 0.15, "he": 5.0}  # better by +15 pp useful flashes / +5 damage
HARM_EXTRA_PER_OWN = 0.3  # extra weight per teammate/self blinded by a harmful flash
FULL_BUY_TEAM_NADES = 6  # team threw at least this many grenades in the round -> utility was available
SIDE_FLIP = {"T": "CT", "CT": "T"}

# validated against rating (fewer per grenade in higher tiers) -> "ошибки"; the rest -> "подсказки"
ERRORS = {"вредная флешка", "промах: техника", "промах раскидки"}
WEIGHTS = {"вредная флешка": 1.0, "смерть своего от флешки": 2.0, "промах: техника": 2.0,
           "промах раскидки": 1.0, "плохая раскидка": 1.5, "раунды без гранат": 0.5}  # last: per round
PROBLEM_COLS = ["player_id", "round", "kind", "text", "weight", "grenade_id", "lineup_id", "alt_id"]


def load_data():
    throws, flashes, damage, players = load_tables()
    throws = add_context(add_outcomes(throws, flashes, damage), players)
    intent = pd.read_parquet(DATA / "throw_intent.parquet")
    throws = throws.merge(intent, on="grenade_id", how="left")
    return {
        "th": throws,
        "fv": flashes,
        "dm": damage,
        "players": players,
        "cat": pd.read_parquet(DATA / "lineups.parquet"),
    }


def player_teams(match_id, data):
    """player_id -> team ('A' = started as T, 'B' = started as CT), inferred from any event with a side."""
    throws = data["th"]
    events = pd.concat([
        throws.loc[throws.match_id == match_id, ["thrower_id", "round", "side"]].set_axis(
            ["player_id", "round", "side"], axis=1),
        data["fv"].loc[data["fv"].match_id == match_id, ["player_id", "round", "side"]],
        data["dm"].loc[data["dm"].match_id == match_id, ["player_id", "round", "side"]],
    ])
    events = events[events["round"] < 24]  # regulation only: sides swap once, after round 11
    side = events.side.astype(str)
    started = np.where(events["round"] < 12, side, side.map(SIDE_FLIP))
    team = pd.Series(np.where(started == "T", "A", "B"), index=events.player_id)
    return team.groupby(level=0).agg(lambda s: s.mode().iloc[0])


def _problem(player_id, round_idx, kind, text, grenade_id=None, lineup_id=None, extra=0.0, alt_id=None):
    return {
        "player_id": player_id, "round": round_idx, "kind": kind, "text": text,
        "weight": WEIGHTS[kind] + extra, "grenade_id": grenade_id, "lineup_id": lineup_id, "alt_id": alt_id,
    }


def _alternative(row, norms, catalog):
    """Better lineup of the same type/side thrown from near the player's spot, or None."""
    grenade_type = row.type
    if grenade_type not in ALT_MARGIN:
        return None
    candidates = catalog[(catalog.type == grenade_type) & (catalog.side == row.side)].join(norms, how="inner")
    if row.intent_id not in norms.index:
        return None
    target = catalog.loc[row.intent_id]
    candidates = candidates[
        (candidates.attempts >= ALT_MIN_ATTEMPTS)
        & (np.hypot(candidates.throw_cx - row.throw_x, candidates.throw_cy - row.throw_y) <= ALT_RADIUS)
        & (np.hypot(candidates.land_cx - target.land_cx, candidates.land_cy - target.land_cy) <= ALT_LAND_RADIUS)
    ]
    metric = "flash_полезная" if grenade_type == "flash" else "enemy_dmg"
    if candidates.empty:
        return None
    own = norms.loc[row.intent_id, metric]
    best = candidates[metric].idxmax()
    other = candidates.loc[best, metric]
    return (best, own, other) if other >= own + ALT_MARGIN[grenade_type] else None


def _flash_problems(throws, flashes):
    problems = []
    for throw in throws[throws.type == "flash"].itertuples():
        victims = flashes[flashes.grenade_id == throw.grenade_id]
        own = victims[(victims.victim != "enemy") & (victims.duration > BLIND_SEC)]
        deaths = victims[(victims.victim != "enemy") & victims.killed]
        if throw.flash_result == "вредная":
            who = ", ".join(
                f"{'себя' if v.victim == 'self' else 'тиммейта'} {v.duration:.1f}с"
                for v in own.itertuples()
            )
            problems.append(_problem(
                throw.thrower_id, throw.round, "вредная флешка",
                f"флешка ослепила только своих: {who}; врагов дольше {BLIND_SEC:g} с — никого",
                throw.grenade_id, throw.intent_id, extra=HARM_EXTRA_PER_OWN * len(own),
            ))
        if len(deaths):
            problems.append(_problem(
                throw.thrower_id, throw.round, "смерть своего от флешки",
                f"{len(deaths)} свой(их) убит(ы), пока были ослеплены твоей флешкой",
                throw.grenade_id, throw.intent_id,
            ))
    return problems


def _miss_problems(throws, norms):
    """Lineup misses. Flashes skipped: geometric hit is a weak signal for them."""
    problems = []
    misses = throws[(throws.hit == False) & (throws.type != "flash")]  # noqa: E712 (nullable boolean)
    for throw in misses.itertuples():
        if throw.intent_id not in norms.index or norms.loc[throw.intent_id, "hit_rate"] < MIN_LINEUP_HIT:
            continue
        jump = norms.loc[throw.intent_id, "airborne_hit"]
        if jump > JUMP_LINEUP and not throw.airborne:
            kind, text = "промах: техника", (
                f"{throw.intent_id} не долетел: бросок без прыжка, а у попавших {jump:.0%} — в прыжке"
            )
        elif jump < NO_JUMP_LINEUP and throw.airborne:
            kind, text = "промах: техника", (
                f"{throw.intent_id} улетел мимо: бросок в прыжке, а раскидку кидают без прыжка"
            )
        else:
            kind, text = "промах раскидки", (
                f"{throw.intent_id} упал в {throw.land_dev:.0f} ед. от нормы "
                f"(попадают {norms.loc[throw.intent_id, 'hit_rate']:.0%})"
            )
        problems.append(_problem(
            throw.thrower_id, throw.round, kind, text, throw.grenade_id, throw.intent_id,
        ))
    return problems


def _alt_problems(throws, norms, catalog):
    """Lineups that are bad by themselves, with a better option from the same spot."""
    problems = []
    for throw in throws[throws.intent_id.notna() & throws.type.isin(list(ALT_MARGIN))].itertuples():
        alt = _alternative(throw, norms, catalog)
        if not alt:
            continue
        best, own, other = alt
        unit = "% полезных" if throw.type == "flash" else " урона"
        scale = 100 if throw.type == "flash" else 1
        problems.append(_problem(
            throw.thrower_id, throw.round, "плохая раскидка",
            f"{throw.intent_id} в норме даёт {own * scale:.0f}{unit}; "
            f"с того же места {best} — {other * scale:.0f}{unit}",
            throw.grenade_id, throw.intent_id, alt_id=best,
        ))
    return problems


def _idle_problems(throws, roster, teams):
    """Rounds without grenades while the team had utility."""
    problems = []
    with_team = throws.assign(team=throws.thrower_id.map(teams))
    team_round = with_team.groupby(["team", "round"]).size()
    per_player = with_team.groupby(["thrower_id", "round"]).size()
    full_buy = team_round[team_round >= FULL_BUY_TEAM_NADES]
    for player in roster.dropna(subset=["team"]).itertuples():
        team_full = [rnd for (team, rnd) in full_buy.index if team == player.team]
        idle = [rnd for rnd in team_full if per_player.get((player.player_id, rnd), 0) == 0]
        if not idle:
            continue
        problems.append(_problem(
            player.player_id, idle[0], "раунды без гранат",
            f"{len(idle)} из {len(team_full)} раундов, где команда кинула {FULL_BUY_TEAM_NADES}+ гранат, "
            f"ты не кинул ни одной (раунды {', '.join(str(r + 1) for r in idle)}; возможно, погиб раньше)",
            extra=WEIGHTS["раунды без гранат"] * (len(idle) - 1),
        ))
    return problems


def match_report(match_id, data):
    throws = data["th"][data["th"].match_id == match_id]
    norms = lineup_norms(data["th"][(data["th"].match_id != match_id) & data["th"].intent_id.notna()])
    teams = player_teams(match_id, data)
    roster = data["players"][data["players"].match_id == match_id].drop_duplicates("player_id")
    roster = roster.assign(team=roster.player_id.map(teams))

    problems = (
        _flash_problems(throws, data["fv"])
        + _miss_problems(throws, norms)
        + _alt_problems(throws, norms, data["cat"])
        + _idle_problems(throws, roster, teams)
    )
    problems = pd.DataFrame(problems, columns=PROBLEM_COLS)
    problems.insert(2, "category", np.where(problems.kind.isin(ERRORS), "ошибка", "подсказка"))
    return _summary(throws, roster, problems), problems.sort_values(["player_id", "round"])


def _summary(throws, roster, problems):
    by_player = throws.groupby("thrower_id")
    flashes = throws[throws.type == "flash"].groupby("thrower_id")
    n_rounds = throws["round"].nunique()
    summary = pd.DataFrame(index=roster.player_id)
    roster_i = roster.set_index("player_id")
    summary["team"] = roster_i.team
    summary["rating"] = roster_i.rating
    summary["гранат"] = by_player.size()
    for grenade_type in ["smoke", "flash", "he", "fire"]:
        summary[grenade_type] = throws[throws.type == grenade_type].groupby("thrower_id").size()
    summary = summary.fillna({c: 0 for c in ["гранат", "smoke", "flash", "he", "fire"]})
    summary["гранат/раунд"] = summary["гранат"] / n_rounds
    summary["флешки: полезных"] = (throws.flash_result == "полезная").groupby(throws.thrower_id).sum()
    summary["флешки: вредных"] = (throws.flash_result == "вредная").groupby(throws.thrower_id).sum()
    summary["врагов ослеплено"] = flashes.enemy_blind.sum()
    summary["своих ослеплено"] = flashes.team_blind.sum() + flashes.self_blind.sum()
    summary["урон HE+огонь"] = throws.groupby("thrower_id").enemy_dmg.sum()
    intended = throws[throws.intent_id.notna()]
    summary["раскидок"] = intended.groupby("thrower_id").size()
    summary["попаданий %"] = intended.groupby("thrower_id").hit.mean() * 100
    errors = problems[problems.category == "ошибка"]
    summary["ошибок"] = errors.groupby("player_id").size()
    summary["вес ошибок"] = errors.groupby("player_id").weight.sum()
    summary["подсказок"] = problems[problems.category == "подсказка"].groupby("player_id").size()
    summary = summary.fillna({c: 0 for c in [
        "флешки: полезных", "флешки: вредных", "врагов ослеплено", "своих ослеплено",
        "урон HE+огонь", "раскидок", "ошибок", "вес ошибок", "подсказок",
    ]})
    return summary.sort_values(["team", "вес ошибок"], ascending=[True, False])
