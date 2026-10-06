"""Grenades on Cache: interactive dashboard.

Run:  streamlit run dashboard/app.py
All heavy computation is precomputed in data/ by the src/ scripts; here we only filter and draw.
"""
import os

# PyArrow's default pool (mimalloc) segfaults when Streamlit reruns the script
# on its background thread and boolean-indexes Arrow string columns.
os.environ["ARROW_DEFAULT_MEMORY_POOL"] = "system"

import pyarrow as pa

pa.set_memory_pool(pa.system_memory_pool())

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib.path import Path as Polygon

from mapviz import add_arrow, add_density, map_figure
from metrics import FLASH_RESULTS, RATING_LABELS
from report import load_data, match_report

st.set_page_config(page_title="Гранаты на Cache", layout="wide")

TYPES = {"smoke": "Смок", "flash": "Флешка", "he": "HE", "fire": "Молотов"}
PALETTE = px.colors.qualitative.Light24
NO_RATING = "без рейтинга"
ZOOM_PAD_LINEUP = 500
ZOOM_PAD_PROBLEM = 700


@st.cache_resource(show_spinner="Загружаю данные…")
def data():
    d = load_data()
    d["norms"] = pd.read_parquet(ROOT / "data" / "lineup_norms.parquet")
    return d


@st.cache_data(show_spinner="Разбираю матч…")
def report(match_id):
    return match_report(match_id, data())


def pct(x):
    return f"{x:.0%}" if pd.notna(x) else "—"


def selected_row(event):
    rows = event.selection.rows if event is not None else []
    return rows[0] if rows else None


def catalog():
    d = data()
    return d["cat"].join(d["norms"])


def zoom(fig, x0, y0, x1, y1, pad):
    fig.update_xaxes(range=[min(x0, x1) - pad, max(x0, x1) + pad])
    fig.update_yaxes(range=[min(y0, y1) - pad, max(y0, y1) + pad])


# ---------------------------------------------------------------- shared pieces

def lineup_table(ids):
    cat, norms = data()["cat"], data()["norms"]
    rows = cat.loc[ids, ["type", "side", "n", "players", "airborne", "land_spread"]].join(norms)
    return pd.DataFrame({
        "раскидка": rows.index, "бросков": rows.n, "игроков": rows.players, "попыток": rows.attempts,
        "попаданий": rows.hit_rate, "в прыжке": rows.airborne, "разброс": rows.land_spread.round(0),
        "полезных": rows.get("flash_полезная"), "вредных": rows.get("flash_вредная"), "пустых": rows.get("flash_пустая"),
        "урон": rows.enemy_dmg.round(1),
    }).reset_index(drop=True)


LINEUP_COLS = {
    "попаданий": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
    "в прыжке": st.column_config.NumberColumn(format="percent"),
    "полезных": st.column_config.NumberColumn(format="percent"),
    "вредных": st.column_config.NumberColumn(format="percent"),
    "пустых": st.column_config.NumberColumn(format="percent"),
}


def lineups_map(ids, height=650, title=""):
    """Arrows throw -> land for lineups; markers at landing carry the lineup id for click selection."""
    cat, norms = data()["cat"], data()["norms"]
    fig = map_figure(height)
    rows = cat.loc[ids].join(norms)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(rows))]
    for color, (_, row) in zip(colors, rows.iterrows()):
        add_arrow(fig, row.throw_cx, row.throw_cy, row.land_cx, row.land_cy, color)
    hover = [
        f"<b>{lid}</b><br>бросков {row.n:,}<br>попаданий {pct(row.hit_rate)}<br>в прыжке {pct(row.airborne)}"
        for lid, row in rows.iterrows()
    ]
    fig.add_trace(go.Scatter(
        x=rows.land_cx, y=rows.land_cy, mode="markers+text",
        text=[lid.split("-")[-1] for lid in ids],
        textposition="top center", textfont=dict(color="white", size=10),
        marker=dict(size=10, color=colors, line=dict(color="white", width=1)),
        customdata=list(ids), hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(title=title)
    return fig


def lineup_card(lineup_id):
    cat, norms, throws = data()["cat"], data()["norms"], data()["th"]
    row, n = cat.loc[lineup_id], norms.loc[lineup_id]
    st.subheader(f"Раскидка {lineup_id}")
    cols = st.columns(6)
    cols[0].metric("Бросков (по месту падения)", f"{row.n:,}")
    cols[1].metric("Попыток (по намерению)", f"{int(n.attempts):,}")
    cols[2].metric("Попаданий", pct(n.hit_rate))
    cols[3].metric("В прыжке у попавших", pct(n.airborne_hit))
    cols[4].metric("Разброс падения", f"{row.land_spread:.0f} ед.")
    if row.type == "flash":
        cols[5].metric("Полезных / вредных", f"{pct(n['flash_полезная'])} / {pct(n['flash_вредная'])}")
    elif row.type in ("he", "fire"):
        cols[5].metric("Средний урон", f"{n.enemy_dmg:.1f}")

    attempts = throws[throws.intent_id == lineup_id]
    attempts = attempts.sample(min(1500, len(attempts)), random_state=0)
    fig = map_figure(600)
    for hit, color, name in [(True, "lime", "попадание"), (False, "red", "промах")]:
        pts = attempts[attempts.hit == hit]
        fig.add_trace(go.Scatter(
            x=pts.throw_x, y=pts.throw_y, mode="markers",
            marker=dict(size=3, color=color, opacity=.4),
            name=f"{name}: откуда", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=pts.land_x, y=pts.land_y, mode="markers",
            marker=dict(size=5, color=color, symbol="x"),
            name=f"{name}: куда", hoverinfo="skip",
        ))
    add_arrow(fig, row.throw_cx, row.throw_cy, row.land_cx, row.land_cy, "white", 3)
    zoom(fig, row.throw_cx, row.throw_cy, row.land_cx, row.land_cy, ZOOM_PAD_LINEUP)
    fig.update_layout(
        showlegend=True, legend=dict(bgcolor="rgba(0,0,0,.5)", font=dict(color="white")),
        title=f"Попытки {lineup_id}: зелёные — попадания, красные — промахи",
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------- page: map

def page_map():
    throws = data()["th"]
    st.title("Карта: куда и откуда кидают")
    filters = st.columns(5)
    side = filters[0].radio("Сторона", ["T", "CT"])
    grenade_type = filters[1].selectbox("Граната", list(TYPES), format_func=TYPES.get)
    rating = filters[2].selectbox("Рейтинг", ["все"] + RATING_LABELS + [NO_RATING])
    pistol = filters[3].radio("Раунды", ["все", "пистолетные", "остальные"])
    where = filters[4].radio("Точки", ["куда упала", "откуда кинули"])

    tier = throws.rating_tier.astype(object).fillna(NO_RATING)
    mask = (throws.side == side) & (throws.type == grenade_type)
    if rating != "все":
        mask &= tier == rating
    if pistol != "все":
        mask &= throws.pistol == (pistol == "пистолетные")
    filtered = throws[mask]
    x_col, y_col = ("land_x", "land_y") if where == "куда упала" else ("throw_x", "throw_y")

    kpi(filtered, grenade_type)
    left, right = st.columns([3, 2])
    with left:
        fig = map_figure(680)
        add_density(fig, filtered[x_col], filtered[y_col])
        sample = filtered.sample(min(6000, len(filtered)), random_state=0)
        # almost invisible points: Plotly needs them so box/lasso selection works
        fig.add_trace(go.Scatter(
            x=sample[x_col], y=sample[y_col], mode="markers",
            marker=dict(size=3, color="white", opacity=0.05), hoverinfo="skip",
        ))
        fig.update_layout(
            dragmode="select",
            title="Выделите область рамкой (или лассо в панели графика) → справа статистика по ней",
        )
        event = st.plotly_chart(
            fig, width="stretch", on_select="rerun", selection_mode=("box", "lasso"), key="map_sel",
        )
    with right:
        region = selected_region(event, filtered[x_col].to_numpy(), filtered[y_col].to_numpy())
        if region is None:
            st.info("Выделите область на карте, чтобы увидеть, что туда кидают и с каким результатом.")
            return
        zone = filtered[region]
        st.subheader(f"Выделено бросков: {len(zone):,}")
        kpi(zone, grenade_type, compact=True)
        top = zone.intent_id.value_counts().head(10)
        if len(top):
            st.markdown("**Самые частые раскидки в зоне**")
            st.dataframe(lineup_table(top.index), hide_index=True, column_config=LINEUP_COLS, width="stretch")


def selected_region(event, x, y):
    sel = event.selection if event is not None else None
    if not sel:
        return None
    if sel.get("box"):
        box = sel["box"][0]
        return (x >= min(box["x"])) & (x <= max(box["x"])) & (y >= min(box["y"])) & (y <= max(box["y"]))
    if sel.get("lasso"):
        poly = sel["lasso"][0]
        return Polygon(np.column_stack([poly["x"], poly["y"]])).contains_points(np.column_stack([x, y]))
    return None


def kpi(throws, grenade_type, compact=False):
    hit = pct(throws.hit.mean()) if throws.hit.notna().any() else "—"
    items = [("Бросков", f"{len(throws):,}"), ("Попаданий в раскидку", hit)]
    if grenade_type == "flash":
        result = throws.flash_result.value_counts(normalize=True)
        items += [
            ("Полезные", pct(result.get("полезная"))),
            ("Вредные", pct(result.get("вредная"))),
            ("Пустые", pct(result.get("пустая"))),
        ]
    elif grenade_type in ("he", "fire"):
        label = "Урон HE" if grenade_type == "he" else "Урон молотова"
        items = [("Бросков", f"{len(throws):,}"), (label, f"{throws.enemy_dmg.mean():.1f}" if len(throws) else "—")]
    if compact:
        items = items[1:]
    for col, (label, value) in zip(st.columns(len(items)), items):
        col.metric(label, value)


# ---------------------------------------------------------------- page: lineups

SORTS = {
    "популярность": ("n", False),
    "% попаданий": ("hit_rate", False),
    "% полезных флешек": ("flash_полезная", False),
    "% вредных флешек": ("flash_вредная", False),
    "средний урон": ("enemy_dmg", False),
    "разброс падения": ("land_spread", True),
}


def page_lineups():
    cat = catalog()
    st.title("Каталог раскидок")
    filters = st.columns(5)
    grenade_type = filters[0].selectbox("Граната", list(TYPES), format_func=TYPES.get)
    side = filters[1].radio("Сторона", ["T", "CT"], horizontal=True)
    sort = filters[2].selectbox("Сортировка", [s for s in SORTS if grenade_type == "flash" or "флеш" not in s])
    min_attempts = filters[3].slider("Мин. попыток", 0, 2000, 200, 50)
    top_n = filters[4].slider("Сколько показать", 5, 40, 15)

    filtered = cat[(cat.type == grenade_type) & (cat.side == side) & (cat.attempts >= min_attempts)]
    col, ascending = SORTS[sort]
    ids = filtered.sort_values(col, ascending=ascending).head(top_n).index
    st.caption(
        f"Раскидок {TYPES[grenade_type].lower()} {side} с ≥{min_attempts} попытками: {len(filtered)}. "
        f"Показаны топ-{len(ids)} по «{sort}»."
    )

    left, right = st.columns([3, 2])
    with left:
        map_event = st.plotly_chart(
            lineups_map(ids, title="Клик по точке падения — карточка раскидки"),
            width="stretch", on_select="rerun", selection_mode="points",
            key=f"lm_{grenade_type}_{side}",
        )
    with right:
        table = lineup_table(ids)
        table_event = st.dataframe(
            table, hide_index=True, column_config=LINEUP_COLS, width="stretch",
            on_select="rerun", selection_mode="single-row", key=f"lt_{grenade_type}_{side}",
        )
    lineup_id = picked_lineup(map_event, table_event, table, ids)
    if lineup_id:
        lineup_card(lineup_id)


def picked_lineup(map_event, table_event, table, ids):
    points = map_event.selection.points if map_event is not None else []
    if points and points[0].get("customdata"):
        return points[0]["customdata"]
    row = selected_row(table_event)
    if row is not None:
        return table.loc[row, "раскидка"]
    return ids[0] if len(ids) else None


# ---------------------------------------------------------------- page: flashes

def page_flashes():
    cat = catalog()
    st.title("Флешки: какие работают, а какие слепят своих")
    filters = st.columns(3)
    side = filters[0].radio("Сторона", ["T", "CT"], horizontal=True)
    min_attempts = filters[1].slider("Мин. попыток", 50, 2000, 300, 50)
    top_n = filters[2].slider("Сколько показать", 5, 20, 8)
    flashes = cat[(cat.type == "flash") & (cat.side == side) & (cat.attempts >= min_attempts)]
    best = flashes.sort_values("flash_полезная", ascending=False).head(top_n).index
    worst = flashes.sort_values("flash_вредная", ascending=False).head(top_n).index

    left, right = st.columns(2)
    with left:
        st.plotly_chart(lineups_map(best, 520, "Лучшие: чаще всего слепят только врагов"), width="stretch")
        st.dataframe(lineup_table(best), hide_index=True, column_config=LINEUP_COLS, width="stretch")
    with right:
        st.plotly_chart(lineups_map(worst, 520, "Худшие: чаще всего слепят только своих"), width="stretch")
        st.dataframe(lineup_table(worst), hide_index=True, column_config=LINEUP_COLS, width="stretch")

    st.subheader("Распределение по всем флешкам-раскидкам")
    fig = px.scatter(
        flashes.reset_index(), x="flash_полезная", y="flash_вредная", size="attempts", hover_name="lineup_id",
        labels={"flash_полезная": "доля полезных", "flash_вредная": "доля вредных"}, height=420,
    )
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------- page: match

SUMMARY_COLS = [
    "team", "rating", "гранат", "smoke", "flash", "he", "fire",
    "флешки: полезных", "флешки: вредных", "врагов ослеплено", "своих ослеплено",
    "урон HE+огонь", "попаданий %", "ошибок", "вес ошибок", "подсказок",
]


def page_match():
    throws = data()["th"]
    st.title("Разбор матча")
    ids = throws.match_id.drop_duplicates().sort_values().tolist()
    if "match" not in st.session_state:
        st.session_state.match = ids[0]
    cols = st.columns([3, 1])
    if cols[1].button("🎲 Случайный матч", width="stretch"):
        st.session_state.match = int(np.random.choice(ids))
    match_id = cols[0].selectbox("Матч", ids, index=ids.index(st.session_state.match))
    st.session_state.match = match_id

    summary, problems = report(match_id)
    match_throws = throws[throws.match_id == match_id]
    st.caption(
        f"Раундов: {match_throws['round'].nunique()} · гранат: {len(match_throws)} · "
        f"игроков с рейтингом: {summary.rating.notna().sum()} / {len(summary)}"
    )
    st.markdown("**Игроки** (команда A начала за T). Выберите строку, чтобы открыть разбор игрока.")
    table = summary[SUMMARY_COLS].reset_index().rename(
        columns={"player_id": "игрок", "team": "команда", "rating": "рейтинг"},
    )
    event = st.dataframe(
        table.round(1), hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        key=f"players_{match_id}",
        column_config={
            "вес ошибок": st.column_config.ProgressColumn(
                min_value=0, max_value=float(summary["вес ошибок"].max() or 1), format="%.1f",
            ),
        },
    )
    row = selected_row(event)
    player_id = table.loc[row, "игрок"] if row is not None else summary["вес ошибок"].idxmax()
    player_view(match_id, player_id, summary, problems)


def player_view(match_id, player_id, summary, problems):
    st.header(f"Игрок {player_id} · команда {summary.loc[player_id, 'team']}")
    mine = problems[problems.player_id == player_id].sort_values(["category", "weight"], ascending=[True, False])
    left, right = st.columns([2, 3])
    with left:
        if mine.empty:
            st.success("Проблем не найдено.")
            selected = None
        else:
            st.markdown("**Ошибки** — подтверждены: у сильных игроков их меньше. **Подсказки** — что можно попробовать.")
            view = mine.assign(раунд=mine["round"] + 1)[["раунд", "category", "kind", "text"]].rename(
                columns={"category": "тип", "kind": "проблема", "text": "что произошло"},
            )
            event = st.dataframe(
                view, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                key=f"prob_{match_id}_{player_id}", height=min(600, 40 + 35 * len(view)),
            )
            row = selected_row(event)
            selected = mine.iloc[row] if row is not None else mine.iloc[0]
    with right:
        st.plotly_chart(problem_map(match_id, player_id, selected), width="stretch")
    rounds_table(match_id, summary)


def problem_map(match_id, player_id, selected):
    throws, flashes, cat = data()["th"], data()["fv"], data()["cat"]
    fig = map_figure(620)
    own = throws[(throws.match_id == match_id) & (throws.thrower_id == player_id)]
    for throw in own.itertuples():
        add_arrow(fig, throw.throw_x, throw.throw_y, throw.land_x, throw.land_y, "rgba(200,200,200,.35)", 1)
    title = "Все броски игрока за матч"
    if selected is not None and pd.notna(selected.grenade_id):
        grenade = throws[throws.grenade_id == selected.grenade_id].iloc[0]
        add_arrow(fig, grenade.throw_x, grenade.throw_y, grenade.land_x, grenade.land_y, "red", 3)
        if selected.kind in ("вредная флешка", "смерть своего от флешки"):
            add_flash_victims(fig, flashes, selected.grenade_id)
        if pd.notna(selected.lineup_id):
            lineup = cat.loc[selected.lineup_id]
            color = "lime" if selected.kind.startswith("промах") else "orange"
            add_arrow(fig, lineup.throw_cx, lineup.throw_cy, lineup.land_cx, lineup.land_cy, color, 3)
        if pd.notna(selected.alt_id):
            alt = cat.loc[selected.alt_id]
            add_arrow(fig, alt.throw_cx, alt.throw_cy, alt.land_cx, alt.land_cy, "deepskyblue", 3)
        zoom(fig, grenade.throw_x, grenade.throw_y, grenade.land_x, grenade.land_y, ZOOM_PAD_PROBLEM)
        legends = {
            "вредная флешка": "красная — бросок; точки — ослеплённые (красные — свои, зелёные — враги)",
            "смерть своего от флешки": "красная — бросок; † — убит, пока был ослеплён",
            "плохая раскидка": "красная — бросок, оранжевая — раскидка игрока, голубая — альтернатива",
        }
        legend = legends.get(selected.kind, "красная — бросок, зелёная — норма раскидки")
        title = f"Раунд {selected['round'] + 1} · {selected.kind}<br><sup>{legend}</sup>"
    fig.update_layout(title=title)
    return fig


def add_flash_victims(fig, flashes, grenade_id):
    victims = flashes[flashes.grenade_id == grenade_id]
    color = np.where(victims.victim == "enemy", "lime", "red")
    labels = [
        f"{kind} {duration:.1f}с" + (" †" if dead else "")
        for kind, duration, dead in zip(victims.victim, victims.duration, victims.killed)
    ]
    fig.add_trace(go.Scatter(
        x=victims.pos_x, y=victims.pos_y, mode="markers+text",
        marker=dict(size=12, color=color, line=dict(color="black", width=1)),
        text=labels, textposition="middle right", textfont=dict(color="white"), hoverinfo="skip",
    ))


def rounds_table(match_id, summary):
    throws = data()["th"]
    match_throws = throws[throws.match_id == match_id].assign(team=lambda x: x.thrower_id.map(summary.team))
    table = match_throws.pivot_table(
        index=["team", "thrower_id"], columns="round", values="grenade_id", aggfunc="size", fill_value=0,
    )
    table = table.reindex(pd.MultiIndex.from_frame(summary.reset_index()[["team", "player_id"]]), fill_value=0)
    table.columns = [c + 1 for c in table.columns]
    with st.expander("Гранаты по раундам (контекст для «раундов без гранат»)"):
        fig = px.imshow(
            table.to_numpy(),
            x=[str(c) for c in table.columns],
            y=[f"{team} · {player}" for team, player in table.index],
            text_auto=True, color_continuous_scale="Blues", aspect="auto", height=420,
            labels=dict(x="раунд", y="игрок", color="гранат"),
        )
        st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------- navigation

PAGES = {"Карта": page_map, "Раскидки": page_lineups, "Флешки": page_flashes, "Разбор матча": page_match}
names = list(PAGES)
start = st.query_params.get("page", names[0])  # pages are linkable: ?page=Раскидки
choice = st.sidebar.radio("Страница", names, index=names.index(start) if start in names else 0)
st.query_params["page"] = choice
st.sidebar.caption("10 000 матчей Premier/ММ на de_cache · только события гранат")
PAGES[choice]()
