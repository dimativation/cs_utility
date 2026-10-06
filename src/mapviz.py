"""Cache map drawn from the data itself.

Every known player position (throw points, flashed players, damaged players)
goes into a 2D histogram; its density outlines the walkable area. The image is
placed on charts in game coordinates, so no game->pixel transform is needed.

Build once:  python src/mapviz.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

DATA = Path(__file__).resolve().parent.parent / "data"
MAP_PNG = DATA / "map.png"
MAP_META = DATA / "map_meta.npz"

BIN = 8  # game units per pixel
X_RANGE = (-1900, 3400)
Y_RANGE = (-1600, 2400)
Z_MIN = 1500  # a few positions have z=0 (broken records)


def _xyz(path, columns):
    df = pd.read_parquet(path, columns=columns)
    df.columns = ["x", "y", "z"]
    return df


def load_positions():
    pos = pd.concat([
        _xyz(DATA / "throws.parquet", ["throw_x", "throw_y", "throw_z"]),
        _xyz(DATA / "flash_victims.parquet", ["pos_x", "pos_y", "pos_z"]),
        _xyz(DATA / "damage.parquet", ["pos_x", "pos_y", "pos_z"]),
    ])
    return pos[pos.z > Z_MIN]


def _hist2d(x, y, bin_size, extent):
    x0, x1, y0, y1 = extent
    x_edges = np.arange(x0, x1 + bin_size, bin_size)
    y_edges = np.arange(y0, y1 + bin_size, bin_size)
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    return x_edges, y_edges, counts.T  # rows = y, as imshow/heatmap expect


def build_map():
    pos = load_positions()
    _, _, counts = _hist2d(pos.x, pos.y, BIN, (*X_RANGE, *Y_RANGE))
    img = np.log1p(counts)[::-1]  # flip y: image row 0 is the top of the map
    img = img / img.max()
    plt.imsave(MAP_PNG, img, cmap="gray", vmin=0, vmax=1)
    np.savez(MAP_META, extent=np.array([X_RANGE[0], X_RANGE[1], Y_RANGE[0], Y_RANGE[1]]))
    print(f"map from {len(pos):,} positions -> {MAP_PNG} ({img.shape[1]}x{img.shape[0]})")


def _extent():
    return np.load(MAP_META)["extent"]


def draw_map(ax=None, alpha=1.0):
    """Matplotlib: map in game coordinates on `ax`."""
    ax = ax or plt.gca()
    ax.imshow(plt.imread(MAP_PNG), extent=_extent(), alpha=alpha)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def draw_density(ax, x, y, cmap="hot", bin_size=24):
    """Matplotlib: map + log-density of points (empty cells transparent)."""
    draw_map(ax, alpha=0.5)
    _, _, counts = _hist2d(x, y, bin_size, _extent())
    dens = np.ma.masked_equal(np.log1p(counts), 0)
    ax.imshow(dens, extent=_extent(), origin="lower", cmap=cmap, alpha=0.85)
    return ax


def draw_lineups(ax, catalog, throws, max_points=300):
    """Matplotlib: map + lineups (throw dots, landing crosses, arrow centre->centre).
    catalog: rows of lineups.parquet; throws: throws joined with lineup_id."""
    draw_map(ax, alpha=0.6)
    colors = plt.cm.tab20(np.arange(len(catalog)) % 20)
    for (lineup_id, row), color in zip(catalog.iterrows(), colors):
        pts = throws[throws.lineup_id == lineup_id]
        pts = pts.sample(min(max_points, len(pts)), random_state=0)
        ax.scatter(pts.throw_x, pts.throw_y, s=3, color=color)
        ax.scatter(pts.land_x, pts.land_y, s=3, color=color, marker="x")
        ax.annotate("", xy=(row.land_cx, row.land_cy), xytext=(row.throw_cx, row.throw_cy),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        ax.text(row.land_cx, row.land_cy, f"{lineup_id.split('-')[-1]} n={row.n} прыжок={row.airborne:.0%}",
                fontsize=8, color="white", bbox=dict(fc=color, alpha=0.8, lw=0))
    return ax


def map_figure(height=700):
    """Plotly: empty figure with the map as background, axes in game coordinates."""
    from PIL import Image

    x0, x1, y0, y1 = _extent()
    fig = go.Figure()
    fig.add_layout_image(source=Image.open(MAP_PNG), xref="x", yref="y", x=x0, y=y1,
                         sizex=x1 - x0, sizey=y1 - y0, sizing="stretch", layer="below")
    fig.update_xaxes(range=[x0, x1], visible=False)
    fig.update_yaxes(range=[y0, y1], visible=False, scaleanchor="x")
    fig.update_layout(height=height, plot_bgcolor="black", margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
    return fig


def add_density(fig, x, y, bin_size=24, colorscale="Hot"):
    """Plotly: log-density heatmap over the map, empty cells transparent."""
    x_edges, y_edges, counts = _hist2d(x, y, bin_size, _extent())
    z = np.where(counts > 0, np.log1p(counts), np.nan)
    fig.add_trace(go.Heatmap(
        x=x_edges[:-1] + bin_size / 2,
        y=y_edges[:-1] + bin_size / 2,
        z=z, colorscale=colorscale, opacity=0.8, showscale=False, hoverinfo="skip",
    ))
    return fig


def add_arrow(fig, x0, y0, x1, y1, color, width=2):
    """Plotly: arrow from (x0, y0) to (x1, y1) in game coordinates."""
    fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", text="",
                       showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=width, arrowcolor=color)
    return fig


if __name__ == "__main__":
    build_map()
