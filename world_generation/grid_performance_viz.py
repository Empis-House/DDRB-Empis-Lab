"""
Grid Point Performance Visualizer
==================================
Visualizes iterative performance of 2D grid points as:
  - Animated heatmap (slider per iteration)
  - Performance curves per point
  - Summary detail table

Usage:
    python grid_performance_viz.py
    # or import and call plot_grid_performance(data, grid_points)
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── sample data ──────────────────────────────────────────────────────────────

GRID_POINTS = [[0, 0.5], [0, 1]]

RAW_PERFORMANCE = [
    {(0,   0):   np.array([0.04, 0.12, 0.17])},
    {(0,   1):   np.array([1.,   1.,   1.  ])},
    {(0.5, 0):   np.array([0.,   0.08, 0.21])},
    {(0.5, 1):   np.array([1.,   1.,   1.  ])},
]


# ── palette ───────────────────────────────────────────────────────────────────

POINT_COLORS = ["#185FA5", "#D85A30", "#1D9E75", "#7F77DD"]

HEATMAP_COLORSCALE = [
    [0.0,  "#E6F1FB"],
    [0.25, "#85B7EB"],
    [0.5,  "#378ADD"],
    [0.75, "#185FA5"],
    [1.0,  "#042C53"],
]


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_data(raw_performance):
    """Convert list-of-dicts format → list of (pt_tuple, perf_array)."""
    return [(list(d.keys())[0], list(d.values())[0]) for d in raw_performance]


def build_grid(data, xs, ys, iter_idx):
    """Return (ny × nx) numpy array of performance at a given iteration."""
    grid = np.full((len(ys), len(xs)), np.nan)
    for pt, perf in data:
        xi = xs.index(pt[0])
        yi = ys.index(pt[1])
        grid[yi, xi] = perf[iter_idx]
    return grid


def trend_symbol(perf):
    if perf[-1] > perf[0]:
        return "↑"
    elif perf[-1] < perf[0]:
        return "↓"
    return "→"


# ── main plot function ────────────────────────────────────────────────────────

def plot_grid_performance(raw_performance, grid_points=None, *, show=True):
    """
    Parameters
    ----------
    raw_performance : list[dict]
        Each dict maps one (x, y) tuple to a numpy array of performance values.
        Example: [{(0, 0): array([0.04, 0.12, 0.17])}, ...]
    grid_points : list[list], optional
        [[x_values], [y_values]] — inferred from data if not supplied.
    show : bool
        Call fig.show() when True (default). Set False to get the Figure back.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    data   = parse_data(raw_performance)
    n_iter = len(data[0][1])

    xs = sorted({pt[0] for pt, _ in data})
    ys = sorted({pt[1] for pt, _ in data})

    x_labels = [f"x = {x}" for x in xs]
    y_labels = [f"y = {y}" for y in ys]
    iter_labels = [f"iter {i + 1}" for i in range(n_iter)]

    # ── layout: 2 rows × 2 cols ───────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.72, 0.28],
        column_widths=[0.48, 0.52],
        subplot_titles=[
            "Performance heatmap",
            "Performance curves",
            "",            # detail table lives here
            "",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
        specs=[
            [{"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "table",    "colspan": 2}, None],
        ],
    )

    # ── trace 0 — heatmap (will be updated by frames) ────────────────────
    grid0 = build_grid(data, xs, ys, 0)

    fig.add_trace(
        go.Heatmap(
            z=grid0,
            x=x_labels,
            y=y_labels,
            colorscale=HEATMAP_COLORSCALE,
            zmin=0, zmax=1,
            text=[[f"{v:.2f}" if not np.isnan(v) else "—" for v in row]
                  for row in grid0],
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            colorbar=dict(
                x=0.46, len=0.72, thickness=14,
                tickfont=dict(size=11),
                title=dict(text="perf", side="right", font=dict(size=11)),
            ),
            name="heatmap",
            hovertemplate="point (%{x}, %{y})<br>performance: %{z:.3f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ── traces 1…N — one line per grid point ─────────────────────────────
    for i, (pt, perf) in enumerate(data):
        fig.add_trace(
            go.Scatter(
                x=list(range(1, n_iter + 1)),
                y=perf,
                mode="lines+markers",
                name=f"({pt[0]}, {pt[1]})",
                line=dict(color=POINT_COLORS[i % len(POINT_COLORS)], width=2.5),
                marker=dict(size=9, color=POINT_COLORS[i % len(POINT_COLORS)],
                            line=dict(width=1.5, color="white")),
                hovertemplate=(
                    f"point ({pt[0]}, {pt[1]})<br>"
                    "iteration %{x}<br>"
                    "performance: %{y:.3f}<extra></extra>"
                ),
            ),
            row=1, col=2,
        )

    # vertical marker line on curve chart (updated with frames)
    fig.add_trace(
        go.Scatter(
            x=[1, 1], y=[0, 1.05],
            mode="lines",
            line=dict(color="rgba(80,80,80,0.35)", width=1.5, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
            name="iter_marker",
        ),
        row=1, col=2,
    )

    # ── trace N+2 — summary table ─────────────────────────────────────────
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>point (x, y)</b>", "<b>peak perf</b>",
                        "<b>final perf</b>", "<b>trend</b>",
                        *[f"<b>iter {i+1}</b>" for i in range(n_iter)]],
                fill_color="#185FA5",
                font=dict(color="white", size=12),
                align="center",
                height=28,
            ),
            cells=dict(
                values=[
                    [f"({pt[0]}, {pt[1]})" for pt, _ in data],
                    [f"{perf.max():.3f}"    for _, perf in data],
                    [f"{perf[-1]:.3f}"      for _, perf in data],
                    [trend_symbol(perf)     for _, perf in data],
                    *[[f"{perf[i]:.3f}" for _, perf in data]
                      for i in range(n_iter)],
                ],
                fill_color=[
                    ["#E6F1FB"] * len(data),
                    ["#f7fafe"] * len(data),
                    ["#f7fafe"] * len(data),
                    ["#f7fafe"] * len(data),
                    *[["#f7fafe"] * len(data) for _ in range(n_iter)],
                ],
                font=dict(size=12),
                align="center",
                height=26,
            ),
        ),
        row=2, col=1,
    )

    # ── animation frames ──────────────────────────────────────────────────
    # trace indices: 0 = heatmap, 1…len(data) = lines, len(data)+1 = marker
    marker_trace_idx = len(data) + 1

    frames = []
    for k in range(n_iter):
        grid_k = build_grid(data, xs, ys, k)
        frame = go.Frame(
            name=str(k),
            traces=[0, marker_trace_idx],
            data=[
                go.Heatmap(
                    z=grid_k,
                    text=[[f"{v:.2f}" if not np.isnan(v) else "—" for v in row]
                          for row in grid_k],
                    texttemplate="%{text}",
                    textfont={"size": 14, "color": "white"},
                ),
                go.Scatter(x=[k + 1, k + 1], y=[0, 1.05]),
            ],
        )
        frames.append(frame)

    fig.frames = frames

    # ── slider ────────────────────────────────────────────────────────────
    sliders = [dict(
        active=0,
        currentvalue=dict(
            prefix="Iteration: ",
            font=dict(size=13, color="#333"),
            xanchor="left",
        ),
        pad=dict(t=55, b=10),
        len=0.90,
        x=0.05,
        steps=[
            dict(
                method="animate",
                label=str(k + 1),
                args=[
                    [str(k)],
                    dict(
                        frame=dict(duration=350, redraw=True),
                        transition=dict(duration=200),
                        mode="immediate",
                    ),
                ],
            )
            for k in range(n_iter)
        ],
    )]

    # ── layout polish ─────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="Grid point performance explorer",
            font=dict(size=17, color="#1a1a1a"),
            x=0.02,
        ),
        height=680,
        template="plotly_white",
        sliders=sliders,
        legend=dict(
            x=1.01, y=0.78,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ddd",
            borderwidth=0.5,
            font=dict(size=12),
        ),
        margin=dict(l=55, r=40, t=80, b=110),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        font=dict(family="'Helvetica Neue', Arial, sans-serif"),
    )

    # axes
    fig.update_xaxes(title_text="X coordinate", row=1, col=1,
                     showgrid=False, tickfont=dict(size=11))
    fig.update_yaxes(title_text="Y coordinate", row=1, col=1,
                     showgrid=False, tickfont=dict(size=11))

    fig.update_xaxes(title_text="Iteration", row=1, col=2,
                     tickmode="linear", tick0=1, dtick=1,
                     showgrid=True, gridcolor="#eee", tickfont=dict(size=11))
    fig.update_yaxes(title_text="Performance", range=[-0.02, 1.08],
                     row=1, col=2,
                     showgrid=True, gridcolor="#eee", tickfont=dict(size=11))

    if show:
        fig.show()

    return fig


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig = plot_grid_performance(RAW_PERFORMANCE, GRID_POINTS)
    # Optionally save as self-contained HTML:
    # fig.write_html("grid_performance.html")
