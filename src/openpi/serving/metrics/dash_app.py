"""Dash-based metrics dashboard with plotly-resampler for dynamic chart resampling.

FigureResampler is used for time-series scatter charts (batch sizes, GPU busy) so that
the browser only receives ~2000 points on load, and re-samples to full resolution on zoom.
Uses plotly-resampler 0.10+ API: construct_update_data_patch returns a Patch directly.
"""

from __future__ import annotations

import datetime
import uuid

import dash
from dash import Input
from dash import Output
from dash import Patch
from dash import State
from dash import ctx
from dash import dcc
from dash import html
import numpy as np
from openpi_client.schemas import ServerMetadata
import plotly.graph_objects as go
from plotly_resampler import FigureResampler

from openpi.serving.metrics.store import MetricsStore

# ---------------------------------------------------------------------------
# Server-side resampler cache  {session_id -> {chart_key -> FigureResampler}}
# ---------------------------------------------------------------------------
_resamplers: dict[str, dict[str, FigureResampler]] = {}

# ---------------------------------------------------------------------------
# Plotly dark theme helpers
# ---------------------------------------------------------------------------
_DARK: dict = {
    "paper_bgcolor": "#1a1a1a",
    "plot_bgcolor": "#111",
    "font": {"color": "#bbb", "family": "'SF Mono', monospace", "size": 11},
    "margin": {"t": 32, "r": 16, "b": 48, "l": 56},
    "xaxis": {"gridcolor": "#222", "zerolinecolor": "#333"},
    "yaxis": {"gridcolor": "#222", "zerolinecolor": "#333"},
    "legend": {"bgcolor": "#1a1a1a", "bordercolor": "#333"},
}


def _layout(**extra: object) -> dict:
    return {
        **_DARK,
        **extra,
        "xaxis": {**_DARK["xaxis"], **extra.get("xaxis", {})},  # type: ignore[arg-type]
        "yaxis": {**_DARK["yaxis"], **extra.get("yaxis", {})},  # type: ignore[arg-type]
    }


# ---------------------------------------------------------------------------
# Shared styles
# ---------------------------------------------------------------------------
_BODY = {
    "background": "#111",
    "color": "#ccc",
    "fontFamily": "'SF Mono', 'Fira Code', monospace",
    "padding": "24px",
    "fontSize": "14px",
    "minHeight": "100vh",
}
_CARD = {
    "background": "#1a1a1a",
    "border": "1px solid #222",
    "borderRadius": "6px",
    "overflow": "hidden",
    "marginBottom": "12px",
}
_CARD_HDR = {
    "display": "flex",
    "alignItems": "center",
    "gap": "10px",
    "padding": "10px 14px",
    "borderBottom": "1px solid #222",
    "fontSize": "0.8em",
    "color": "#666",
}
_SECTION_TITLE = {
    "fontSize": "0.7em",
    "textTransform": "uppercase",
    "letterSpacing": "1.5px",
    "color": "#555",
    "borderBottom": "1px solid #1e1e1e",
    "paddingBottom": "5px",
    "marginBottom": "12px",
}
_STAT_CARD = {
    "background": "#1a1a1a",
    "border": "1px solid #222",
    "borderRadius": "6px",
    "padding": "14px 16px",
}
_INPUT = {
    "background": "#1e1e1e",
    "border": "1px solid #333",
    "color": "#ccc",
    "padding": "3px 8px",
    "borderRadius": "3px",
    "fontFamily": "inherit",
    "fontSize": "0.95em",
}
_BTN_PRIMARY = {
    "background": "#0d2d4a",
    "border": "1px solid #2979ff",
    "color": "#82b1ff",
    "padding": "6px 16px",
    "borderRadius": "4px",
    "cursor": "pointer",
    "fontFamily": "inherit",
    "fontSize": "0.85em",
}
_CFG = {"displaylogo": False, "responsive": True}


# ---------------------------------------------------------------------------
# Small layout helpers
# ---------------------------------------------------------------------------
def _section(title: str, *children: object) -> html.Div:
    return html.Div(
        style={"marginBottom": "28px"},
        children=[html.Div(title.upper(), style=_SECTION_TITLE), *children],
    )


def _stat_card(value: str, label: str) -> html.Div:
    return html.Div(
        style=_STAT_CARD,
        children=[
            html.Div(value, style={"fontSize": "1.9em", "fontWeight": 700, "color": "#4fc3f7", "lineHeight": "1.1"}),
            html.Div(label, style={"color": "#555", "fontSize": "0.72em", "marginTop": "5px"}),
        ],
    )


def _robot_table(robots: dict) -> html.Element:
    if not robots:
        return html.Div("No robots yet.", style={"color": "#444", "padding": "12px"})
    th = {
        "textAlign": "left",
        "color": "#555",
        "fontWeight": 400,
        "padding": "6px 12px",
        "borderBottom": "1px solid #222",
    }
    td_base = {"padding": "6px 12px"}
    return html.Table(
        style={"width": "100%", "borderCollapse": "collapse", "fontSize": "0.85em"},
        children=[
            html.Thead(
                html.Tr(
                    [
                        html.Th("Robot", style=th),
                        html.Th("Starvations", style=th),
                        html.Th("Avg Net Delay (ms)", style=th),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(rid, style={**td_base, "color": "#ccc"}),
                            html.Td(
                                str(r["total_starvations"]),
                                style={
                                    **td_base,
                                    "color": "#ff8a65" if r["total_starvations"] > 0 else "#4fc3f7",
                                    "fontWeight": 600,
                                },
                            ),
                            html.Td(
                                f"{r['avg_network_delay_ms']:.2f}",
                                style={**td_base, "color": "#4fc3f7", "fontWeight": 600},
                            ),
                        ]
                    )
                    for rid, r in robots.items()
                ]
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Non-resampled figure builders
# ---------------------------------------------------------------------------
def _gpu_dist_fig(batches: list[dict]) -> go.Figure:
    fig = go.Figure()
    by_size: dict[int, list[float]] = {}
    for b in batches:
        by_size.setdefault(b["batch_size"], []).append(b["gpu_time_ms"])
    for s in sorted(by_size):
        fig.add_trace(
            go.Box(
                y=by_size[s],
                name=str(s),
                boxpoints="outliers",
                jitter=0.4,
                pointpos=0,
                marker_size=3,
                marker_opacity=0.5,
                line_width=1.5,
            )
        )
    fig.update_layout(**_layout(xaxis={"title": "Batch size"}, yaxis={"title": "GPU time (ms)"}, showlegend=False))
    return fig


def _gantt_fig(batches: list[dict], window_s: float) -> go.Figure:
    fig = go.Figure()
    if not batches:
        return fig
    max_t = batches[-1]["t"]
    visible = [b for b in batches if b["t"] >= max_t - window_s]
    all_robots = sorted({rid for b in visible for rid in b["robot_ids"]})
    palette = [
        "#4fc3f7",
        "#81c784",
        "#ff8a65",
        "#ce93d8",
        "#ffb74d",
        "#f06292",
        "#4db6ac",
        "#aed581",
        "#7986cb",
        "#4dd0e1",
    ]
    rc = {r: palette[i % len(palette)] for i, r in enumerate(all_robots)}
    tmap: dict[str, dict] = {}
    for b in visible:
        dur = b["inference_end_t"] - b["inference_start_t"]
        for rid in b["robot_ids"]:
            if rid not in tmap:
                tmap[rid] = {"x": [], "base": [], "y": [], "color": rc[rid]}
            tmap[rid]["x"].append(dur)
            tmap[rid]["base"].append(b["inference_start_t"])
            tmap[rid]["y"].append(rid)
    for rid, td in tmap.items():
        fig.add_trace(
            go.Bar(x=td["x"], base=td["base"], y=td["y"], orientation="h", name=rid, marker_color=td["color"])
        )
    fig.update_layout(
        **_layout(
            barmode="overlay",
            height=max(200, len(all_robots) * 32 + 80),
            xaxis={"title": "Time since server start (s)"},
            yaxis={"autorange": "reversed"},
            showlegend=False,
        )
    )
    return fig


def _stage_figs(hist: dict, robot: str) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    inbound, queue_, infer_, outbound = [], [], [], []
    for b in hist["batches"]:
        for req in b["per_request"]:
            if robot != "all" and req["robot_id"] != robot:
                continue
            inbound.append(req["inbound_ms"])
            queue_.append(req["queue_ms"])
            infer_.append(req["infer_ms"])
    od = hist.get("outbound_delays_ms", {})
    outbound = list(od.get(robot) or []) if robot != "all" else [d for v in od.values() for d in v]

    small = {**_DARK, "margin": {"t": 36, "r": 8, "b": 44, "l": 48}}
    figs = []
    for data, color, title in [
        (inbound, "#4fc3f7", "Inbound Network"),
        (queue_, "#ff8a65", "Queue Wait"),
        (infer_, "#81c784", "Inference"),
        (outbound, "#ce93d8", "Outbound Network"),
    ]:
        f = go.Figure()
        if data:
            f.add_trace(go.Histogram(x=data, nbinsx=100, marker_color=color, marker_opacity=0.85, name=title))
        f.update_layout(
            **{
                **small,
                "title": {"text": title, "font": {"size": 12, "color": "#888"}},
                "xaxis": {**_DARK["xaxis"], "title": "ms"},
                "yaxis": {**_DARK["yaxis"], "title": "count"},
            }
        )
        figs.append(f)
    return tuple(figs)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_dash_app(metadata: ServerMetadata, metrics_store: MetricsStore) -> dash.Dash:
    app = dash.Dash(
        __name__,
        url_base_pathname="/dashboard/",
        title="openpi · metrics",
        suppress_callback_exceptions=True,
    )

    meta_line = (
        f"config: {metadata.config_name}  ·  env: {metadata.env}  ·  "
        f"max_batch: {metadata.max_batch_size}  ·  "
        f"location: {metadata.location or 'unknown'}  ·  "
        f"checkpoint: {metadata.checkpoint_dir}"
    )

    app.layout = html.Div(
        style=_BODY,
        children=[
            dcc.Store(id="store-session", storage_type="session"),
            dcc.Store(id="store-xrange"),
            # Header
            html.H1(
                "openpi · metrics",
                style={"fontSize": "1.3em", "fontWeight": 700, "color": "#fff", "marginBottom": "2px"},
            ),
            html.Div(id="div-subtitle", style={"color": "#555", "fontSize": "0.8em", "marginBottom": "4px"}),
            html.Div(
                meta_line,
                style={"color": "#444", "fontSize": "0.75em", "marginBottom": "20px", "fontFamily": "inherit"},
            ),
            # Toolbar
            html.Div(
                style={
                    "display": "flex",
                    "gap": "8px",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                    "marginBottom": "24px",
                },
                children=[
                    html.Button("⟳ Refresh", id="btn-refresh", n_clicks=0, style=_BTN_PRIMARY),
                    html.Label("Last", style={"color": "#555", "fontSize": "0.85em"}),
                    dcc.Input(
                        id="input-window",
                        type="number",
                        placeholder="all",
                        min=1,
                        debounce=True,
                        style={**_INPUT, "width": "80px", "padding": "5px 8px"},
                    ),
                    html.Label("seconds", style={"color": "#555", "fontSize": "0.85em"}),
                    html.Span(id="span-status", style={"color": "#555", "fontSize": "0.8em", "marginLeft": "4px"}),
                ],
            ),
            # Summary
            _section(
                "Summary",
                html.Div(
                    id="div-stats",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fill, minmax(150px, 1fr))",
                        "gap": "8px",
                    },
                ),
            ),
            # Per-Robot
            _section("Per-Robot", html.Div(id="div-robots")),
            # Charts
            _section(
                "Charts",
                # GPU inference dist
                html.Div(
                    style=_CARD,
                    children=[
                        html.Div("GPU Inference Time by Batch Size", style=_CARD_HDR),
                        dcc.Graph(id="graph-gpu-dist", config=_CFG),
                    ],
                ),
                # Stage latency
                html.Div(
                    style=_CARD,
                    children=[
                        html.Div(
                            style=_CARD_HDR,
                            children=[
                                html.Span("Stage Latency Distributions"),
                                html.Label("Robot:", style={"color": "#555"}),
                                dcc.Dropdown(
                                    id="dd-robot",
                                    options=[{"label": "all", "value": "all"}],
                                    value="all",
                                    clearable=False,
                                    style={"width": "160px", "fontSize": "0.9em"},
                                ),
                            ],
                        ),
                        html.Div(
                            style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)"},
                            children=[
                                dcc.Graph(id="graph-inbound", config=_CFG),
                                dcc.Graph(id="graph-queue", config=_CFG),
                                dcc.Graph(id="graph-infer", config=_CFG),
                                dcc.Graph(id="graph-outbound", config=_CFG),
                            ],
                        ),
                    ],
                ),
                # Batch sizes (resampled)
                html.Div(
                    style=_CARD,
                    children=[
                        html.Div("Batch Sizes Over Time", style=_CARD_HDR),
                        dcc.Graph(id="graph-batch", config=_CFG),
                    ],
                ),
                # GPU busy (resampled)
                html.Div(
                    style=_CARD,
                    children=[
                        html.Div("GPU Busy (%)", style=_CARD_HDR),
                        dcc.Graph(id="graph-busy", config=_CFG),
                    ],
                ),
                # Gantt
                html.Div(
                    style=_CARD,
                    children=[
                        html.Div("GPU Gantt", style=_CARD_HDR),
                        dcc.Graph(id="graph-gantt", config=_CFG),
                    ],
                ),
            ),
        ],
    )

    # -------------------------------------------------------------------------
    # Session init
    # -------------------------------------------------------------------------
    @app.callback(Output("store-session", "data"), Input("store-session", "data"))
    def _init_session(data: dict | None) -> dict:
        return data if data else {"id": str(uuid.uuid4())}

    # -------------------------------------------------------------------------
    # Main refresh: stats, robots, GPU dist, Gantt
    # -------------------------------------------------------------------------
    @app.callback(
        Output("div-subtitle", "children"),
        Output("div-stats", "children"),
        Output("div-robots", "children"),
        Output("graph-gpu-dist", "figure"),
        Output("graph-gantt", "figure"),
        Output("dd-robot", "options"),
        Output("span-status", "children"),
        Input("btn-refresh", "n_clicks"),
        State("input-window", "value"),
    )
    def _refresh_main(
        n_clicks: int,
        window_s: float | None,
    ) -> tuple:
        snap = metrics_store.snapshot(window_s)
        hist = metrics_store.history(window_s)
        batches = hist["batches"]

        def f(v: float) -> str:
            return f"{v:.1f}"

        subtitle = (
            f"uptime {f(snap['uptime_s'])}s · {snap['total_requests']:,} requests · {snap['total_batches']:,} batches"
        )

        stats = [
            ("total requests", f"{snap['total_requests']:,}"),
            ("req / s", f(snap["requests_per_second"])),
            ("p50 latency (ms)", f(snap["p50_latency_ms"])),
            ("p99 latency (ms)", f(snap["p99_latency_ms"])),
            ("avg GPU time (ms)", f(snap["avg_gpu_time_ms"])),
            ("GPU busy (%)", f"{snap['gpu_busy_pct']:.1f}%"),
            ("avg queue delay (ms)", f(snap["avg_queue_delay_ms"])),
            ("total batches", f"{snap['total_batches']:,}"),
        ]
        stat_cards = [_stat_card(v, lbl) for lbl, v in stats]

        robots = snap.get("per_robot", {})
        robot_el = _robot_table(robots)

        gpu_fig = _gpu_dist_fig(batches)
        gantt = _gantt_fig(batches, float(window_s) if window_s else float("inf"))

        robot_opts = [{"label": "all", "value": "all"}] + [{"label": rid, "value": rid} for rid in robots]
        status = "last refresh: " + datetime.datetime.now(datetime.UTC).astimezone().strftime("%H:%M:%S")

        return subtitle, stat_cards, robot_el, gpu_fig, gantt, robot_opts, status

    # -------------------------------------------------------------------------
    # Batch sizes: initial load (resampled)
    # -------------------------------------------------------------------------
    @app.callback(
        Output("graph-batch", "figure"),
        Input("btn-refresh", "n_clicks"),
        State("input-window", "value"),
        State("store-session", "data"),
    )
    def _load_batch(n_clicks: int, window_s: float | None, session_data: dict | None) -> FigureResampler:
        sid = (session_data or {}).get("id", "default")
        hist = metrics_store.history(window_s)
        batches = hist["batches"]

        fr = FigureResampler(go.Figure(), default_n_shown_samples=2000)
        if batches:
            times = np.array([b["t"] for b in batches], dtype=float)
            sizes = np.array([b["batch_size"] for b in batches], dtype=float)
            fr.add_trace(
                go.Scatter(mode="markers", name="batch size", marker={"size": 4, "color": "#ce93d8", "opacity": 0.6}),
                hf_x=times,
                hf_y=sizes,
            )
        fr.update_layout(**_layout(xaxis={"title": "Time since server start (s)"}, yaxis={"title": "Batch size"}))
        _resamplers.setdefault(sid, {})["batch"] = fr
        return fr

    # Batch sizes: zoom/pan update
    @app.callback(
        Output("graph-batch", "figure", allow_duplicate=True),
        Input("graph-batch", "relayoutData"),
        State("store-session", "data"),
        prevent_initial_call=True,
    )
    def _resample_batch(relayout_data: dict | None, session_data: dict | None) -> Patch:
        sid = (session_data or {}).get("id", "default")
        fr = _resamplers.get(sid, {}).get("batch")
        if fr is None or not relayout_data:
            raise dash.exceptions.PreventUpdate
        return fr.construct_update_data_patch(relayout_data)

    # -------------------------------------------------------------------------
    # GPU busy: initial load (resampled)
    # -------------------------------------------------------------------------
    @app.callback(
        Output("graph-busy", "figure"),
        Input("btn-refresh", "n_clicks"),
        State("input-window", "value"),
        State("store-session", "data"),
    )
    def _load_busy(n_clicks: int, window_s: float | None, session_data: dict | None) -> FigureResampler:
        sid = (session_data or {}).get("id", "default")
        hist = metrics_store.history(window_s)
        batches = hist["batches"]

        fr = FigureResampler(go.Figure(), default_n_shown_samples=2000)
        if len(batches) >= 2:
            t0 = batches[0]["inference_start_t"]
            t1 = batches[-1]["inference_end_t"]
            n = int(t1 - t0) + 1
            pct = [0.0] * n
            for b in batches:
                s = b["inference_start_t"] - t0
                e = b["inference_end_t"] - t0
                lo, hi = int(s), min(int(e), n - 1)
                for k in range(lo, hi + 1):
                    ov = min(e, k + 1) - max(s, k)
                    if ov > 0:
                        pct[k] += ov * 100
            times = np.array([t0 + i + 0.5 for i in range(n)], dtype=float)
            fr.add_trace(
                go.Scatter(
                    mode="lines",
                    fill="tozeroy",
                    line={"color": "#4fc3f7", "width": 1.5},
                    fillcolor="rgba(79,195,247,0.12)",
                    name="busy %",
                ),
                hf_x=times,
                hf_y=np.array(pct, dtype=float),
            )
        fr.update_layout(
            **_layout(
                xaxis={"title": "Time since server start (s)"}, yaxis={"title": "GPU busy (%)", "range": [0, 100]}
            )
        )
        _resamplers.setdefault(sid, {})["busy"] = fr
        return fr

    # GPU busy: zoom/pan update
    @app.callback(
        Output("graph-busy", "figure", allow_duplicate=True),
        Input("graph-busy", "relayoutData"),
        State("store-session", "data"),
        prevent_initial_call=True,
    )
    def _resample_busy(relayout_data: dict | None, session_data: dict | None) -> Patch:
        sid = (session_data or {}).get("id", "default")
        fr = _resamplers.get(sid, {}).get("busy")
        if fr is None or not relayout_data:
            raise dash.exceptions.PreventUpdate
        return fr.construct_update_data_patch(relayout_data)

    # -------------------------------------------------------------------------
    # X-axis sync: batch / busy / gantt share the same time axis
    # -------------------------------------------------------------------------
    @app.callback(
        Output("store-xrange", "data"),
        Input("graph-batch", "relayoutData"),
        Input("graph-busy", "relayoutData"),
        Input("graph-gantt", "relayoutData"),
        prevent_initial_call=True,
    )
    def _capture_xrange(batch_relay: dict | None, busy_relay: dict | None, gantt_relay: dict | None) -> list | None:
        relay_map = {
            "graph-batch": batch_relay,
            "graph-busy": busy_relay,
            "graph-gantt": gantt_relay,
        }
        relay = relay_map.get(ctx.triggered_id)
        if relay and "xaxis.range[0]" in relay:
            return [relay["xaxis.range[0]"], relay["xaxis.range[1]"]]
        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output("graph-batch", "figure", allow_duplicate=True),
        Output("graph-busy", "figure", allow_duplicate=True),
        Output("graph-gantt", "figure", allow_duplicate=True),
        Input("store-xrange", "data"),
        prevent_initial_call=True,
    )
    def _apply_xrange(xrange: list | None) -> tuple:
        if not xrange:
            raise dash.exceptions.PreventUpdate
        p = Patch()
        p["layout"]["xaxis"]["range"] = xrange
        return p, p, p

    # -------------------------------------------------------------------------
    # Stage latency distributions
    # -------------------------------------------------------------------------
    @app.callback(
        Output("graph-inbound", "figure"),
        Output("graph-queue", "figure"),
        Output("graph-infer", "figure"),
        Output("graph-outbound", "figure"),
        Input("btn-refresh", "n_clicks"),
        Input("dd-robot", "value"),
        State("input-window", "value"),
    )
    def _update_stage(
        n_clicks: int,
        robot: str,
        window_s: float | None,
    ) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
        hist = metrics_store.history(window_s)
        return _stage_figs(hist, robot or "all")

    # Histogram zoom: refine bin size when user zooms in
    def _register_histogram_zoom(gid: str) -> None:
        @app.callback(
            Output(gid, "figure", allow_duplicate=True),
            Input(gid, "relayoutData"),
            prevent_initial_call=True,
        )
        def _zoom_histogram(relay: dict | None) -> Patch:
            if not relay or "xaxis.range[0]" not in relay:
                raise dash.exceptions.PreventUpdate
            x0 = float(relay["xaxis.range[0]"])
            x1 = float(relay["xaxis.range[1]"])
            if x1 <= x0:
                raise dash.exceptions.PreventUpdate
            p = Patch()
            p["data"][0]["xbins"]["size"] = (x1 - x0) / 150
            return p

    for _gid in ["graph-inbound", "graph-queue", "graph-infer", "graph-outbound"]:
        _register_histogram_zoom(_gid)

    return app
