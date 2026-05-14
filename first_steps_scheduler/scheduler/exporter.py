"""
Export schedule results to CSV and Folium map.
"""

from __future__ import annotations

import io
from typing import Optional

import folium
import pandas as pd

from .optimizer import DayRoute, ScheduleResult, ScheduledSession

# Day → color for map routes
_DAY_COLORS: dict[str, str] = {
    "Mon": "#e6194b",   # red
    "Tue": "#3cb44b",   # green
    "Wed": "#4363d8",   # blue
    "Thu": "#f58231",   # orange
}

# Indianapolis center for default map view
_INDY_CENTER = (39.7684, -86.1581)


def schedule_to_dataframe(result: ScheduleResult) -> pd.DataFrame:
    """Flatten a ScheduleResult into a tidy DataFrame for display or CSV export."""
    rows: list[dict] = []
    for day_route in result.days:
        for s in day_route.sessions:
            rows.append(
                {
                    "Day": s.day,
                    "Start": s.start_str,
                    "End": s.end_str,
                    "Client": s.client_id,
                    "Variant": result.variant.value,
                }
            )
    if not rows:
        return pd.DataFrame(columns=["Day", "Start", "End", "Client", "Variant"])
    df = pd.DataFrame(rows)
    # Sort by day order then start time
    day_order = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3}
    df["_day_sort"] = df["Day"].map(day_order)
    df = df.sort_values(["_day_sort", "Start"]).drop(columns="_day_sort")
    return df.reset_index(drop=True)


def schedule_to_csv_bytes(result: ScheduleResult) -> bytes:
    """Return UTF-8 CSV bytes for download."""
    df = schedule_to_dataframe(result)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def schedule_to_weekly_grid(result: ScheduleResult) -> pd.DataFrame:
    """Produce a wide-format grid: rows = time slots, columns = Mon–Thu."""
    day_order = ["Mon", "Tue", "Wed", "Thu"]

    # Collect all sessions per day
    day_sessions: dict[str, list[ScheduledSession]] = {d: [] for d in day_order}
    for dr in result.days:
        day_sessions[dr.day] = sorted(dr.sessions, key=lambda s: s.start_min)

    # Find max sessions in any day
    max_sessions = max((len(v) for v in day_sessions.values()), default=0)
    if max_sessions == 0:
        return pd.DataFrame(columns=day_order)

    grid: dict[str, list[str]] = {d: [] for d in day_order}
    for d in day_order:
        sessions = day_sessions[d]
        for s in sessions:
            grid[d].append(f"{s.start_str} {s.client_id}")
        # Pad short columns
        while len(grid[d]) < max_sessions:
            grid[d].append("")

    return pd.DataFrame(grid)


def build_map(
    result: ScheduleResult,
    locations: dict[str, tuple[float, float]],  # client_id → (lat, lon)
    home_base: tuple[float, float],
) -> folium.Map:
    """Build a Folium map showing all daily routes color-coded by day.

    *locations* maps client_id → (lat, lon).
    *home_base* is (lat, lon).
    """
    m = folium.Map(location=_INDY_CENTER, zoom_start=11, tiles="CartoDB positron")

    # Home base marker
    folium.Marker(
        location=home_base,
        popup="Home Base",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    for day_route in result.days:
        if not day_route.sessions:
            continue
        color = _DAY_COLORS.get(day_route.day, "gray")

        # Build ordered waypoints: home → clients → home
        waypoints: list[tuple[float, float]] = [home_base]
        for s in sorted(day_route.sessions, key=lambda x: x.start_min):
            if s.client_id in locations:
                waypoints.append(locations[s.client_id])
        waypoints.append(home_base)

        # Draw route polyline
        folium.PolyLine(
            locations=waypoints,
            color=color,
            weight=2.5,
            opacity=0.8,
            tooltip=day_route.day,
        ).add_to(m)

        # Client markers
        for i, s in enumerate(sorted(day_route.sessions, key=lambda x: x.start_min)):
            if s.client_id not in locations:
                continue
            lat, lon = locations[s.client_id]
            folium.CircleMarker(
                location=(lat, lon),
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(
                    f"<b>{s.client_id}</b><br>{day_route.day} {s.start_str}–{s.end_str}",
                    max_width=180,
                ),
                tooltip=f"{s.client_id} ({day_route.day} {s.start_str})",
            ).add_to(m)

    # Legend
    legend_html = _build_legend_html()
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def _build_legend_html() -> str:
    items = "".join(
        f'<li><span style="background:{c};display:inline-block;'
        f'width:14px;height:14px;margin-right:6px;border-radius:2px;"></span>{d}</li>'
        for d, c in _DAY_COLORS.items()
    )
    return f"""
    <div style="position:fixed;bottom:30px;right:30px;z-index:1000;
                background:white;padding:10px 14px;border-radius:6px;
                box-shadow:0 2px 6px rgba(0,0,0,.3);font-size:13px;font-family:sans-serif;">
      <b>Day</b>
      <ul style="list-style:none;margin:4px 0 0;padding:0">{items}</ul>
    </div>"""


def build_geocoding_preview_map(
    locations: dict[str, tuple[float, float]],  # client_id → (lat, lon)
    home_base: tuple[float, float],
) -> folium.Map:
    """Simple verification map showing all geocoded pins."""
    m = folium.Map(location=_INDY_CENTER, zoom_start=11, tiles="CartoDB positron")

    folium.Marker(
        location=home_base,
        popup="Home Base",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    for client_id, (lat, lon) in locations.items():
        folium.Marker(
            location=(lat, lon),
            popup=client_id,
            icon=folium.Icon(color="blue", icon="user", prefix="fa"),
        ).add_to(m)

    return m
