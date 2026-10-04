"""
First Steps SLP Schedule Optimizer — Streamlit app.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_folium import st_folium

from scheduler.distance_matrix import MatrixError, build_matrix
from scheduler.exporter import (
    build_geocoding_preview_map,
    build_map,
    schedule_to_csv_bytes,
    schedule_to_weekly_grid,
)
from scheduler.geocoder import Geocoder, GeocoderError
from scheduler.optimizer import Client, LunchVariant, ScheduleResult, solve
from scheduler.parser import AvailabilityResult, parse_availability

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

HOME_BASE_ADDRESS = "W Southport Rd & S Harding St, Indianapolis, IN"
HOME_BASE_LATLON = (39.6595, -86.1746)  # pre-geocoded fallback (lat, lon)
SAMPLE_CSV_PATH = Path(__file__).parent / "data" / "sample_clients.csv"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="First Steps Scheduler",
    page_icon="🗓️",
    layout="wide",
)


# ── Session state helpers ─────────────────────────────────────────────────────


def _init_state() -> None:
    defaults: dict = {
        "clients_df": None,
        "parsed_clients": None,
        "geocoded": None,
        "home_latlon": HOME_BASE_LATLON,
        "travel_matrix": None,
        "results": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Helper functions (all defined before use) ─────────────────────────────────


def _require_api_key(api_key: str) -> bool:
    if not api_key:
        st.error("Please enter your ORS API key in the sidebar to continue.")
        return False
    return True


def _parse_clients(df: pd.DataFrame) -> list[Client]:
    """Parse a client DataFrame into Client objects with availability windows."""
    clients: list[Client] = []
    for i, row in df.iterrows():
        avail_raw = str(row.get("availability", "any"))
        result: AvailabilityResult = parse_availability(avail_raw)
        clients.append(
            Client(
                client_id=str(row["client_id"]),
                availability=result,
                location_index=i + 1,  # 0 = home base
                notes=str(row.get("notes", "")),
            )
        )
    return clients


def _geocode_all(
    key: str,
    clients: list[Client],
    df: pd.DataFrame,
) -> tuple[dict[str, tuple[float, float]], list[str]]:
    """Return (geocoded dict, error list).  geocoded maps client_id → (lat, lon)."""
    geocoder = Geocoder(api_key=key)
    geocoded: dict[str, tuple[float, float]] = {}
    errors: list[str] = []

    try:
        lon, lat = geocoder.geocode(HOME_BASE_ADDRESS)
        st.session_state["home_latlon"] = (lat, lon)
    except GeocoderError:
        pass  # keep fallback

    for client, (_, row) in zip(clients, df.iterrows()):
        address = str(row.get("intersection", ""))
        if not address:
            errors.append(f"{client.client_id}: missing intersection")
            continue
        try:
            lon, lat = geocoder.geocode(address)
            geocoded[client.client_id] = (lat, lon)
        except GeocoderError as e:
            errors.append(f"{client.client_id}: {e}")

    return geocoded, errors


def _build_location_list(
    clients: list[Client],
    geocoded: dict[str, tuple[float, float]],
    home_latlon: tuple[float, float],
) -> list[tuple[float, float]]:
    """Build [home_base, c0, c1, …] as (lon, lat) for ORS matrix API."""
    home_lon, home_lat = home_latlon[1], home_latlon[0]
    locs: list[tuple[float, float]] = [(home_lon, home_lat)]
    for c in clients:
        if c.client_id in geocoded:
            lat, lon = geocoded[c.client_id]
            locs.append((lon, lat))
        else:
            locs.append((home_lon, home_lat))
    return locs


def _render_variant_tab(
    result: ScheduleResult,
    geocoded: dict[str, tuple[float, float]],
    home_latlon: tuple[float, float],
) -> None:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total Drive Time", f"{result.total_drive_min:.0f} min")
    with col_b:
        sessions_count = sum(len(d.sessions) for d in result.days)
        st.metric("Sessions Scheduled", sessions_count)
    with col_c:
        st.metric("Unscheduled Clients", len(result.unscheduled))

    if result.unscheduled:
        st.error(
            "Could not schedule: " + ", ".join(result.unscheduled)
        )

    gap_violations = sum(d.gap_violations for d in result.days)
    if gap_violations:
        st.warning(f"{gap_violations} gap(s) exceed the 20-minute soft buffer.")

    st.subheader("Weekly Grid")
    grid_df = schedule_to_weekly_grid(result)
    if not grid_df.empty:
        st.dataframe(grid_df, use_container_width=True, hide_index=True)
    else:
        st.info("No sessions scheduled.")

    st.subheader("Day Summary")
    day_rows = [
        {
            "Day": dr.day,
            "Sessions": len(dr.sessions),
            "Drive Time (min)": f"{dr.total_drive_min:.0f}",
            "Gap Violations": dr.gap_violations,
        }
        for dr in result.days
    ]
    st.dataframe(pd.DataFrame(day_rows), use_container_width=True, hide_index=True)

    st.subheader("Route Map")
    fmap = build_map(result, geocoded, home_latlon)
    st_folium(fmap, width="100%", height=480, returned_objects=[])

    csv_bytes = schedule_to_csv_bytes(result)
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name=f"schedule_{result.variant.value}.csv",
        mime="text/csv",
    )


def _render_results(
    results: dict[str, ScheduleResult],
    geocoded: dict[str, tuple[float, float]],
    home_latlon: tuple[float, float],
) -> None:
    st.divider()
    st.header("4. Results")
    tab_labels = ["🥗 No Lunch", "🍱 Daily Lunch", "🔀 Hybrid"]
    variant_keys = [v.value for v in LunchVariant]
    tabs = st.tabs(tab_labels)
    for tab, vkey in zip(tabs, variant_keys):
        result = results.get(vkey)
        if result is None:
            continue
        with tab:
            _render_variant_tab(result, geocoded, home_latlon)


# ── Page layout ───────────────────────────────────────────────────────────────

_init_state()

with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input(
        "ORS API Key",
        value=os.getenv("ORS_API_KEY", ""),
        type="password",
        help="Get a free key at openrouteservice.org",
    )
    st.caption(
        "Your key is used only for geocoding and travel-time requests and is "
        "never stored beyond this session."
    )
    st.divider()
    st.markdown("**Home Base**")
    st.code(HOME_BASE_ADDRESS, language=None)

st.title("🗓️ First Steps SLP Schedule Optimizer")
st.caption(
    "Upload your weekly client list to generate an optimized schedule that "
    "minimises drive time while respecting each family's availability."
)

# ── Step 1: Upload ────────────────────────────────────────────────────────────

st.header("1. Upload Client List")

col_up, col_sample = st.columns([3, 1])
with col_up:
    uploaded_file = st.file_uploader(
        "Upload CSV (client_id, intersection, availability, notes)",
        type=["csv"],
        help="See the README for column format and availability syntax examples.",
    )
with col_sample:
    st.markdown("&nbsp;")
    if st.button("📋 Use Sample Data", use_container_width=True):
        if SAMPLE_CSV_PATH.exists():
            st.session_state["clients_df"] = pd.read_csv(SAMPLE_CSV_PATH)
        else:
            st.error("sample_clients.csv not found in data/")

if uploaded_file is not None:
    st.session_state["clients_df"] = pd.read_csv(uploaded_file)

clients_df: Optional[pd.DataFrame] = st.session_state["clients_df"]

if clients_df is not None:
    required_cols = {"client_id", "intersection", "availability"}
    missing_cols = required_cols - set(clients_df.columns)
    if missing_cols:
        st.error(f"CSV is missing required columns: {', '.join(sorted(missing_cols))}")
        st.stop()

    parsed_clients: list[Client] = _parse_clients(clients_df)
    st.session_state["parsed_clients"] = parsed_clients

    # Show parsed client table
    display_rows = []
    for c, (_, row) in zip(parsed_clients, clients_df.iterrows()):
        display_rows.append(
            {
                "Client ID": c.client_id,
                "Address": row.get("intersection", ""),
                "Availability (raw)": row.get("availability", ""),
                "Parsed Windows": c.availability.summary(),
                "✓": "✅" if c.availability.is_valid else "❌",
                "Notes": c.notes,
            }
        )
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

    invalid_count = sum(1 for c in parsed_clients if not c.availability.is_valid)
    if invalid_count:
        st.warning(
            f"{invalid_count} client(s) have unparseable availability strings. "
            "They will appear as unschedulable in the results."
        )

    # ── Step 2: Geocoding ────────────────────────────────────────────────────

    st.divider()
    st.header("2. Verify Geocoding")

    if st.button("📍 Geocode Addresses", type="primary"):
        if _require_api_key(api_key):
            with st.spinner("Geocoding addresses via OpenRouteService…"):
                geocoded, geo_errors = _geocode_all(
                    api_key, parsed_clients, clients_df
                )
            st.session_state["geocoded"] = geocoded
            if geo_errors:
                st.warning("Geocoding issues:")
                for e in geo_errors:
                    st.markdown(f"- {e}")
            if geocoded:
                st.success(
                    f"Geocoded {len(geocoded)} / {len(parsed_clients)} addresses."
                )

    geocoded: Optional[dict] = st.session_state.get("geocoded")
    if geocoded:
        preview_map = build_geocoding_preview_map(
            geocoded, st.session_state["home_latlon"]
        )
        st_folium(preview_map, width="100%", height=400, returned_objects=[])

        # ── Step 3: Generate ─────────────────────────────────────────────────

        st.divider()
        st.header("3. Generate Schedule")

        if st.button("🚀 Generate All Three Variants", type="primary"):
            if _require_api_key(api_key):
                with st.spinner("Building travel-time matrix…"):
                    locs = _build_location_list(
                        parsed_clients,
                        geocoded,
                        st.session_state["home_latlon"],
                    )
                    try:
                        matrix = build_matrix(api_key=api_key, locations=locs)
                        st.session_state["travel_matrix"] = matrix
                    except MatrixError as e:
                        st.error(f"Matrix API error: {e}")
                        st.stop()

                new_results: dict[str, ScheduleResult] = {}
                for variant in LunchVariant:
                    label = variant.value.replace("_", " ").title()
                    with st.spinner(f"Optimizing — {label}…"):
                        res = solve(
                            clients=parsed_clients,
                            travel_matrix=st.session_state["travel_matrix"],
                            variant=variant,
                        )
                        new_results[variant.value] = res
                st.session_state["results"] = new_results
                st.success("All three variants generated!")

        results: dict = st.session_state.get("results", {})
        if results:
            _render_results(results, geocoded, st.session_state["home_latlon"])
