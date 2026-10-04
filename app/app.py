"""Pocket Capper dashboard.  streamlit run app/app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pocket_capper import db
from pocket_capper.market.odds_math import fmt_american
from pocket_capper.settings import config, secret

st.set_page_config(page_title="Pocket Capper", page_icon="🏈", layout="wide")

# Dark-surface chart tokens (validated reference palette, dark steps)
SURFACE, TEXT, TEXT2, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#383835"
SERIES = ["#3987e5", "#d95926", "#199e70"]  # DK, sharp, market median
GOOD, BAD = "#53d769", "#e66767"

st.markdown(
    """
<style>
.pc-card {border:1px solid #383835;border-radius:10px;padding:12px 16px;margin-bottom:10px;background:#1f1f1d}
.pc-sel {font-size:1.15rem;font-weight:700}
.pc-u {display:inline-block;padding:2px 10px;border-radius:999px;background:#53d769;color:#0b0b0b;font-weight:800;margin-left:8px}
.pc-meta {color:#c3c2b7;font-size:.85rem}
.pc-chip {display:inline-block;border:1px solid #555;border-radius:6px;padding:1px 6px;margin:2px 4px 0 0;font-size:.75rem;color:#c3c2b7}
.pc-chip.pos {border-color:#53d769} .pc-chip.neg {border-color:#e66767}
.pc-warn {color:#e66767;font-size:.85rem;font-weight:600;margin-top:4px}
</style>
""",
    unsafe_allow_html=True,
)

cfg = config()


def chart_layout(fig: go.Figure, title: str, ytitle: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=15)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=TEXT2),
        margin=dict(l=40, r=20, t=50, b=40), hovermode="x unified", height=340,
        legend=dict(orientation="h", y=-0.2),
        yaxis=dict(title=ytitle, gridcolor=GRID, zerolinecolor=GRID),
        xaxis=dict(gridcolor=GRID),
    )
    return fig


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🏈 Pocket Capper")
    st.caption("Blind handicap first → then the market.")
    sports = st.multiselect("Sports", ["nfl", "cfb"], default=["nfl", "cfb"], format_func=str.upper)
    include_props = st.checkbox("Player props + TD Zone (NFL)", value=True)
    bankroll = st.number_input("Bankroll ($)", min_value=0, value=1000, step=100,
                               help="Only used to show $ per play. 1U = unit_pct_bankroll% in config.yaml.")
    UNIT_USD = bankroll * cfg["units"]["unit_pct_bankroll"] / 100
    if st.button("▶ Run", type="primary", width="stretch"):
        from pocket_capper.engine import grading, slate

        status = st.status("Running Pocket Capper…", expanded=True)
        with status:
            grading.grade_all(log=st.write)
            res = slate.run(tuple(sports), include_props=include_props, progress=st.write)
        status.update(label=f"Done: {res['picks']} plays" + (f" ({len(res['errors'])} errors)" if res["errors"] else ""),
                      state="error" if res["errors"] else "complete")
    if st.button("Grade finished picks", width="stretch"):
        from pocket_capper.engine import grading

        st.write(grading.grade_all(log=st.write))

    st.divider()
    st.caption("Data connections")
    for name, label in (("ODDS_API_KEY", "The Odds API (lines/props)"), ("CFBD_API_KEY", "CollegeFootballData"),
                        ("ANTHROPIC_API_KEY", "Claude rationales (optional)"), ("DATABASE_URL", "Hosted DB (optional)")):
        st.write(("✅ " if secret(name) else "⚪ ") + label)
    runs = db.read(db.runs, "ORDER BY started_at DESC LIMIT 1")
    if not runs.empty:
        st.caption(f"Last run: {runs['started_at'].iloc[0]} UTC ({runs['status'].iloc[0]})")
        with st.expander("Last run log"):
            st.code(runs["log"].iloc[0] or "")

tabs = st.tabs(["🏈 NFL", "🎓 CFB", "🎯 TD Zone", "📊 Prop Hub", "📈 Line Market", "🧠 Sharp Inputs", "📚 History"])


def pick_card(p: pd.Series) -> None:
    sigs = json.loads(p["signals"]) if isinstance(p["signals"], str) and p["signals"] else []
    chips = "".join(
        f"<span class='pc-chip {'pos' if s['direction'] > 0 else 'neg'}'>{'▲' if s['direction'] > 0 else '▼'} {s['name']}</span>"
        for s in sigs
    )
    k = pd.to_datetime(p["kickoff"], utc=True, errors="coerce")
    kick = k.tz_convert("America/New_York").strftime("%a %b %d %I:%M %p ET") if pd.notna(k) else ""
    moved = ""
    if pd.notna(p.get("latest_line")) and pd.notna(p.get("line")) and p["latest_line"] != p["line"]:
        moved = f" · now {p['latest_line']:g}"
    elif pd.notna(p.get("latest_price")) and p["latest_price"] != p["price"]:
        moved = f" · now {fmt_american(p['latest_price'])}"
    health = ""
    if pd.notna(p.get("latest_ev")):
        if p["latest_ev"] < 0:
            health = f"<div class='pc-warn'>⚠ Edge gone at the current number ({p['latest_ev']:+.1%} EV now). Don't add.</div>"
        elif p["latest_ev"] < p["ev"] / 2:
            health = f"<div class='pc-meta'>↘ Edge shrinking: {p['latest_ev']:+.1%} EV at the current number</div>"
    usd = f" · &#36;{p['units'] * UNIT_USD:,.0f}" if UNIT_USD else ""
    st.markdown(
        f"""<div class='pc-card'>
<span class='pc-sel'>{p['selection']} {fmt_american(p['price'])}</span><span class='pc-u'>{p['units']:g}U</span>
<div class='pc-meta'>{p['matchup']} · {kick}
 · DK · EV {p['ev']:+.1%} · win {p['final_prob']:.1%}{usd}{moved}</div>{health}{chips}</div>""",
        unsafe_allow_html=True,
    )
    with st.expander("Rationale"):
        st.markdown(p["rationale"] or "")


def pending_picks(sport: str, categories: tuple[str, ...], upcoming_only: bool = True) -> pd.DataFrame:
    df = db.read(db.picks, "WHERE sport = :s AND status = 'pending'", {"s": sport})
    if df.empty:
        return df
    df = df[df["category"].isin(categories)]
    if upcoming_only:  # kicked-off picks wait in History until graded
        df = df[pd.to_datetime(df["kickoff"], utc=True, errors="coerce") > pd.Timestamp.now(tz="UTC")]
    return df.sort_values(["units", "ev"], ascending=False)


def card_text(df: pd.DataFrame) -> str:
    lines = []
    for _, p in df.iterrows():
        usd = f" (${p['units'] * UNIT_USD:,.0f})" if UNIT_USD else ""
        lines.append(f"{p['units']:g}U{usd}  {p['selection']} {fmt_american(p['price'])}  [{p['matchup']}]")
    return "\n".join(lines)


def sport_tab(sport: str) -> None:
    picks = pending_picks(sport, ("side", "total", "ml"))
    board, as_of = db.latest_board(sport, "games")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Plays on the board", len(picks))
    c2.metric("Units in play", f"{picks['units'].sum():g}U" if not picks.empty else "0U")
    c3.metric("Avg EV", f"{picks['ev'].mean():+.1%}" if not picks.empty else "—")
    c4.metric("Games handicapped", len(board))
    if as_of:
        st.caption(f"Board as of {as_of} UTC. Plays are locked at the line first issued; later runs update 'now'.")
    left, right = st.columns([1, 1.4])
    with left:
        st.subheader("Plays")
        if picks.empty:
            st.info("No edges right now. That's a valid result - no forced plays.")
        else:
            with st.expander("📋 Copy card"):
                extra = pending_picks(sport, ("prop", "attd")) if sport == "nfl" else pd.DataFrame()
                st.code(card_text(pd.concat([picks, extra])), language=None)
        for _, p in picks.iterrows():
            pick_card(p)
    with right:
        st.subheader("Full slate - blind vs market")
        if board.empty:
            st.info("Hit ▶ Run to build the board.")
            return
        show = pd.DataFrame({
            "Matchup": board["away"] + " @ " + board["home"],
            "Proj": board["proj_away_pts"].round(1).astype(str) + "-" + board["proj_home_pts"].round(1).astype(str),
            "Blind spread (home)": board["fair_spread_home"],
            "DK spread": board.get("dk_spread_home"),
            "Sharp": board.get("sharp_spread_home"),
            "Blind total": board["fair_total"],
            "DK total": board.get("dk_total"),
            "Home win%": (board["home_win_prob"] * 100).round(0),
            "Result": board["best_play"],
        })
        st.dataframe(show, hide_index=True, width="stretch")
        notes = board[board["notes"].fillna("") != ""] if "notes" in board else pd.DataFrame()
        flags = board[board["flag"].notna()] if "flag" in board else pd.DataFrame()
        if not notes.empty or not flags.empty:
            with st.expander("Adjustments & flags"):
                for _, r in notes.iterrows():
                    st.write(f"**{r['away']} @ {r['home']}**: {r['notes']}")
                for _, r in flags.iterrows():
                    st.write(f"⚠️ **{r['away']} @ {r['home']}**: {r['flag']}")
        ratings, _ = db.latest_board(sport, "ratings")
        if not ratings.empty:
            with st.expander("Power ratings (points vs average, blind)"):
                st.dataframe(ratings.round(2), hide_index=True, width="stretch")


with tabs[0]:
    sport_tab("nfl")
with tabs[1]:
    sport_tab("cfb")

# ------------------------------------------------------------------ TD Zone
with tabs[2]:
    st.subheader("🎯 TD Zone - anytime TD, +EV only")
    attd, as_of = db.latest_board("nfl", "attd")
    if attd.empty:
        st.info("Run with props enabled (needs ODDS_API_KEY) to populate the TD Zone.")
    else:
        st.caption(f"As of {as_of} UTC. Model % comes from red-zone usage x blind team TD expectation, calibrated on 2025.")
        only = st.toggle("Plays only", value=False, key="attd_only")
        d = attd[attd["qualifies"]] if only else attd[attd["ev"] > 0]
        d = d.sort_values("ev", ascending=False)
        st.dataframe(pd.DataFrame({
            "Play": d["qualifies"].map({True: "✅", False: ""}),
            "Player": d["player"], "Pos": d["position"], "Team": d["team"], "Opp": d["opp"],
            "DK": d["price"].map(fmt_american), "Model %": (d["model_prob"] * 100).round(1),
            "Market %": (d["market_prob"] * 100).round(1), "Fair odds": d["fair_odds"].map(fmt_american),
            "EV": (d["ev"] * 100).round(1).astype(str) + "%", "U": d["units"],
        }), hide_index=True, width="stretch")

# ------------------------------------------------------------------ Prop Hub
with tabs[3]:
    st.subheader("📊 NFL Prop Hub")
    props, as_of = db.latest_board("nfl", "props")
    if props.empty:
        st.info("Run with props enabled (needs ODDS_API_KEY) to populate the Prop Hub.")
    else:
        from pocket_capper.models.nfl_props import STAT_LABEL

        c1, c2, c3 = st.columns(3)
        stat = c1.multiselect("Market", sorted(props["stat"].unique()), format_func=lambda s: STAT_LABEL.get(s, s))
        team = c2.multiselect("Team", sorted(props["team"].unique()))
        only = c3.toggle("Plays only", value=True, key="prop_only")
        d = props.copy()
        if stat:
            d = d[d["stat"].isin(stat)]
        if team:
            d = d[d["team"].isin(team)]
        if only:
            d = d[d["qualifies"]]
        d = d.sort_values("ev", ascending=False)
        st.dataframe(pd.DataFrame({
            "Play": d["qualifies"].map({True: "✅", False: ""}),
            "Player": d["player"], "Team": d["team"], "Opp": d["opp"], "Market": d["stat"].map(STAT_LABEL),
            "Side": d["side"].str.title(), "DK line": d["line"], "DK": d["price"].map(fmt_american),
            "Projection": d["projection"].round(1), "Model %": (d["model_prob"] * 100).round(1),
            "Market %": (d["market_prob"] * 100).round(1), "EV": (d["ev"] * 100).round(1).astype(str) + "%",
            "U": d["units"],
        }), hide_index=True, width="stretch")
        st.caption(f"As of {as_of} UTC. Projections are blind (built from usage, efficiency, opponent and the blind game script).")

# ------------------------------------------------------------------ Line Market
with tabs[4]:
    st.subheader("📈 Line Market")
    sport = st.radio("Sport", ["nfl", "cfb"], horizontal=True, format_func=str.upper, key="lm_sport")
    board, _ = db.latest_board(sport, "games")
    if board.empty or "event_id" not in board or board["event_id"].isna().all():
        st.info("No market data yet - add ODDS_API_KEY and Run.")
    else:
        b = board[board["event_id"].notna()].copy()
        b["label"] = b["away"] + " @ " + b["home"]
        for c in ("dk_open_spread_home", "dk_spread_home", "sharp_spread_home", "spread_bets_away", "spread_bets_home",
                  "spread_money_away", "spread_money_home", "dk_open_total", "dk_total", "total_bets_over", "total_money_over"):
            if c not in b:
                b[c] = None

        def pct(c):
            return b[c].map(lambda v: "—" if pd.isna(v) else f"{v:.0f}%")

        overview = pd.DataFrame({
            "Matchup": b["label"],
            "DK open": b.get("dk_open_spread_home"), "DK now": b.get("dk_spread_home"), "Sharp": b.get("sharp_spread_home"),
            "Blind": b["fair_spread_home"],
            "Bets% away/home": pct("spread_bets_away") + " / " + pct("spread_bets_home"),
            "Money% away/home": pct("spread_money_away") + " / " + pct("spread_money_home"),
            "Total open": b["dk_open_total"], "Total now": b["dk_total"],
            "Over bets% / money%": pct("total_bets_over") + " / " + pct("total_money_over"),
        })
        st.dataframe(overview, hide_index=True, width="stretch")

        game = st.selectbox("Game detail", b["label"].tolist())
        g = b[b["label"] == game].iloc[0]
        snap = db.read(db.odds_snapshots, "WHERE event_id = :e AND player IS NULL", {"e": g["event_id"]})
        if not snap.empty:
            snap["fetched_at"] = pd.to_datetime(snap["fetched_at"], utc=True)
            latest = snap.sort_values("fetched_at").groupby(["book", "market", "side"]).last().reset_index()

            def cell(r):
                if pd.isna(r["point"]):
                    return fmt_american(r["price"])
                pt = f"{r['point']:+g}" if r["market"] == "spread" else f"{r['point']:g}"
                return f"{pt} ({fmt_american(r['price'])})"

            latest["v"] = latest.apply(cell, axis=1)
            latest["col"] = latest["market"] + " " + latest["side"]
            grid = latest.pivot(index="book", columns="col", values="v")
            order = [c for c in ["spread away", "spread home", "total over", "total under", "ml away", "ml home"] if c in grid]
            grid = grid[order]
            primary = cfg["books"]["primary"]
            grid = grid.reindex([primary] + [x for x in grid.index if x != primary])
            st.markdown("**All books (DK first)**")
            st.dataframe(grid, width="stretch")

            mkt = st.radio("Movement", ["spread", "total"], horizontal=True, key="mv_mkt")
            side = "home" if mkt == "spread" else "over"
            s = snap[(snap["market"] == mkt) & (snap["side"] == side)]
            fig = go.Figure()
            dk = s[s["book"] == primary].sort_values("fetched_at")
            sharp = s[s["book"].isin(cfg["books"]["sharp"])].groupby("fetched_at")["point"].median()
            med = s.groupby("fetched_at")["point"].median()
            for name, x, y, color in (("DraftKings", dk["fetched_at"], dk["point"], SERIES[0]),
                                      ("Sharp books", sharp.index, sharp.values, SERIES[1]),
                                      ("All-book median", med.index, med.values, SERIES[2])):
                if len(x):
                    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines+markers", line=dict(color=color, width=2),
                                             marker=dict(size=8, line=dict(color=SURFACE, width=2)), line_shape="hv"))
            label = f"{g['home']} spread" if mkt == "spread" else "Total"
            st.plotly_chart(chart_layout(fig, f"{game}: {label} over time", label), width="stretch")
            st.caption("History builds from every Run and scheduled snapshot - more snapshots, finer movement.")

    with st.expander("Import betting splits (Action / VSIN / any source)"):
        st.caption("CSV columns: sport, away, home, market (spread|total|ml), side (home|away|over|under), bets_pct, money_pct")
        up = st.file_uploader("Splits CSV", type=["csv"])
        src = st.text_input("Source name", value="Action Network")
        if up is not None and st.button("Import splits"):
            from pocket_capper.data.splits import parse_manual_csv

            try:
                df = parse_manual_csv(up.getvalue().decode(), source=src)
                db.insert_df(db.splits, df)
                st.success(f"Imported {len(df)} rows. They'll be used on the next Run.")
            except Exception as e:
                st.error(str(e))

# ------------------------------------------------------------------ Sharp inputs
with tabs[5]:
    st.subheader("🧠 Sharp Inputs")
    st.caption("Log plays from sharp sources you follow. They become signals on matching games and get their own graded record.")
    sport = st.radio("Sport", ["nfl", "cfb"], horizontal=True, format_func=str.upper, key="sh_sport")
    board, _ = db.latest_board(sport, "games")
    if board.empty or "event_id" not in board:
        st.info("Run first so there's a slate to attach plays to.")
    else:
        b = board[board["event_id"].notna()].copy()
        b["label"] = b["away"] + " @ " + b["home"]
        with st.form("sharp"):
            c1, c2, c3 = st.columns(3)
            source = c1.text_input("Source", value="Bambino")
            game = c2.selectbox("Game", b["label"].tolist())
            market = c3.selectbox("Market", ["spread", "total", "ml"])
            g = b[b["label"] == game].iloc[0] if len(b) else None
            sides = {"spread": [g["away"], g["home"]], "ml": [g["away"], g["home"]], "total": ["Over", "Under"]}[market] if g is not None else []
            c4, c5, c6, c7 = st.columns(4)
            sel = c4.selectbox("Side", sides)
            line = c5.number_input("Line (side's number)", value=0.0, step=0.5)
            price = c6.number_input("Price", value=-110, step=5)
            units = c7.number_input("Units", value=1.0, step=0.5)
            notes = st.text_input("Notes")
            if st.form_submit_button("Add sharp play") and g is not None:
                side = {g["away"]: "away", g["home"]: "home", "Over": "over", "Under": "under"}[sel]
                db.insert_df(db.sharp_plays, pd.DataFrame([{
                    "created_at": db.now(), "source": source, "sport": sport, "event_id": g["event_id"],
                    "game_id": str(g["game_id"]), "matchup": game, "market": market, "selection": side,
                    "line": None if market == "ml" else line, "price": price, "units": units, "notes": notes,
                    "status": "pending",
                }]))
                st.success("Added. Re-run to fold it into the signals.")
    sp = db.read(db.sharp_plays)
    if not sp.empty:
        rec = sp[sp["status"].isin(["win", "loss", "push"])].groupby("source").agg(
            W=("status", lambda s: (s == "win").sum()), L=("status", lambda s: (s == "loss").sum()),
            P=("status", lambda s: (s == "push").sum()), Units=("profit_units", "sum"),
        )
        if not rec.empty:
            st.markdown("**Source records**")
            st.dataframe(rec.round(2), width="stretch")
        st.dataframe(sp.sort_values("created_at", ascending=False)[
            ["created_at", "source", "matchup", "market", "selection", "line", "price", "units", "status", "profit_units"]],
            hide_index=True, width="stretch")

# ------------------------------------------------------------------ History
with tabs[6]:
    st.subheader("📚 History")
    h = db.read(db.picks)
    if h.empty:
        st.info("No picks yet.")
    else:
        h["kickoff_dt"] = pd.to_datetime(h["kickoff"], utc=True, errors="coerce")
        c1, c2, c3 = st.columns(3)
        sp_f = c1.multiselect("Sport", sorted(h["sport"].unique()), default=sorted(h["sport"].unique()), format_func=str.upper)
        cat_f = c2.multiselect("Type", sorted(h["category"].unique()), default=sorted(h["category"].unique()))
        st_f = c3.multiselect("Status", ["pending", "win", "loss", "push", "void"], default=["win", "loss", "push", "pending"])
        d = h[h["sport"].isin(sp_f) & h["category"].isin(cat_f) & h["status"].isin(st_f)]
        graded = d[d["status"].isin(["win", "loss", "push"])]
        w, l, pu = (graded["status"] == "win").sum(), (graded["status"] == "loss").sum(), (graded["status"] == "push").sum()
        risked = graded["units"].sum()
        pl = graded["profit_units"].sum()
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Record", f"{w}-{l}-{pu}")
        k2.metric("Units", f"{pl:+.2f}U")
        k3.metric("ROI", f"{pl / risked:+.1%}" if risked else "—")
        clv = d["clv_prob"].dropna()
        k4.metric("Avg CLV", f"{clv.mean():+.1%}" if len(clv) else "—", help="Implied-prob gain vs DK closing price. Positive = beating the close.")
        k5.metric("Beat close", f"{(clv > 0).mean():.0%}" if len(clv) else "—")
        if not graded.empty:
            g = graded.sort_values("kickoff_dt")
            g["cum"] = g["profit_units"].cumsum()
            fig = go.Figure(go.Scatter(x=g["kickoff_dt"], y=g["cum"], mode="lines", line=dict(color=SERIES[0], width=2),
                                       name="Cumulative units", hovertemplate="%{y:+.2f}U<extra></extra>"))
            fig.add_hline(y=0, line_color=GRID)
            st.plotly_chart(chart_layout(fig, "Cumulative units", "Units"), width="stretch")
            by = graded.groupby("category").agg(Plays=("status", "size"), Wins=("status", lambda s: (s == "win").sum()),
                                                Units=("profit_units", "sum"), Risked=("units", "sum"))
            by["ROI"] = (by["Units"] / by["Risked"]).map(lambda x: f"{x:+.1%}")
            st.dataframe(by.round(2), width="stretch")
        show = d.sort_values("kickoff_dt", ascending=False)
        st.dataframe(pd.DataFrame({
            "Kickoff": show["kickoff_dt"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %I:%M %p"),
            "Sport": show["sport"].str.upper(), "Type": show["category"], "Matchup": show["matchup"],
            "Play": show["selection"], "Price": show["price"].map(fmt_american), "U": show["units"],
            "EV": (show["ev"] * 100).round(1), "Close": show["close_line"], "CLV%": (show["clv_prob"] * 100).round(1),
            "Status": show["status"].map({"win": "✅ win", "loss": "❌ loss", "push": "➖ push", "pending": "⏳ pending", "void": "void"}),
            "P/L": show["profit_units"].round(2),
        }), hide_index=True, width="stretch")
        pick = st.selectbox("Rationale for", show["selection"] + " - " + show["matchup"])
        if pick:
            r = show[(show["selection"] + " - " + show["matchup"]) == pick].iloc[0]
            st.markdown(r["rationale"] or "")
        st.download_button("Download history CSV", d.drop(columns=["kickoff_dt"]).to_csv(index=False), "pocket_capper_history.csv")
