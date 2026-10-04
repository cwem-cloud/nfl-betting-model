"""NFL blind handicap: fair spread / total / win% for each game, built without any line."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pocket_capper.data import nflverse
from pocket_capper.models import distributions as dist
from pocket_capper.models.team_ratings import Ratings, fit_ratings

OUT_STATUSES = {"Out", "Doubtful", "IR", "Injured Reserve", "PUP", "Suspended"}


def build_games(seasons: list[int]) -> pd.DataFrame:
    """Completed + upcoming games with EPA-implied points attached."""
    sched = nflverse.schedules()
    g = sched[sched["season"].isin(seasons) & sched["game_type"].isin(["REG", "WC", "DIV", "CON", "SB"])].copy()
    g = g.rename(columns={"home_team": "home", "away_team": "away", "home_score": "home_pts", "away_score": "away_pts"})
    g["neutral"] = g["location"].eq("Neutral")
    # playoffs continue the week count
    epa = nflverse.team_game_epa([s for s in seasons])
    if not epa.empty:
        done = g.dropna(subset=["home_pts"])
        m = epa.merge(
            pd.concat(
                [
                    done[["game_id", "home", "home_pts"]].rename(columns={"home": "posteam", "home_pts": "pts"}),
                    done[["game_id", "away", "away_pts"]].rename(columns={"away": "posteam", "away_pts": "pts"}),
                ]
            ),
            on=["game_id", "posteam"],
        )
        # Map offensive EPA (non-garbage) onto the points scale by regression on history.
        if len(m) > 50:
            b, a = np.polyfit(m["off_epa"], m["pts"], 1)
            epa["eff_pts"] = a + b * epa["off_epa"]
            eff = epa.set_index(["game_id", "posteam"])["eff_pts"]
            g["home_eff_pts"] = [eff.get((gid, t), np.nan) for gid, t in zip(g["game_id"], g["home"])]
            g["away_eff_pts"] = [eff.get((gid, t), np.nan) for gid, t in zip(g["game_id"], g["away"])]
    return g


def fit(games: pd.DataFrame, season: int, week: int, cfg: dict) -> Ratings:
    return fit_ratings(
        games,
        as_of_season=season,
        as_of_week=week,
        half_life_weeks=cfg["decay_half_life_weeks"],
        prior_season_weight=cfg["prior_season_weight"],
        alpha=cfg["ridge_alpha"],
        eff_blend=cfg["epa_blend"],
    )


def qb_status(season: int, week: int) -> dict[str, dict]:
    """Starting-QB availability per team from the official injury report."""
    inj = nflverse.injuries(season)
    out = {}
    if inj.empty:
        return out
    wk = inj[(inj["week"] == week) & (inj["position"] == "QB")]
    for _, r in wk.iterrows():
        status = str(r.get("report_status") or "")
        out.setdefault(r["team"], []).append({"name": r["full_name"], "status": status, "id": r["gsis_id"]})
    return out


def team_qb_value(player_stats: pd.DataFrame, season: int) -> dict[str, dict]:
    """Primary QB (most dropbacks this season) and his EPA/play edge over a typical backup."""
    ps = player_stats[(player_stats["season"] == season) & (player_stats["position"] == "QB")]
    res = {}
    if ps.empty:
        return res
    agg = ps.groupby(["team", "player_id", "player_display_name"]).agg(
        att=("attempts", "sum"), epa=("passing_epa", "sum"), rush_epa=("rushing_epa", "sum")
    ).reset_index()
    agg["epa_pp"] = (agg["epa"] + agg["rush_epa"].fillna(0)) / agg["att"].clip(lower=1)
    for team, grp in agg.groupby("team"):
        top = grp.sort_values("att", ascending=False).iloc[0]
        res[team] = {"id": top["player_id"], "name": top["player_display_name"], "epa_pp": float(top["epa_pp"])}
    return res


def key_skill_players(player_stats: pd.DataFrame, season: int, week: int) -> dict[str, list[dict]]:
    """Each team's featured skill players this season: WR1 (>=24% targets), TE (>=20%), workhorse RB (>=55% carries)."""
    need = {"season", "week", "season_type", "player_id", "team", "position", "target_share", "carries"}
    if player_stats.empty or not need <= set(player_stats.columns):
        return {}
    ps = player_stats[(player_stats["season"] == season) & (player_stats["week"] < week)
                      & (player_stats["season_type"] == "REG") & player_stats["player_id"].notna()]
    out: dict[str, list[dict]] = {}
    for team, grp in ps.groupby("team"):
        if grp["week"].nunique() < 3:
            continue
        for pos, thr in (("WR", 0.24), ("TE", 0.20)):
            p = grp[grp["position"] == pos].groupby(["player_id", "player_display_name"]).agg(
                share=("target_share", "mean"), games=("week", "nunique")).reset_index()
            p = p[(p["games"] >= 3) & (p["share"] >= thr)].sort_values("share", ascending=False).head(1)
            for r in p.itertuples():
                out.setdefault(team, []).append({"id": r.player_id, "name": r.player_display_name, "role": f"{pos}1"})
        rb = grp[grp["position"] == "RB"].groupby(["player_id", "player_display_name"])["carries"].sum()
        if not rb.empty and rb.max() / max(grp["carries"].sum(), 1) >= 0.55:
            pid, name = rb.idxmax()
            out.setdefault(team, []).append({"id": pid, "name": name, "role": "lead RB"})
    return out


def out_player_ids(season: int, week: int) -> set[str]:
    inj = nflverse.injuries(season)
    if inj.empty:
        return set()
    wk = inj[inj["week"] == week]
    return set(wk[wk["report_status"].isin(OUT_STATUSES)]["gsis_id"])


def project_games(
    upcoming: pd.DataFrame,
    ratings: Ratings,
    cfg: dict,
    qb_flags: dict | None = None,
    qb_values: dict | None = None,
    weather: dict | None = None,
    key_players: dict | None = None,
    out_ids: set | None = None,
) -> pd.DataFrame:
    """Blind projections. `upcoming` must contain game_id, home, away, neutral, home_rest, away_rest, roof."""
    rows = []
    for _, g in upcoming.iterrows():
        hp, ap = ratings.project(g["home"], g["away"], bool(g.get("neutral", False)))
        if not bool(g.get("neutral", False)):
            # EPA-blended ratings under-credit home field (special teams/penalties); 2020-25 residual ~+0.6
            hp += cfg["home_edge_adjust"] / 2
            ap -= cfg["home_edge_adjust"] / 2
        notes = []
        # Featured skill players ruled out (2020-25: WR1/TE1/lead RB absences cost ~1.5-1.8 pts vs model)
        for side, team in (("home", g["home"]), ("away", g["away"])):
            missing = [k for k in (key_players or {}).get(team, []) if k["id"] in (out_ids or set())]
            if missing:
                hit = min(cfg["skill_out_cap"], cfg["skill_out_points"] * len(missing))
                if side == "home":
                    hp -= hit
                else:
                    ap -= hit
                notes.append(f"{team} without " + ", ".join(f"{k['role']} {k['name']}" for k in missing) + f" (-{hit:.1f} pts)")
        # Rest / bye edge
        rest_diff = float(np.nan_to_num(g.get("home_rest", 7)) - np.nan_to_num(g.get("away_rest", 7)))
        rest_diff = float(np.clip(rest_diff, -7, 7))
        if abs(rest_diff) >= 3:
            adj = rest_diff * cfg["rest_points_per_day"]
            hp += adj / 2
            ap -= adj / 2
            notes.append(f"Rest edge {'home' if rest_diff > 0 else 'away'} ({abs(rest_diff):.0f} days, {abs(adj):.1f} pts)")
        # Starting QB out
        for side, team in (("home", g["home"]), ("away", g["away"])):
            flags = (qb_flags or {}).get(team, [])
            starter = (qb_values or {}).get(team)
            if starter and any(f["id"] == starter["id"] and f["status"] in OUT_STATUSES for f in flags):
                # fit on 561 primary-QB-out games 2020-25: lost pts ~= 2.2 + 9.8 * starter EPA/play
                hit = float(np.clip(cfg["qb_out_base"] + cfg["qb_out_per_epa"] * starter["epa_pp"], 0, cfg["qb_out_cap"]))
                if side == "home":
                    hp -= hit
                else:
                    ap -= hit
                notes.append(f"{team} QB {starter['name']} OUT (-{hit:.1f} pts)")
        # Roof + weather (coefficients fit on 2020-25 blind-model total residuals)
        roof = str(g.get("roof", "outdoors"))
        if roof in ("dome", "closed"):
            hp += cfg["dome_total"] / 2
            ap += cfg["dome_total"] / 2
        wx = (weather or {}).get(g["game_id"])
        if wx and roof in ("outdoors", "open"):
            tot_adj = 0.0
            over = wx.get("wind_mph", 0) - cfg["wind_threshold_mph"]
            if over > 0:
                tot_adj += max(cfg["wind_total_cap"], cfg["wind_total_per_mph"] * over)
            if wx.get("temp_f", 60) < 25:
                tot_adj += cfg["cold_total_below_25f"]
            if wx.get("precip_prob", 0) >= 60:
                tot_adj += cfg["precip_total"]
            if tot_adj:
                hp += tot_adj / 2
                ap += tot_adj / 2
                notes.append(
                    f"Weather: {wx.get('wind_mph', 0):.0f} mph wind, {wx.get('temp_f', 0):.0f}F, "
                    f"{wx.get('precip_prob', 0):.0f}% precip ({tot_adj:+.1f} total)"
                )
        margin = hp - ap
        total = hp + ap
        rows.append(
            {
                "game_id": g["game_id"],
                "home": g["home"],
                "away": g["away"],
                "home_pts": round(hp, 1),
                "away_pts": round(ap, 1),
                "fair_margin": margin,
                "fair_spread_home": dist.margin_to_spread(margin),
                "fair_total": round(total * 2) / 2,
                "home_win_prob": dist.win_prob(margin, cfg["margin_sigma"], "nfl"),
                "adjust_notes": notes,
            }
        )
    return pd.DataFrame(rows)
