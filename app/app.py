import streamlit as st
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from nfl_model.feature_engineering import compute_targets, build_features
from nfl_model.modeling import train_ats, train_ml, train_totals
from nfl_model.utils import moneyline_to_prob, edge_to_confidence, pick_text

st.set_page_config(page_title="NFL Model", layout="wide")

@st.cache_resource
def load_config():
    with open("config.yaml","r") as f:
        return yaml.safe_load(f)

def load_data():
    df_games = pd.read_csv("data/raw/games/mock_games_2024.csv")
    df_odds  = pd.read_csv("data/raw/odds/mock_odds_2024.csv")
    return df_games, df_odds

def refresh_pipeline(cfg):
    games, odds = load_data()
    df = compute_targets(games, odds)
    df, feats = build_features(df)

    # Train simple baselines
    ats_model = train_ats(df, feats)
    ml_model  = train_ml(df, feats)
    tot_model = train_totals(df, feats)

    # Generate naive model spreads/totals and probs by comparing to market
    # For demo, use model predicted probabilities as proxy signals.
    X = df[feats]
    df["model_prob_ats_home"] = ats_model.predict_proba(X)[:,1]
    df["model_prob_ml_home"]  = ml_model.predict_proba(X)[:,1]
    df["model_total_points"]  = tot_model.predict(X)

    # Convert model probs to "model spreads" using a simple centered mapping for demo
    df["model_spread_home"] = - (df["model_prob_ats_home"] - 0.5) * 10  # demo mapping
    df["model_total"] = df["model_total_points"]

    # Market probabilities for ML from moneylines
    df["market_prob_home"] = df["ml_home"].apply(moneyline_to_prob)

    picks = []
    for _, r in df.iterrows():
        # ATS pick
        edge_pts = r["model_spread_home"] - r["spread_close_home"]
        conf = edge_to_confidence(edge_pts, ceiling=7.0)
        picks.append({
            "season": r["season"], "week": r["week"],
            "home_team": r["home_team"], "away_team": r["away_team"],
            "bet_type": "ATS",
            "pick_home": (edge_pts < 0),  # negative model spread -> home favored more -> pick home
            "model_spread": r["model_spread_home"],
            "market_spread": r["spread_close_home"],
            "edge_points": edge_pts,
            "confidence": conf
        })

        # ML pick
        edge_prob = r["model_prob_ml_home"] - r["market_prob_home"]
        conf = edge_to_confidence(edge_prob, ceiling=0.07)
        picks.append({
            "season": r["season"], "week": r["week"],
            "home_team": r["home_team"], "away_team": r["away_team"],
            "bet_type": "ML",
            "pick_home": (edge_prob > 0),
            "model_prob": r["model_prob_ml_home"],
            "market_prob": r["market_prob_home"],
            "edge_prob": edge_prob,
            "confidence": conf
        })

        # Totals pick
        edge_pts_total = r["model_total"] - r["total_close"]
        conf = edge_to_confidence(edge_pts_total, ceiling=7.0)
        picks.append({
            "season": r["season"], "week": r["week"],
            "home_team": r["home_team"], "away_team": r["away_team"],
            "bet_type": "TOTAL",
            "pick_over": (edge_pts_total > 0),
            "model_total": r["model_total"],
            "market_total": r["total_close"],
            "edge_points": edge_pts_total,
            "confidence": conf
        })

    picks_df = pd.DataFrame(picks)
    # Rationale text
    def _r(row):
        return pick_text(row)
    picks_df["rationale"] = picks_df.apply(_r, axis=1)
    return df, picks_df

cfg = load_config()

st.title("🏈 NFL Betting Model (Starter)")
st.caption("Baseline demo with mock data. Replace with real feeds and retrain via the Refresh button.")

if st.button("🔄 Refresh (rebuild features → retrain → score)"):
    st.session_state["last_run"] = "running"

if "last_run" not in st.session_state or st.session_state["last_run"] == "running":
    base_df, picks_df = refresh_pipeline(cfg)
    st.session_state["picks_df"] = picks_df
    st.session_state["last_run"] = datetime.now().isoformat()

min_conf = st.slider("Minimum confidence", min_value=1, max_value=10, value=cfg["app"]["default_min_confidence"])
bet_types = st.multiselect("Bet types", ["ATS","ML","TOTAL"], default=["ATS","ML","TOTAL"])

if "picks_df" in st.session_state:
    view = st.session_state["picks_df"].copy()
    view = view[view["bet_type"].isin(bet_types)]
    view = view[view["confidence"] >= min_conf]
    st.dataframe(view.sort_values(["week","bet_type","confidence"], ascending=[True, True, False]), use_container_width=True)
else:
    st.info("Press Refresh to generate picks.")
