import pandas as pd
import numpy as np

def compute_targets(df_games: pd.DataFrame, df_odds: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(df_games, df_odds, on=["season","week","home_team","away_team"], how="left")
    df["spread_close_away"] = -df["spread_close_home"]
    df["total_points"] = df["home_score"] + df["away_score"]
    df["margin"] = df["home_score"] - df["away_score"]

    # ATS target: did home -spread cover?
    df["ats_home_cover"] = (df["margin"] + (-df["spread_close_home"])) > 0
    df["ats_home_cover"] = df["ats_home_cover"].astype(int)

    # ML target: home win
    df["ml_home_win"] = (df["margin"] > 0).astype(int)

    # Totals target: regression to total_points
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Minimal feature set for demo purposes
    feats = [
        "home_EPA_off_rolling8","home_EPA_def_rolling8","away_EPA_off_rolling8","away_EPA_def_rolling8",
        "qb_adj_home","qb_adj_away","rest_days_home","rest_days_away",
        "wind_mph","temp_f","altitude_bucket","is_divisional","is_prime_time"
    ]
    # Encode stadium_type
    df = df.copy()
    df["is_dome"] = (df["stadium_type"] == "dome").astype(int)
    feats.append("is_dome")
    return df, feats
