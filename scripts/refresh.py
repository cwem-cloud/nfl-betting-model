"""Placeholder refresh script.

Replace the TODO sections with your data pulls (odds, injuries, weather).
You can run this on a schedule via GitHub Actions or any cron to precompute picks
and save them to data/processed/picks_latest.csv for the app to load quickly.
"""
import os, pandas as pd, datetime as dt

os.makedirs("data/processed", exist_ok=True)
# TODO: pull latest odds + games + features and retrain models

# Placeholder: write a timestamp file
pd.DataFrame([{"refreshed_at": dt.datetime.utcnow().isoformat()}]).to_csv("data/processed/last_refresh.csv", index=False)
print("Wrote data/processed/last_refresh.csv")
