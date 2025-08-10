
# NFL Betting Model Starter

A production-friendly starter repo for an NFL ATS/Totals/Moneyline model with:
- Weekly "Refresh" workflow (press a button in the Streamlit app).
- Data layers for games, odds, weather, injuries, and market context.
- Baseline modeling (logistic for ATS/ML, regression for Totals) with calibration.
- Picks with **1–10 confidence** + short rationale.
- Edge filters and basic bankroll logic.

> This is a scaffold with mock data so you can run it immediately. Swap in real sources later.

## Quickstart

```bash
# 1) Create a fresh env (recommended)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install deps
pip install -r requirements.txt

# 3) Launch app
streamlit run app/app.py
```

## Data Expectations (replace mocks later)

- `data/raw/games/`: historical game-level features (by week).
- `data/raw/odds/`: opening & closing lines from multiple books.
- Optional: `data/raw/weather/`, `data/raw/injuries/`, `data/raw/market/` (tickets vs handle).

### Suggested Sources (swap in your credentials/pipelines)
- **Play-by-play & team stats**: nflfastR (SportsDataverse).
- **Odds history**: The Odds API, Pinnacle, book-specific CSVs or paid feeds.
- **Weather**: NWS/NOAA/VisualCrossing.
- **Injuries**: team reports or paid feeds.
- **Market percentages**: book or aggregators.

## Modeling Notes
- Rolling features (4/8/16 games) with exponential decay.
- QB/Coach adjustments; opponent-adjusted EPA/SR; trench metrics.
- Market features: open→close movement, price dispersion, key numbers.
- Walk-forward validation by week; calibration with Platt/Isotonic.
- Metrics: ROI, CLV vs close, Brier/log loss, calibration curve.

## App Features
- Refresh button to (re)build features → fit → score next slate.
- Picks table with implied edge, confidence 1–10, and rationale text.
- Filters: min edge, bet type (ATS/ML/Totals), and book.

---

**Disclaimer:** For educational use only. No guarantee of profitability. Bet responsibly.
