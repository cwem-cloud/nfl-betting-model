# 🏈 Pocket Capper

A NFL + CFB betting model with one button. Hit **Run** and it:

1. **Handicaps every game blind** (no line) from opponent-adjusted, recency-weighted
   power ratings, then adjusts for QB injuries, rest, and kickoff weather. Each game gets a fair
   spread, total and win %, and every NFL skill player gets a stat projection.
   These numbers are saved **before** any odds are pulled, so the market can't anchor them.
2. **Pulls the market**: DraftKings plus up to 9 other books (including Pinnacle as the sharp
   reference), DK Network bet % and money % splits, and line movement from every past snapshot.
3. **Finds sharp signals**: DK off-market vs sharp books, reverse line movement, big-money
   vs public-ticket splits, steam moves, and plays you log from sharp sources such as Bambino.
4. **Prices the edge at DK**: blended win probability, EV, and units from ¼-Kelly
   adjusted by the sharp signals. Every play gets a rationale explaining how the number was built.
5. **Tracks everything**: plays are locked at the line first issued. They're auto-graded
   with closing line value (CLV), and the full history lives in the History tab.

No edge, no play. Games where the model and market disagree by an unreasonable amount get
flagged rather than bet, since that usually means the model is missing news.

## Dashboard tabs

| Tab | What's there |
|---|---|
| 🏈 NFL / 🎓 CFB | Plays with U rating, EV and rationale, plus the full slate: blind line vs DK vs sharp, adjustments, power ratings |
| 🎯 TD Zone | Anytime TD: model % (red-zone usage × blind team TD expectation) vs DK odds, fair odds, +EV plays |
| 📊 Prop Hub | NFL player props: blind projection vs DK line, model/market %, EV, filters |
| 📈 Line Market | Open → current at DK, sharp line, bet % / money %, all-books grid, movement chart, splits CSV import |
| 🧠 Sharp Inputs | Log plays from sharp sources. They become signals and get a graded record per source |
| 📚 History | Every pick ever: record, units, ROI, CLV, cumulative units, breakdown by bet type, rationale lookup, CSV export |

## Setup (about 15 minutes)

1. **Keys**, set as Streamlit secrets or GitHub Actions secrets (see `.streamlit/secrets.toml.example`):
   - `ODDS_API_KEY`: [the-odds-api.com](https://the-odds-api.com). The **20K-credit plan (~$30/mo)** covers
     5 runs a week with props plus odds snapshots (≈4–5K credits/month). The free 500 covers game lines only.
   - `CFBD_API_KEY`: free at [collegefootballdata.com/key](https://collegefootballdata.com/key).
   - `ANTHROPIC_API_KEY` *(optional)*: Claude writes a short prose rationale on top of the factual bullets.
   - `DATABASE_URL` *(strongly recommended)*: a free [Neon](https://neon.tech) or Supabase Postgres connection string.
     It keeps one shared history for the scheduled runs and the dashboard's Run button. Without it, history lives in
     `data/pocket_capper.db` (SQLite) and the GitHub Action commits it after each run. That works, but it grows the
     repo all season and Run-button picks made on a hosted dashboard aren't kept. To limit the growth, only changed
     prices are stored and old boards are pruned.
2. **Dashboard** on [Render](https://render.com) (free; the repo has a `Dockerfile` and `render.yaml`):
   **New → Blueprint** → pick this repo → paste the same keys when prompted → **Apply**. It redeploys
   on every push to `main`. The free tier sleeps after 15 idle minutes, so the first load takes about a minute.
   The same `Dockerfile` runs on Hugging Face Spaces, Railway or Fly. Streamlit Community Cloud also works:
   main file `app/app.py`.
3. **Schedule**: `.github/workflows/pocket_capper.yml` runs the full slate at 10:15 ET on Mon and Thu–Sun,
   re-runs Sunday at 11:45 ET after inactives are announced, snapshots odds and splits six times a day Thu–Mon,
   and does a grading sweep on Tuesday. You can also trigger it manually from the Actions tab.

Run locally:

```bash
pip install -r requirements.txt
streamlit run app/app.py               # dashboard with the Run button
python scripts/run_slate.py            # same thing headless
python scripts/backtest.py             # walk-forward NFL backtest vs closing lines
python scripts/backtest.py --props 2025
python -m pytest -q tests
```

## How the numbers are built

**Blind ratings** (`pocket_capper/models/team_ratings.py`): one ridge regression over every team-game:
`points = μ + OFF[team] + DEF[opp] + HFA`. Games are weighted with an exponential recency decay
(8-week half-life in the NFL), last season is discounted, and every team is shrunk toward
league average. The target blends actual points with **EPA-implied points** (NFL, from nflverse
play-by-play with garbage time removed) or **PPA-implied points** (CFB, from CFBD). That smooths out
scoreboard noise like pick-sixes and late garbage TDs.

**Key numbers**: margins are scored with a discrete distribution reweighted by real NFL
final-margin frequencies from 2002–2025, so 3 and 7 carry their true weight, and push
probability on -3 is priced.

**Blending with the market**: the blind number always comes first. For sizing, its win probability
is blended with the de-vigged sharp-book probability, with weights set by the backtest:

| Walk-forward 2023–25 (768 games) | Blind model | Closing line |
|---|---|---|
| Margin MAE | 10.33 | 9.81 |
| Total MAE | 10.29 | 10.11 |
| O/U hit rate when model ≥3 pts off close | 101-88 (53.4%) | |
| ATS hit rate when model ≥2 pts off close | 163-183 (47.1%) | |

Adjustments fit on history (nflverse 2020–25), not guessed:
- **QB out**: 561 games where a team's primary QB didn't start. Points lost ≈ 2.2 + 9.8 × his EPA/play,
  so a ~0.17 EPA/play starter is worth ~4 pts, and swapping out a below-average starter costs almost nothing.
- **Featured skill player out** (WR1 ≥24% targets, TE ≥20%, lead RB ≥55% carries): measured 1.5–1.8 pts.
  We apply -1.25 each, capped at -3, because the market already prices part of it.
- **Wind / roof**: totals fall ~0.35 pts per mph above 8 mph (capped at -6). Domes and closed roofs run +2 over the
  all-venue ratings. Below 25°F, -3.
- **Home field** is fit unpenalized, plus a +0.5 correction for what EPA-based ratings miss.
These were fit on 2020–25, which overlaps the 2023–25 test seasons, so read the table above as slightly flattering.

Read it honestly: **the closing line beats any public-data model on sides.** So sides use
a small model weight (0.20), and the edge there comes mostly from DK lagging the sharp books and
from sharp signals. Totals showed real signal, so they get more weight (0.30). The weights live in
`config.yaml` so you can retune them as CLV data builds up.

**Props / TD Zone** (`pocket_capper/models/nfl_props.py`): team volume (pass attempts and carries,
adjusted for the blind game script) × player share (decay-weighted targets and carries, with
injured players' share redistributed) × efficiency (shrunk to position priors) × opponent
adjustment. Distributions are gamma for yards and negative binomial for counts, with spreads fit to 2025
projection-vs-actual residuals. The anytime-TD
probability is λ = blind team TDs × player TD share (red-zone opportunity share, overall share,
actual TD share), then logistic-recalibrated on 2025 outcomes. That calibration is in-sample, so watch the
TD Zone's CLV and record as 2026 data comes in.

Props are **priced from the market first**. Our projections track the books' lines closely (correlation
0.88–0.92 for rushing and receiving in live testing), but they're less sharp. So we back out the mean
the market implies, move it 5–20% of the way toward our projection (the weight depends on how predictive
that stat proved in 2025), and re-price the DK line. TD props are blended the same way in log-odds (30%).

**CFB** (2025 walk-forward vs CFBD closing lines): margin MAE 12.98 vs 11.87 for the close, total 12.39 vs 12.27.
College sides lean almost entirely on the market (model weight 0.10) and totals slightly more (0.20).

**Units**: ¼-Kelly on the blended edge, ±0.25U per net agreeing or disagreeing sharp signal.
Sides and totals are capped at 0.5–3U, props at 1.5U, ATTD at 0.5U. Props are capped at 12 a run, ATTD at 8, and 3 per game.
Spread and ML on the same team are treated as one bet (only the better one is played).

**Trends** (ATS records by situation since 2012) are shown with sample sizes **as context
only**. They never move the number.

## About sharp sources and paid sites

Action Network Pro and SportsLine have no API, and scraping a logged-in account breaks their ToS.
Instead:
- Log sharp plays (e.g. Bambino) in **🧠 Sharp Inputs**. They feed the signals and get their own record.
- Export or copy splits from Action into a CSV and import them in **📈 Line Market**.
- DK's own bet % and money % come from the public DK Network splits page, all pages, NFL and CFB. The
  parser was built against the live page. If DK changes the layout it returns nothing rather than
  breaking the run, and the `diagnose` workflow will show it.

## Live diagnostics

`.github/workflows/diagnose.yml` hits every live source with your keys: Odds API lines and props, CFBD,
Open-Meteo and DK splits. It reports team and player name-matching gaps, runs a full slate into a
throwaway database and prints calibration stats. Run it from the Actions tab, or push to any `diag/*`
branch. It never touches your real history.

## Layout

```
pocket_capper/
  data/      nflverse, CFBD, Odds API, Open-Meteo weather, splits, team-name matching
  models/    ratings, NFL/CFB blind models, props, key-number distributions, trends, backtests
  market/    odds math, sharp signals & movement, edge evaluation
  engine/    run orchestration, rationale, grading + CLV
  db.py      SQLite/Postgres: runs, projections, odds_snapshots, splits, sharp_plays, picks, boards
app/app.py   Streamlit dashboard
scripts/     run_slate, snapshot_odds, grade, backtest
```

Bet responsibly. This is a decision-support tool, and every model loses some weeks. Judge it on CLV and
results over hundreds of plays, not one Sunday.
