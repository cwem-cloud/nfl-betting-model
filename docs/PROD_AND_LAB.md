# Prod and Lab

Pocket Capper has two halves. **Prod** is the card: a small set of frozen rules that runs on
schedule and tells you what to bet, at which book, for how much. **Lab** is everything we're still
learning about: it runs alongside prod every week, logs what it *would* have bet, and never touches
the card, the record or the exposure caps. Ideas move from lab to prod only through the gate below.

The goal is steady, not clever: the card changes rarely and for a measured reason, while the lab is
free to try many things because trying costs nothing.

## Prod (rules v2026-10-30, frozen through week 12)

| Piece | What it does |
|---|---|
| Blind projections | Team ratings and player usage projected **before** any line is pulled |
| Market anchor | Probabilities start from the de-vigged market and move only part of the way toward the model (sides 20%, props by stat) |
| On the card | NFL spreads + ML, NFL player props + anytime TD, CFB totals (half stakes) |
| Tracking only | NFL totals (2-8 live), CFB spreads / ML: priced and graded as if bet, never bet |
| Line shopping | Every pick takes the best price among **DK, FanDuel, BetMGM, Fanatics** at DK's number; a price more than 6% (implied) better than DK's is treated as stale data |
| Sizing | ¼ Kelly, 0.5-3U, daily exposure caps per sport and bucket |
| Guards | EV caps ("too good to be true"), no chasing moved lines, model-far-off-market = missing news, unstable roles (a lead back returning) |
| Scoring | **Closing-line value first** (measured at the book you bet), record second |
| Arb scanner | Arbs and middles across your four books, informational |

Schedule (UTC cron in `.github/workflows/pocket_capper.yml`): slates Thu-Mon mornings, a Sunday
re-run after inactives, odds snapshots for line movement and CLV, Tuesday grading, Tuesday lab run.

## Lab

| Experiment | Where | Logged as |
|---|---|---|
| Situational angles (travel, bye, blowout bounce, wind, ...) | `models/angles.py` | shadow picks `source=angle` |
| Matchup factors (fill-in corners, coverage LBs, DL/OL starters out) | `models/matchups.py` | shadow prop picks `source=angle` |
| Stacks (2+ angles or factors on one bet) | same | `angle-stack`, `angle-stack_proven`, `angle-mstack` |
| Switched-off markets | `sports.*.markets` | `tracking=1` |
| Paper ladders | `market/ladder.py` | `ladder_plans` |
| Reasoning capper (Claude) | `engine/capper.py` | `source=capper` (dormant until a token is set) |

Research tools: `scripts/angle_lab.py` re-tests every angle and factor on history;
`scripts/report.py` prints the live scorecard (backtest, live record, ROI, CLV, verdict).

## The gate

An idea goes to prod only when **all** of these hold:

1. **Backtest**: beats 52.4% (-110 break-even) in *both* 2010-19 and 2020-25 for game angles, or
   shows the same sign in both of the last two seasons for player factors.
2. **Live shadow**: at least 30 graded shadow picks, winning > 54% with positive units.
3. **Closing line**: shadow picks beat the close at least as often as they lose to it.
4. **Timing**: promoted at the end of a freeze window (or immediately only if it's a bug fix),
   with a new `rules.version`, so every result belongs to exactly one rule set.

The scorecard marks candidates that pass 2-3 as `READY: promote?`.

**Demotion** works the same way in reverse: a prod market that loses to the close on 55%+ of 50+
picks, or is clearly losing live, goes back to tracking-only (as NFL totals did).

## Complexity budget

- Prod changes at most once per freeze window, each with a written reason in `config.yaml`.
- Every lab idea must be a rule that produces shadow picks; if it can't be graded, it can't be promoted.
- Anything that hasn't produced a single graded pick in a month gets removed (NBA was removed on that basis).
