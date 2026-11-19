# GTA VI Trophy Rarity Forecast

[![competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01/competition.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Aboys25/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Aboys25/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
on make.fun: forecast every GTA VI (PS5) trophy's Exophase earn rate at
2026-11-26T00:00:00Z as five quantiles (5%, 25%, 50%, 75%, 95%), scored by
pinball loss on the logit scale.

## The program

`main.py` (Python 3.13, stdlib only, no network) reads one case
`/input/<id>.in` and writes `/output/<id>.out`. Outside the grader,
`MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` stand in for `/input` / `/output`:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## The model

The starter (house baseline #1) only says every rate stays where it is, with a
random-walk spread. That is the floor, not the forecast. This entry models
growth:

- **Logistic growth toward a fitted ceiling.** For each trophy, work in
  `u = logit(rate / cap)`, where growth is near-linear, so the fitted slope
  extrapolates and bends into the ceiling instead of running away to 100%.
- **The ceiling is fitted, not just assumed.** A grid of candidate ceilings is
  scored by the weighted least-squares residual of the `u`-vs-time fit (on a
  log scale, against the 2-decimal rounding noise floor) plus a penalty for
  moving away from the prior ceiling for the trophy's kind (story chapters
  saturate high; the platinum and the 100%-completion grind saturate low).
  When the history already bends, the data pulls the ceiling down; when the
  trophy is still in its exponential phase, the prior holds.
- **Slope per trophy** from weighted least squares over the last 36 h of
  reads, weighted by players-tracked and tilted toward recent reads, then
  **shrunk toward its group's slope** (same kind of trophy) and the global
  slope — a trophy flat at 0% borrows the slope of trophies of its kind that
  already move.
- **Stale-page handling.** Consecutive reads Exophase never refreshed (same
  tracked count, same rates) collapse to one point — launch night sat frozen
  for hours — and a refresh jump (tracked leaping >1.5x between reads)
  discards the stale page before it, so the slope is fitted on the live page.
- **Honest, horizon-aware spread.**
  `sigma^2 = base^2 + (0.30·sqrt(h))^2 + (slope_se·h)^2` in logit units, so the
  final's ~4.5-day horizon gets a far wider band than a 90-minute practice
  horizon, and trophies that never moved get a wider band than trophies with a
  clean slope. Quantiles are `median_logit + z_tau · sigma`.

Deterministic: pure arithmetic over the case file, no clock, no randomness.
Runs in ~0.3 s per case (limit 8 s).

## Backtest

The practice history is flat (Exophase froze launch night), so no real-data
backtest exists. Instead a synthetic week-one was simulated with growth
parameters deliberately different from this model's priors (different ceilings
and speeds, four scenarios, six random seeds), generating 30-minute reads for
~2 days with a frozen launch-night page until the first refresh, then scoring
pinball loss on the logit scale against the starter. Final-horizon
(6,510 min) results, mean over seeds:

| scenario | this model | starter | improvement |
| --- | --- | --- | --- |
| base (60k tracked) | 0.207 | 0.620 | 66% |
| slow (25k, caps ×0.7, speed ×0.7) | 0.218 | 0.643 | 66% |
| fast (150k, caps ×1.3, speed ×1.3) | 0.202 | 0.539 | 62% |
| low ceilings (caps ×0.5) | 0.214 | 0.361 | 41% |

The model also beats the starter at every shorter horizon tested (90 min,
6 h, 24 h: +44% to +89%). The low-ceiling scenario is the stress test: it is
where a plain logit-linear extrapolation overshoots by whole logit units, and
where the fitted ceiling earns its keep.

## Verification

- `p00.in` / `p01.in`: valid output (55 rows, one per trophy, five finite
  0–100 non-decreasing quantiles each, 2,681 bytes ≤ 65,536).
- Determinism: two runs from different working directories with different
  `PYTHONHASHSEED` values produce byte-identical output
  (sha256 `04de4f35cbc4f984a02c0834657e61d01de6e38fb23e98e4135562a975ab6bdc` for
  both practice cases).
- Edge cases pass without crashing and with valid output: single read, all
  rates null, all rates 0%, empty history, tracked-count jump with a 100%
  and a 0% trophy, zero horizon, 30-day horizon, partial nulls, unparseable
  timestamps, fast-rising rates, all rates at 90%.
- No network or non-stdlib imports (`urllib`/`socket`/`requests`/`subprocess`
  absent).
- Kit check (`node cli/makefun.mjs check --kit . --repo . --worktree`): passes
  `competition_open`, `snapshot`, `license_osi`, `entrypoint`. The only failing
  check is `transcripts_present`, which is supplied at submit time by the
  bundle, not by this repository.

## Risks

- The final's true ceiling per trophy is unknowable from two days of data;
  the fitted ceiling mitigates, but does not remove, overshoot risk if the
  real week-one saturates much lower than any history bend suggests.
- Exophase refresh timing is irregular: a long freeze followed by a jump
  between the deadline and the target would move the answer in ways no
  history-based model can foresee.
- Practice cases cannot validate the slope model (their history is flat);
  the backtest above is a simulation, not real data.
