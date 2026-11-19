"""gta6-trophy-rarity-forecast-01: forecast each GTA VI (PS5) trophy's Exophase earn rate
at the case target as five quantiles, scored by pinball loss on the logit scale.

Model (written for this entry; the starter's persistence is only the floor it must beat):

  * Rates grow logistically toward a per-trophy ceiling.  For trophy i with ceiling
    cap_i, work in u = logit(rate / cap_i): in u space growth is (near-)linear, so a
    slope fitted on the case history extrapolates and bends into the ceiling instead
    of running away to 100%.
  * The ceiling is fitted, not just assumed: a grid of candidate ceilings is scored
    by the weighted least-squares residual of the u-vs-time fit plus a penalty for
    moving away from the prior ceiling (the trophy kind's plausible week-one
    ceiling: story chapters saturate high, the platinum and the 100%-completion
    grind saturate very low).  When the history already bends, the data pulls the
    ceiling down; when the trophy is still in its exponential phase, the prior
    holds.  A ceiling far above the truth is what makes a plain logit-linear
    extrapolation overshoot by whole logit units, so this is the step that matters.
  * Slope per trophy: weighted least squares of u against time (days), weights by
    players-tracked (a rate from more tracked players is a sharper read) tilted
    toward recent reads, over the last 36 h.  Consecutive reads Exophase never
    refreshed (same tracked count and same rates) collapse to one point first: on
    launch night the page sat frozen for hours, and counting those identical reads
    would drag every slope toward zero.  A refresh jump (tracked leaping >1.5x
    between reads) discards the stale page before it.
  * Shrinkage: a trophy's own slope is noisy (55 trophies, ~2 days of reads, rates
    printed to 2 decimals).  Shrink it toward its group's slope (same kind of
    trophy) and the global slope, weighted by how much the trophy actually moved.
    A trophy flat at 0% borrows the slope of trophies of its kind that already move.
  * Spread: honest and horizon-aware.  sigma_logit^2 = base^2 + (rw * sqrt(h))^2 +
    (slope_se * h)^2, so the final's ~4.5-day horizon gets a far wider band than a
    90-minute practice horizon, and trophies with no movement get a wider band than
    trophies with a clean slope.  Quantiles are median_logit + z_tau * sigma.
  * Tracked-player dilution is handled implicitly: the regression is on the rate
    itself, so if a flood of new tracked players drags a late-game trophy's rate
    down while its earners still rise, the fitted slope is what it is.

Determinism: pure arithmetic over the case file, trophies in file order, no clock,
no randomness, no network, stdlib only.  Runs in well under a second per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module shipped
# beside this file would import.  This entry is a single file; the line stays for
# parity with the kit's starter layout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- tuning constants (chosen on the practice cases + a synthetic growth
# simulation; see the backtest notes in the repo README) ---
FIT_WINDOW_HOURS = 36.0    # only the most recent 36 h of reads drive the fit
BASE_SIGMA = 0.05          # logit spread floor: read noise + refresh jitter
RW_SIGMA = 0.30            # random-walk logit spread per sqrt-day
SLOPE_SE_GOOD = 0.10       # slope standard error (logit/day) for a moving trophy
SLOPE_SE_FLAT = 0.28       # ... for a trophy that never moved in its history
GROUP_SHRINK_W = 3.0       # weight of the group slope in the shrinkage pool
GLOBAL_SHRINK_W = 1.0      # weight of the global slope in the shrinkage pool
MAX_SLOPE_W = 8.0          # cap on the weight a trophy's own slope can carry
CAP_PENALTY = 0.5          # prior penalty on log(cap / prior cap) in cap fit (log-MSR scale)
CAP_GRID_N = 28            # candidate ceilings per trophy


def clamp(x, lo, hi):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (same as the starter's JavaScript-style round)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    """ISO 8601 UTC -> aware datetime; returns None on garbage (never crash)."""
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Trophy kinds and prior ceilings.
#
# A trophy's prior ceiling is the earn rate it plausibly reaches a week after
# launch among Exophase-tracked players (engaged early adopters).  Story
# chapters saturate high; the platinum, 100% completion and the long grinds
# saturate low.  The ceiling fit below can move away from the prior when the
# history bends, but the prior is the anchor when it does not.
# ---------------------------------------------------------------------------
def classify(trophy):
    """-> (group, prior ceiling percent) from tier + description keywords."""
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    text = desc + " " + name

    if tier == "platinum":
        return ("platinum", 12.0)
    if "100% completion" in desc or "100% completion" in text:
        return ("completion100", 8.0)
    if desc.startswith("complete chapter "):
        try:
            ch = int(desc.rsplit(" ", 1)[-1].rstrip("."))
        except ValueError:
            ch = 1
        return ("chapter", {1: 85.0, 2: 78.0, 3: 72.0, 4: 66.0}.get(ch, 60.0))
    if "complete the final mission" in desc:
        return ("final_mission", 55.0)
    if "final mission for" in desc or "mission strands for strangers" in desc:
        return ("strand", 35.0)
    if any(k in desc for k in ("complete 250 stunts", "collectible card",
                               "waink it at", "complete 50 challenges")):
        return ("grind", 8.0)
    if "gold medal" in desc and "shooting range" in desc:
        return ("skill", 10.0)
    if desc.startswith("complete ") and "mission" not in desc:
        # Named story missions (Breakfast Run, The Cartel Heist, ...): on the
        # critical path, so they track the chapters.
        return ("story_mission", 62.0)
    if any(k in desc for k in ("discover 100 vehicles", "explore all of leonida",
                               "purchase all garage", "register 10 vehicles")):
        return ("explore", 30.0)
    if any(k in desc for k in ("complete an arcade game", "beat cal in a fishing",
                               "enjoy a cocktail", "bowling alley", "bowl 'em over",
                               "spend $100,000", "partnership with mass appeal")):
        return ("activity_easy", 35.0)
    if any(k in desc for k in ("win 20 different races", "drone race",
                               "meditate at every spot", "hunt a unique animal",
                               "rob 3 shops", "rob 5 bank vaults", "rob an armored truck",
                               "rob an illegal operation", "combat focus",
                               "unscheduled departure", "without being detected",
                               "rhinos", "laundering marked bills",
                               "lose a 4 star wanted", "take down 5 enemies",
                               "choices made", "second time's the charm",
                               "what the hell just happened")):
        return ("activity_mid", 22.0)
    return ("misc_bronze", 25.0)


# ---------------------------------------------------------------------------
# History preparation
# ---------------------------------------------------------------------------
def dedup_reads(history):
    """Collapse runs of consecutive identical reads (same tracked, same rates):
    Exophase not refreshing is not player progress.  Keep the last of each run
    (the freshest timestamp carrying that page state)."""
    out = []
    for read in history:
        rates = read.get("rates")
        tracked = read.get("players_tracked")
        if out:
            prev = out[-1]
            if prev.get("players_tracked") == tracked and prev.get("rates") == rates:
                out[-1] = read
                continue
        out.append(read)
    return out


def working_rate(rate, tracked):
    """Rate to fit on: a 0% read means 'fewer than one earner', so use half an
    earner as the floor; clamp into (0, 100) so the logit is finite."""
    if rate is None:
        return None
    if rate <= 0:
        n = tracked if isinstance(tracked, (int, float)) and tracked > 0 else 638
        return clamp(0.5 / n * 100.0, 0.01, 99.9)
    return clamp(rate, 0.01, 99.9)


def wls_line(pts):
    """Weighted least squares line over (t, u, w).  Returns (a, b, mean_sq_resid)
    or (None, 0.0, 0.0) when the fit is degenerate."""
    sw = sum(w for _, _, w in pts)
    if sw <= 0 or len(pts) < 2:
        return None, 0.0, 0.0
    mt = sum(w * t for t, _, w in pts) / sw
    mu = sum(w * u for _, u, w in pts) / sw
    den = sum(w * (t - mt) ** 2 for t, _, w in pts)
    if den <= 1e-12:
        return None, 0.0, 0.0
    b = sum(w * (t - mt) * (u - mu) for t, u, w in pts) / den
    a = mu - b * mt
    msr = sum(w * (u - (a + b * t)) ** 2 for t, u, w in pts) / sw
    return a, b, msr


def fit_cap_and_slope(pts, prior_cap):
    """pts: (days, rate, weight) in the fit window.  Pick the ceiling that best
    explains the history (least-squares in u = logit(rate/cap) space, penalised
    toward the prior ceiling) and the slope at that ceiling.
    Returns (cap, a, b, n_moves)."""
    if len(pts) < 2:
        return prior_cap, None, 0.0, 0
    max_r = max(r for _, r, _ in pts)
    c_min = max(1.0, max_r * 1.1 + 0.05)
    c_max = 99.5
    if c_min >= c_max:
        return prior_cap, None, 0.0, 0
    n_moves = sum(1 for k in range(1, len(pts)) if abs(pts[k][1] - pts[k - 1][1]) > 0.005)
    # Log-spaced candidate ceilings, prior included.
    cands = set()
    for k in range(CAP_GRID_N):
        frac = k / (CAP_GRID_N - 1)
        cands.add(c_min * (c_max / c_min) ** frac)
    cands.add(prior_cap)
    best = None
    for cap in sorted(cands):
        if cap <= max_r:
            continue
        upts = [(t, math.log((r / cap) / (1.0 - r / cap)), w) for (t, r, w) in pts]
        a, b, msr = wls_line(upts)
        if a is None:
            continue
        # Log-scale residual: what matters is the MSR relative to the noise floor
        # (2-decimal rounding puts it near 1e-5), not its absolute size.  A
        # ceiling whose MSR is several times the floor is rejected by the data
        # even when the absolute gap looks tiny; a flat history gives every
        # ceiling the same MSR, so the prior penalty then decides.
        score = math.log(msr + 1e-5) + CAP_PENALTY * (math.log(cap / prior_cap)) ** 2
        if best is None or score < best[0]:
            best = (score, cap, a, b)
    if best is None:
        return prior_cap, None, 0.0, n_moves
    _, cap, a, b = best
    return cap, a, b, n_moves


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------
def forecast_case(case, name):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_days = max(0.0, float(case.get("horizon_minutes") or 0) / 1440.0)
    at = parse_time(case.get("at"))
    if at is None and history:
        at = parse_time(history[-1].get("at"))
    if at is None:
        at = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)

    reads = dedup_reads(history)
    # A refresh jump: Exophase sometimes sits on a stale page for hours, then
    # jumps (tracked 638 -> 3868 on launch night, every rate dropping).  The
    # stale reads before such a jump are not player behaviour: drop everything
    # up to the jump, so the fit runs on the live page only.
    for j in range(len(reads) - 1, 0, -1):
        t_prev = reads[j - 1].get("players_tracked")
        t_now = reads[j].get("players_tracked")
        if (isinstance(t_prev, (int, float)) and isinstance(t_now, (int, float))
                and t_prev > 0 and t_now > 1.5 * t_prev):
            reads = reads[j:]
            break
    # Slope window: the most recent FIT_WINDOW_HOURS before the last read.
    window_start = None
    if reads:
        last_t = parse_time(reads[-1].get("at"))
        if last_t is not None:
            window_start = last_t - timedelta(hours=FIT_WINDOW_HOURS)

    # Absolute time base: days since the first read in the window.
    t0 = None
    for read in reads:
        t = parse_time(read.get("at"))
        if t is not None and (window_start is None or t >= window_start):
            if t0 is None or t < t0:
                t0 = t
    if t0 is None:
        t0 = at

    fitted = []
    for i, trophy in enumerate(trophies):
        group, prior_cap = classify(trophy)
        pts = []
        last_rate = None
        last_tracked = None
        for read in reads:
            t = parse_time(read.get("at"))
            if t is None:
                continue
            if window_start is not None and t < window_start:
                continue
            rates = read.get("rates") or []
            if i >= len(rates):
                continue
            r = working_rate(rates[i], read.get("players_tracked"))
            if r is None:
                continue
            last_rate = rates[i]
            last_tracked = read.get("players_tracked")
            days = (t - t0).total_seconds() / 86400.0
            recency = 0.5 + 0.5 * clamp(days / max(0.5, FIT_WINDOW_HOURS / 24.0), 0.0, 1.0)
            w = clamp((read.get("players_tracked") or 638) / 1000.0, 0.1, 20.0) * recency
            pts.append((days, r, w))
        cap, a, b, n_moves = fit_cap_and_slope(pts, prior_cap)
        fitted.append({
            "trophy": trophy, "group": group, "prior_cap": prior_cap,
            "cap": cap, "a": a, "b": b, "n_moves": n_moves,
            "last_rate": last_rate, "last_tracked": last_tracked,
            "t_last": pts[-1][0] if pts else 0.0,
        })

    moving = [s for s in fitted if s["n_moves"] > 0 and s["a"] is not None]
    group_slopes = {}
    for s in moving:
        group_slopes.setdefault(s["group"], []).append(s["b"])
    group_slope = {g: sum(v) / len(v) for g, v in group_slopes.items()}
    global_slope = (sum(s["b"] for s in moving) / len(moving)) if moving else 0.0

    rows = []
    for s in fitted:
        trophy = s["trophy"]
        last_rate = s["last_rate"]
        if last_rate is None:
            # No read ever carried this trophy: tier prior + very wide band.
            prior = {"platinum": 0.5, "gold": 1.0, "silver": 0.8}.get(
                (trophy.get("tier") or "").lower(), 0.9)
            rows.append({"id": trophy["id"], "q": quantiles(prior, 1.4)})
            continue
        if last_rate >= 99.5:
            # Already (near-)universal: it cannot rise and barely falls.
            # Median at the clamp, not below it: a 100% trophy scored at 99.95
            # costs ~1.6 logit on every quantile.
            rows.append({"id": trophy["id"], "q": quantiles(99.99, 0.05)})
            continue

        r = working_rate(last_rate, s["last_tracked"])
        cap = s["cap"]
        # The fitted ceiling must sit above the last observed rate (a fast
        # riser can outgrow its prior ceiling); otherwise u is undefined.
        if cap <= r:
            cap = min(99.95, r * 1.1 + 0.05)
        # Anchor on the fitted line at the last read (smoother than the raw
        # last point, which carries 2-decimal rounding noise).
        if s["a"] is not None:
            u_last = s["a"] + s["b"] * s["t_last"]
        else:
            u_last = math.log((r / cap) / (1.0 - r / cap))
        # Shrink the slope: own slope weighted by movement, group + global.
        w_own = min(float(s["n_moves"]), MAX_SLOPE_W)
        g_slope = group_slope.get(s["group"], global_slope)
        if s["a"] is None or s["n_moves"] == 0:
            slope = g_slope
        else:
            slope = (w_own * s["b"] + GROUP_SHRINK_W * g_slope
                     + GLOBAL_SHRINK_W * global_slope) / (
                         w_own + GROUP_SHRINK_W + GLOBAL_SHRINK_W)
        u_target = u_last + slope * horizon_days
        median = clamp(cap * sigmoid(u_target), 0.01, 99.9)

        # Slope uncertainty: wide when the trophy never moved, tight otherwise.
        frac_own = w_own / (w_own + GROUP_SHRINK_W + GLOBAL_SHRINK_W)
        slope_se = SLOPE_SE_FLAT + (SLOPE_SE_GOOD - SLOPE_SE_FLAT) * frac_own
        sigma = math.sqrt(BASE_SIGMA ** 2 + (RW_SIGMA * math.sqrt(horizon_days)) ** 2
                          + (slope_se * horizon_days) ** 2)
        sigma = clamp(sigma, 0.05, 1.6)

        rows.append({"id": trophy["id"], "q": quantiles(median, sigma)})

    return {"trophies": rows}


def quantiles(median_pct, sigma):
    """Five quantiles around a median percent, logit scale, non-decreasing."""
    med_logit = logit(median_pct)
    out = []
    prev = -1.0
    for z in Z:
        v = round4(clamp(sigmoid(med_logit + z * sigma) * 100.0, 0.0, 100.0))
        if v < prev:
            v = prev
        out.append(v)
        prev = v
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
