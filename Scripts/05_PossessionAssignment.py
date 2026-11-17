"""Label every frame H / A / D: dead-ball detection (04) + team-control evidence + a decoder that
refuses to flicker.

Run with:
    uv run python Scripts/05_PossessionAssignment.py            # labels + diagnostics; figure saved
    uv run python Scripts/05_PossessionAssignment.py --show     # ... and also open the figure
    uv run python Scripts/05_PossessionAssignment.py --refresh  # rebuild the tracking cache
    uv run python Scripts/05_PossessionAssignment.py --no-tags  # no Smart Tagging prior at all

Outputs go to data/derived/possession/ (inside data/, so git ignores them):
    submission/NOV4.csv         THE SUBMISSION: frame,label for all 199 319 video frames, named
                                after the team as SUBMISSION.md asks (<team_name>.csv)
    dead_spans.csv              one row per dead-ball span: why it is dead, raw vs refined bounds
    possessions.csv             one row per decoded H/A spell
    timeline.png                control evidence and decoded labels over each half
    tracking_cache.npz          positions/signals cache, so re-runs skip the ~60 s kloppy load

Nothing here reads the ground truth. Diagnostics are label-free (counts, durations, agreement with
Smart Tagging). Score with the official scorer separately, as a correctness check only.


HOW IT WORKS
============

The metric (spellF1) treats every maximal H-run or A-run as one possession and matches them by
temporal IoU. A one-second flicker H H A H H turns one correct possession into three wrong ones,
so the whole design is built around NOT switching unless the evidence is sustained.

    tracking -> STAGE 1 dead-ball spans -> alive segments
             -> STAGE 2 per-frame control evidence e(t) in [-1, +1]   (+ = H, - = A)
             -> DECODER per alive segment: Viterbi with a switching cost, then a minimum spell length

STAGE 1  Dead-ball spans
------------------------
1a. Velocity drops, exactly as Script 04 finds them (3 s player speed / 120 s local tempo,
    hysteresis 0.6 / 0.8, merge < 2 s, keep >= 4 s). Not every drop is a dead ball -- a slow
    build-up is a drop too -- so a drop counts as DEAD only if one of these holds inside it:
      * 04's dead-ball flag: the ball is tracked < 50% of the drop (out of play / in hands)
      * 04's break flag: > 90 s or < 10 players tracked (half-time, long injuries)
      * the ball is outside the pitch for >= 0.4 s
      * the ball is "parked": tracked and slower than 0.5 m/s for >= 3 s (placed for a restart)
      * "long stop": the drop lasts >= 12 s and the speed ratio reaches <= 0.5 (both teams at half
        their tempo that long is a stoppage, even with the ball tracked and being carried back)
    Drops failing all five are open-play lulls and stay ALIVE.
1b. Out-of-play spans (fast stoppages 04 cannot see, e.g. a throw-in taken in 5 s): the ball is
    tracked > 1 m outside the lines for >= 0.4 s. Dead from the first outside frame until the first
    kick of the ball after it is back inside.
1c. The match scope is the two graded halves published in DATA.md (GRADED_HALVES), not the
    Smart Tagging kick-off/last-tag frames. Outside them nothing is decoded and the label is X.
1d. Boundary refinement. A velocity drop is smoothed over 3 s, so its edges are late/early by
    seconds -- at IoU 0.7 that loses the possession. Each span's edges are snapped to ball events:
      * START: in [start - 5 s, start + 10 s], find the ball's last fast movement inside the pitch;
        the span starts at the first frame after it where the ball is outside the lines or lost.
        (The shot/pass that went out, then the line crossing.) If the ball kept moving inside the
        pitch after the drop began, play was still on, so the start moves to after that movement.
      * END (the restart): in [end - 8 s, end + 2 s], candidate restarts are "kick onsets": ball
        speed > 4 m/s, sustained (3 of the next 5 frames), after >= 1 s without fast movement.
        Pick the onset that follows the LONGEST pause -- the restart ends the stoppage, while a pass
        straight after it follows only a short pause. No onset -> the ball's reappearance -> keep.
      * Not applied where it misfires: an out-of-play span keeps its start (the line crossing is
        already exact), and a long stop keeps both bounds (touches during the stop look like play).
1e. Clean-up: overlapping spans are merged; an alive gap < 3 s between two dead spans becomes dead
    (a restart that immediately stops again is far likelier a detection gap than a possession);
    dead spans < 1 s are dropped.

STAGE 2  Team-control evidence, per frame, in [-1, +1]
-----------------------------------------------------
Only where the ball is tracked (no ball -> no evidence, the decoder carries the current team):
    d_H, d_A   distance (m) from the ball to the nearest home / away player. A min over a team's
               slots, so duplicated or fragmented tracklets don't matter.
    who        tanh((d_A - d_H) / 1.5 m): +1 clearly H nearer, -1 clearly A, ~0 contested.
    w_dist     1 if someone is within 1.5 m of the ball, decaying exp(-(d - 1.5)/2 m) beyond:
               a ball 8 m from everyone is nobody's.
    w_speed    1 up to 7 m/s (dribbling pace), falling linearly to 0 at 15 m/s: a ball moving that
               fast is a pass or shot in flight, and whoever it flies past has not won it.
    e(t) = w_dist * w_speed * who
Optional weak prior (--no-tags to switch off): Smart Tagging phases that name the team in
possession (BUILD UP, PROGRESSION, CREATING CHANCES, LONG BALL, ATTACKING TRANSITION) add
+/-0.02 per frame, only on frames where the ball is NOT tracked. Their intervals are templated
(30 s windows), so they must never set a boundary -- they only nudge blind stretches.
"DEFENDING ..." codes are their mirrored duplicates, so they are skipped rather than inverted.

DECODER  (per alive segment, so a dead span always separates possessions)
------------------------------------------------------------------------
* 2-state Viterbi over {H, A}. Each frame scores +e(t) if labelled H and -e(t) if labelled A, so
  the two labels differ by 2|e| per frame; every switch costs SWITCH_COST = 15. A permanent change
  therefore needs sum|e| > 7.5 of opposing evidence (~0.3 s at e = 1, ~0.4 s at a typical 0.8),
  and a brief "visit" H -> A -> H pays twice, needing sum|e| > 15 (~0.6-0.75 s). A one-off touch,
  a deflection or a ball flying past the other team never pays for itself; visits that do pay
  but last < 2 s are then removed by the minimum-spell rule below.
* Ties go to switching as LATE as possible: frames with no evidence (ball in flight, ball lost)
  stay with the old team, so the boundary lands on the first frame of the new team's control, not
  in the middle of the pass.
* Who restarts is left to the evidence. A "ball out -> other team restarts" prior was tried and
  switched off (RESTART_PRIOR = 0): the law follows the last TOUCH, and a clearance that goes out
  is one touch inside the other team's possession, so the decoded team before the stoppage is a
  poor proxy. The run still prints how often the restart went to the other team. A tiny
  KEEP_PRIOR only breaks ties in segments with no evidence at all.
* Minimum spell: any H/A spell shorter than 2 s is absorbed into its neighbour (shortest first).
  Short true possessions do exist, but they are near-impossible to match at IoU >= 0.5, and every
  false one costs a false positive AND breaks the IoU of the possession around it.
"""

import argparse
import importlib.util
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from morethanahack.helpers.config import COLOURS, DATA_ROOT, DEFAULT_SOURCE, PROJECT_ROOT
from morethanahack.helpers.explorer import pitch_dimensions, team_metadata
from morethanahack.helpers.loader import VIDEO_FPS, load_smart_tagging


def _load_script(filename):
    """Scripts/0X_*.py start with a digit, so they can't be imported by name."""
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location("script_" + path.stem.split("_", 1)[1], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v03 = _load_script("03_SetPieceIntensity.py")       # in_play_mask
v04 = _load_script("04_VelocityDropDetection.py")   # drop detection, runs, sec, match_matrix

sec, runs = v04.sec, v04.runs

# ---------------------------------------------------------------- params

# Stage 1: dead balls
OUT_MARGIN_M = 1.0          # ball this far beyond a line counts as out (tracking jitter below)
OUT_MIN_S = 0.4             # ... for at least this long
PARKED_SPEED = 0.5          # m/s: a ball slower than this ...
PARKED_S = 3.0              # ... for this long, inside a drop, has been placed for a restart
LONG_STOP_S = 12.0          # a drop this long ...
LONG_STOP_RATIO = 0.5       # ... reaching this speed ratio (half the local tempo) is a stoppage:
                            # open play rarely keeps both teams that slow for that long
START_LOOKBACK_S = 5        # span start refinement window: [start - 5 s, start + 10 s]
START_LOOKAHEAD_S = 10
END_LOOKBACK_S = 8          # restart search window: [end - 8 s, end + 2 s]
END_LOOKAHEAD_S = 2
OUT_RESTART_SEARCH_S = 15   # out-of-play span: look this long after re-entry for the restart kick
MOVING_SPEED = 4.0          # m/s: a ball faster than this has been kicked/thrown
ONSET_SUSTAIN = (3, 5)      # ... on 3 of the next 5 frames
ONSET_QUIET_S = 1.0         # ... after at least 1 s without fast movement
MIN_ALIVE_S = 3.0           # alive gap between two dead spans shorter than this -> dead
MIN_DEAD_S = 1.0            # dead spans shorter than this -> dropped

# Stage 2: control evidence
CONTROL_R = 1.5             # m: within this the nearest player controls the ball
CONTROL_DECAY = 2.0         # m: e-folding distance of control beyond CONTROL_R
SEPARATION_SCALE = 1.5      # m: d_A - d_H at which the evidence is tanh(1) = 0.76
FREE_SPEED, FLIGHT_SPEED = 7.0, 15.0   # m/s: full weight up to 7, none from 15 (in flight)
FAR_M = 30.0                # stand-in distance for a team with no tracked player
TAG_WEIGHT = 0.02           # Smart Tagging prior per frame, ball-missing frames only
POSSESSION_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "LONG BALL",
                    "ATTACKING TRANSITION")

# Decoder
SWITCH_COST = 15.0          # score a team change must earn; labels differ by 2|e| per frame, so a
                            # change needs sum|e| > 7.5 of opposing evidence, a visit > 15
RESTART_PRIOR = 0.0         # head start for the non-possessing team after the ball went out.
                            # OFF: the law gives the restart to the team that did not TOUCH it
                            # last, but a clearance/tackle that puts it out is one touch, not a
                            # possession -- the decoded team before the stoppage agreed with
                            # "other team restarts" only ~41% of the time (label-free check).
KEEP_PRIOR = 0.5            # tie-breaker only: keep the previous team when a segment is blind
MIN_SPELL_S = 2.0           # H/A spells shorter than this are absorbed by their neighbours

HOME, AWAY = 0, 1
SUBMISSION_ROWS = 199_319   # tactical video frames; the validator checks this too
# The graded halves, inclusive video frames (archive/REPO/DATA.md "Which frames are graded", also
# games.half_video_blocks("BAR-ATH")). Published by the hosts as boundaries, not labels. They define
# the match scope: each half's first frame starts a fresh possession segment, and everything outside
# is ungraded, so it is written as X.
GRADED_HALVES = ((11_185, 90_651), (110_808, 190_197))
VALIDATOR = PROJECT_ROOT / "archive" / "REPO" / "evaluation" / "validate_submission.py"
OUT_DIR = DATA_ROOT / "derived" / "possession"
CACHE = OUT_DIR / "tracking_cache.npz"
TEAM_NAME = "NOV4"          # the submission is named after the team: <team_name>.csv
SUBMISSION_FILE = OUT_DIR / "submission" / f"{TEAM_NAME}.csv"

INK, MUTED, GRID, DEAD_COLOUR = "#333333", "#888888", "#E5E5E5", "#B7BDC6"


# ================================================================ loading (cached)

def load_tracking(refresh=False):
    """Per-frame arrays, row i = video frame i. Positions and ball in metres.

    signals: 04's frame signals (speed, n_tracked, ball_x, ball_y, ball_speed, ball_tracked)
    pos:     (frames, slots, 2) float32 player positions, NaN when the slot is empty
    team:    (slots,) HOME / AWAY
    """
    if CACHE.exists() and not refresh:
        z = np.load(CACHE, allow_pickle=False)
        signals = pd.DataFrame({c: z[f"sig_{c}"] for c in z["sig_cols"]})
        return (signals, z["pos"], z["team"], tuple(z["pitch"]),
                (str(z["names"][0]), str(z["names"][1])))

    print("loading tracking (kloppy, ~60 s; cached afterwards) ...")
    dataset, frames = v04.load_frames(DEFAULT_SOURCE)
    signals, pitch = v04.signals_from_frames(dataset, frames)

    teams = {t["ground"]: t for t in team_metadata(dataset)}
    mx, my = pitch_dimensions(dataset)["metres_per_unit"]
    ids, team = [], []
    for code, ground in ((HOME, "home"), (AWAY, "away")):
        for p in teams[ground]["players"]:
            ids.append(p["player_id"])
            team.append(code)
    missing = [c for pid in ids for c in (f"{pid}_x", f"{pid}_y") if c not in frames.columns]
    if missing:
        raise KeyError(f"{len(missing)} player position columns missing from to_df(), e.g. {missing[:2]}")
    pos = np.stack([frames[[f"{pid}_x" for pid in ids]].to_numpy(np.float32) * mx,
                    frames[[f"{pid}_y" for pid in ids]].to_numpy(np.float32) * my], axis=-1)
    team = np.array(team, dtype=np.int8)
    names = (teams["home"]["name"], teams["away"]["name"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, pos=pos, team=team, pitch=np.array(pitch), names=np.array(names, dtype="U"),
             sig_cols=np.array(list(signals.columns), dtype="U"), **{f"sig_{c}": signals[c].to_numpy() for c in signals})
    return signals, pos, team, pitch, names


# ================================================================ STAGE 1: dead balls

def ball_events(signals, pitch):
    """Per-frame ball booleans used by every dead-ball rule, plus the kick-onset table."""
    length, width = pitch
    bx, by = signals["ball_x"].to_numpy(), signals["ball_y"].to_numpy()
    tracked = ~np.isnan(bx)
    with np.errstate(invalid="ignore"):
        outside = tracked & ((bx < -OUT_MARGIN_M) | (bx > length + OUT_MARGIN_M) |
                             (by < -OUT_MARGIN_M) | (by > width + OUT_MARGIN_M))
        inside = tracked & (bx >= 0) & (bx <= length) & (by >= 0) & (by <= width)
    speed = np.nan_to_num(signals["ball_speed"].to_numpy(), nan=0.0)
    moving = speed > MOVING_SPEED
    parked = tracked & ~np.isnan(signals["ball_speed"].to_numpy()) & (speed < PARKED_SPEED)

    # kick onset: moving now, sustained, after a quiet second
    n = len(bx)
    cs = np.concatenate([[0], np.cumsum(moving)])
    idx = np.arange(n)
    k, need = ONSET_SUSTAIN[1], ONSET_SUSTAIN[0]
    ahead = cs[np.minimum(idx + k, n)] - cs[idx]
    q = sec(ONSET_QUIET_S)
    behind = cs[idx] - cs[np.maximum(idx - q, 0)]
    onset = moving & (ahead >= need) & (behind == 0) & (idx >= q)

    # length of the pause (no fast ball) that ends just before each frame
    quiet_len = np.zeros(n, dtype=np.int64)
    run = 0
    for i in range(n):              # one pass, ~0.1 s
        quiet_len[i] = run
        run = 0 if moving[i] else run + 1

    return {"tracked": tracked, "outside": outside, "inside": inside, "moving": moving,
            "parked": parked, "onset": onset, "quiet_len": quiet_len}


def classify_drops(drops, ev, ratio):
    """Which velocity drops are dead balls, and why (first reason that holds)."""
    reasons = []
    for _, d in drops.iterrows():
        a, b = int(d["start_frame"]), int(d["end_frame"])
        parked_run = max((e - s for s, e in zip(*runs(ev["parked"][a:b]))), default=0)
        deep = bool(np.any(ratio[a:b] <= LONG_STOP_RATIO))   # NaN compares False
        if d["is_break"]:
            reasons.append("break")
        elif d["dead_ball"]:
            reasons.append("ball lost")
        elif ev["outside"][a:b].sum() >= sec(OUT_MIN_S):
            reasons.append("ball out")
        elif parked_run >= sec(PARKED_S):
            reasons.append("ball parked")
        elif b - a >= sec(LONG_STOP_S) and deep:
            reasons.append("long stop")
        else:
            reasons.append(None)      # open-play lull -> alive
    return reasons


def out_of_play_spans(ev, n):
    """[(start, end, 'out of play')]: ball beyond a line -> first kick after it is back inside."""
    spans = []
    starts, ends = runs(ev["outside"])
    for a, b in zip(starts, ends):
        if b - a < sec(OUT_MIN_S):
            continue
        back = np.flatnonzero(ev["inside"][b:min(n, b + sec(120))])
        if not len(back):
            continue
        reentry = b + back[0]
        kicks = np.flatnonzero(ev["onset"][reentry:min(n, reentry + sec(OUT_RESTART_SEARCH_S))])
        spans.append((a, reentry + kicks[0] if len(kicks) else reentry, "out of play"))
    return spans


def refine_start(a, b, ev):
    lo, hi = max(0, a - sec(START_LOOKBACK_S)), min(a + sec(START_LOOKAHEAD_S), (a + b) // 2)
    live = np.flatnonzero(ev["moving"][lo:hi] & ev["inside"][lo:hi])
    if not len(live):
        return a                                  # ball never moving in play near the start
    last_live = lo + live[-1]
    gone = np.flatnonzero(ev["outside"][last_live + 1:b] | ~ev["tracked"][last_live + 1:b])
    if len(gone):
        return last_live + 1 + gone[0]            # ball crossed a line / was lost after the play
    return max(a, last_live + 1)                  # ball kept moving inside after the drop began


def refine_end(a, b, ev, n):
    lo, hi = max(a + 1, b - sec(END_LOOKBACK_S)), min(n, b + sec(END_LOOKAHEAD_S))
    cand = lo + np.flatnonzero(ev["onset"][lo:hi])
    if len(cand):
        q = ev["quiet_len"][cand]
        return int(cand[np.flatnonzero(q == q.max())[-1]])   # restart = kick after longest pause
    t = ev["tracked"]
    reappear = lo + np.flatnonzero(t[lo:hi] & ~t[lo - 1:hi - 1])
    if len(reappear):
        return int(reappear[np.argmin(np.abs(reappear - b))])
    return b


def dead_spans(signals, pitch, scope):
    """Stage 1. Returns (dead mask, span table)."""
    n = len(signals)
    ev = ball_events(signals, pitch)

    smooth, _, ratio = v04.speed_ratio(signals)
    drops = v04.describe_drops(v04.detect_drops(ratio, v04.ENTER, v04.EXIT), signals, smooth, ratio, pitch)
    reasons = classify_drops(drops, ev, ratio)
    raw = [(int(a), int(b), r) for a, b, r in zip(drops["start_frame"], drops["end_frame"], reasons)
           if r is not None]
    n_lulls = sum(r is None for r in reasons)
    raw += out_of_play_spans(ev, n)

    rows = []
    for a, b, why in sorted(raw):
        # an out-of-play span already starts on the line crossing -- the exact event. A long stop
        # keeps the drop's own bounds: the ball is carried / knocked about during it, and the
        # ball-event refinement reads those touches as play and shrinks the span.
        a2 = a if why in ("out of play", "long stop") else refine_start(a, b, ev)
        b2 = b if why == "long stop" else refine_end(a, b, ev, n)
        if b2 > a2:
            rows.append({"raw_start": a, "raw_end": b, "start_frame": a2, "end_frame": b2,
                         "reason": why, "ball_out": bool(ev["outside"][a:b2].any())})
    table = pd.DataFrame(rows)

    dead = ~scope
    for a, b in zip(table["start_frame"], table["end_frame"]):
        dead[a:b] = True
    # clean-up: short alive gaps -> dead, then short dead spans -> alive
    for a, b in zip(*runs(~dead)):
        if b - a < sec(MIN_ALIVE_S) and a > 0 and b < n:
            dead[a:b] = True
    for a, b in zip(*runs(dead)):
        if b - a < sec(MIN_DEAD_S):
            dead[a:b] = False
    return dead, table, ev, n_lulls


# ================================================================ STAGE 2: control evidence

def control_evidence(signals, pos, team):
    """e(t) in [-1, +1], + = home controls the ball. 0 where the ball isn't tracked."""
    n = len(signals)
    bx, by = signals["ball_x"].to_numpy(), signals["ball_y"].to_numpy()
    rows = np.flatnonzero(~np.isnan(bx))
    d = np.hypot(pos[rows, :, 0] - bx[rows, None], pos[rows, :, 1] - by[rows, None])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # a team with no tracked player -> FAR_M
        d_home = np.nan_to_num(np.nanmin(np.where(team == HOME, d, np.nan), axis=1), nan=FAR_M)
        d_away = np.nan_to_num(np.nanmin(np.where(team == AWAY, d, np.nan), axis=1), nan=FAR_M)

    near = np.minimum(d_home, d_away)
    w_dist = np.exp(-np.maximum(near - CONTROL_R, 0) / CONTROL_DECAY)
    speed = signals["ball_speed"].to_numpy()[rows]
    w_speed = np.clip((FLIGHT_SPEED - np.nan_to_num(speed, nan=FREE_SPEED)) /
                      (FLIGHT_SPEED - FREE_SPEED), 0, 1)
    who = np.tanh((d_away - d_home) / SEPARATION_SCALE)

    e = np.zeros(n)
    e[rows] = w_dist * w_speed * who
    return e


def tag_prior(tags, names, n):
    """+1 / -1 per frame inside a phase tag naming the team in possession, else 0."""
    prior = np.zeros(n)
    sign = {names[0]: 1.0, names[1]: -1.0}
    phases = tags[tags["code"].isin(POSSESSION_CODES) & tags["Team"].isin(sign)]
    for _, t in phases.iterrows():
        prior[max(0, t["start_frame"]):min(n, t["end_frame"] + 1)] += sign[t["Team"]]
    return np.clip(prior, -1, 1), len(phases)


# ================================================================ DECODER

def viterbi_2state(e, switch_cost, init):
    """Best H(0)/A(1) path for score sum(+e if H else -e) - switch_cost * switches.

    init = (bonus_H, bonus_A) at the first frame. On ties the path switches as late as possible,
    so evidence-free frames (ball in flight / lost) stay with the team that had the ball.
    """
    m = len(e)
    back = np.zeros((m, 2), dtype=bool)       # True = arrived by switching
    sh, sa = init[0] + e[0], init[1] - e[0]
    for t in range(1, m):
        stay_h, sw_h = sh, sa - switch_cost
        stay_a, sw_a = sa, sh - switch_cost
        back[t, 0], back[t, 1] = sw_h >= stay_h, sw_a >= stay_a   # ">=": tie -> switch now (late)
        sh = (sw_h if back[t, 0] else stay_h) + e[t]
        sa = (sw_a if back[t, 1] else stay_a) - e[t]
    path = np.empty(m, dtype=np.int8)
    state = 0 if sh > sa else 1 if sa > sh else int(init[1] > init[0])
    for t in range(m - 1, -1, -1):
        path[t] = state
        if back[t, state]:
            state = 1 - state
    return path


def enforce_min_spell(path, min_len):
    """Absorb spells shorter than min_len into their neighbours, shortest first."""
    path = path.copy()
    while True:
        edges = np.flatnonzero(np.diff(path)) + 1
        bounds = np.concatenate([[0], edges, [len(path)]])
        lens = np.diff(bounds)
        if len(lens) == 1 or lens.min() >= min_len:
            return path
        i = int(np.argmin(lens))
        a, b = bounds[i], bounds[i + 1]
        path[a:b] = 1 - path[a]   # 2 states: every neighbour of a spell is the other team


def decode(e, dead, restart_after_out):
    """Per alive segment: Viterbi + minimum spell. Returns labels 'H'/'A'/'D' and possessions."""
    labels = np.full(len(e), "D", dtype="<U1")
    prev_team = None
    stats = {"restart_prior_used": 0, "restart_prior_followed": 0}
    for a, b in zip(*runs(~dead)):
        init = [0.0, 0.0]
        if prev_team is not None:
            if restart_after_out[a]:
                init[1 - prev_team] += RESTART_PRIOR
                stats["restart_prior_used"] += 1
            if not restart_after_out[a] or RESTART_PRIOR == 0:
                init[prev_team] += KEEP_PRIOR
        path = viterbi_2state(e[a:b], SWITCH_COST, init)
        path = enforce_min_spell(path, sec(MIN_SPELL_S))
        if prev_team is not None and restart_after_out[a]:
            stats["restart_prior_followed"] += int(path[0] == 1 - prev_team)
        labels[a:b] = np.where(path == HOME, "H", "A")
        prev_team = int(path[-1])
    return labels, stats


def restart_flags(dead, spans, n):
    """True at the first alive frame after a dead span in which the ball went out of play."""
    flag = np.zeros(n, dtype=bool)
    out_ends = spans.loc[spans["ball_out"], "end_frame"].to_numpy()
    for a, b in zip(*runs(dead)):
        if b < n and ((out_ends >= a) & (out_ends <= b)).any():
            flag[b] = True
    return flag


def possession_table(labels, scope):
    rows = []
    for code in ("H", "A"):
        for a, b in zip(*runs((labels == code) & scope)):
            rows.append({"start_frame": a, "end_frame": b, "team": code,
                         "duration_s": (b - a) / VIDEO_FPS})
    return pd.DataFrame(rows).sort_values("start_frame", ignore_index=True)


# ================================================================ figure

def plot_timeline(e, labels, scope):
    halves = list(zip(*runs(scope)))
    fig, axes = plt.subplots(2 * len(halves), 1, figsize=(15, 2.6 * len(halves)),
                             gridspec_kw={"height_ratios": [3, 1] * len(halves)}, squeeze=False)
    minutes = np.arange(len(e)) / VIDEO_FPS / 60
    ev1 = pd.Series(e).rolling(sec(1), center=True, min_periods=1).mean().to_numpy()
    colour = {"H": COLOURS["home"], "A": COLOURS["away"], "D": DEAD_COLOUR}
    for h, (lo, hi) in enumerate(halves):
        ax, band = axes[2 * h, 0], axes[2 * h + 1, 0]
        ax.fill_between(minutes[lo:hi], 0, np.clip(ev1[lo:hi], 0, None), color=COLOURS["home"],
                        linewidth=0, label="home control (1 s mean)")
        ax.fill_between(minutes[lo:hi], 0, np.clip(ev1[lo:hi], None, 0), color=COLOURS["away"],
                        linewidth=0, label="away control (1 s mean)")
        ax.set_ylim(-1, 1)
        ax.set_ylabel("evidence")
        ax.set_xlim(minutes[lo], minutes[hi - 1])
        ax.tick_params(labelbottom=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        for code in "HAD":
            for a, b in zip(*runs(labels[lo:hi] == code)):
                band.axvspan(minutes[lo + a], minutes[lo + b - 1], color=colour[code], linewidth=0)
        band.set_xlim(minutes[lo], minutes[hi - 1])
        band.set_yticks([])
        band.set_ylabel("label", rotation=0, ha="right", va="center")
        for side in ("top", "right", "left"):
            band.spines[side].set_visible(False)
    from matplotlib.patches import Patch
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles.append(Patch(color=DEAD_COLOUR, label="dead ball (D)"))
    axes[0, 0].legend(handles=handles, loc="upper left", ncol=3, fontsize=8, frameon=False,
                      bbox_to_anchor=(0, 1.3))
    axes[-1, 0].set_xlabel("video time (minutes)")
    fig.suptitle("Control evidence (top) and decoded possession (bottom), per half",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    return fig


# ================================================================ main

def write_submission(path, labels):
    """The submission contract (archive/REPO/SUBMISSION.md): header `frame,label`, one row per
    video frame 0..N-1 in order, label in H/A/D/X, no blanks, no index column. Written with '\\n'
    endings so the file is byte-identical on every OS (the rules ask for the same file)."""
    if len(labels) != SUBMISSION_ROWS:
        raise ValueError(f"{len(labels)} labels, the submission needs {SUBMISSION_ROWS}")
    bad = set(np.unique(labels)) - set("HADX")
    if bad:
        raise ValueError(f"labels outside H/A/D/X: {sorted(bad)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(
        path, index=False, lineterminator="\n")
    return path


def official_validation(path):
    """Run the hosts' validator as a command (same checks as the official scorer).
    Returns True on pass. Its output is aggregate only: label counts and possession totals."""
    if not VALIDATOR.is_file():
        print(f"  validator not found at {VALIDATOR} -- skipped")
        return False
    result = subprocess.run([sys.executable, str(VALIDATOR), str(path), "--rows", str(SUBMISSION_ROWS)],
                            capture_output=True, text=True, cwd=VALIDATOR.parent)
    print("\n== Official validation (archive/REPO/evaluation/validate_submission.py) ==")
    print("  " + (result.stdout + result.stderr).strip().replace("\n", "\n  "))
    return result.returncode == 0


def graded_mask(n):
    """True on the graded frames: the two halves published in DATA.md."""
    mask = np.zeros(n, dtype=bool)
    for first, last in GRADED_HALVES:
        mask[first:last + 1] = True
    return mask


def main(show=False, refresh=False, use_tags=True):
    signals, pos, team, pitch, names = load_tracking(refresh)
    n = len(signals)
    tags = load_smart_tagging(DEFAULT_SOURCE)
    scope = graded_mask(n)
    tag_scope = v03.in_play_mask(tags, n)
    for (first, last), (a, b) in zip(GRADED_HALVES, zip(*runs(tag_scope))):
        print(f"  half {first}..{last}: Smart Tagging scope was {a}..{b - 1} "
              f"(start {a - first:+d}, end {b - 1 - last:+d} frames)")
    print(f"{n} frames, {pos.shape[1]} slots ({(team == HOME).sum()} home / {(team == AWAY).sum()} away), "
          f"match scope {scope.sum() / VIDEO_FPS / 60:.1f} min")

    # ---------- STAGE 1
    dead, spans, ev, n_lulls = dead_spans(signals, pitch, scope)
    in_scope_spans = list(zip(*runs(dead & scope)))
    print(f"\n== Stage 1: dead balls ==")
    print(f"  velocity drops kept alive as open-play lulls: {n_lulls}")
    print("  dead-span sources:", spans["reason"].value_counts().to_dict())
    shift_s = (spans["start_frame"] - spans["raw_start"]) / VIDEO_FPS
    shift_e = (spans["end_frame"] - spans["raw_end"]) / VIDEO_FPS
    print(f"  refinement moved starts by median {shift_s.median():+.1f} s "
          f"(IQR {shift_s.quantile(.25):+.1f}..{shift_s.quantile(.75):+.1f}), "
          f"ends by median {shift_e.median():+.1f} s "
          f"(IQR {shift_e.quantile(.25):+.1f}..{shift_e.quantile(.75):+.1f})")
    durs = np.array([b - a for a, b in in_scope_spans]) / VIDEO_FPS
    print(f"  after merging: {len(in_scope_spans)} dead spans in scope, "
          f"{(dead & scope).sum() / scope.sum():.0%} of match time, "
          f"median {np.median(durs):.1f} s")

    set_pieces = tags[(tags["code"] == "SET PIECES") & (tags["Type"] != "Kick Off (Start)")]
    span_df = pd.DataFrame(in_scope_spans, columns=["start_frame", "end_frame"])
    m = v04.match_matrix(set_pieces["start_frame"].to_numpy(), span_df)
    print(f"  Smart Tagging check: {m.any(axis=1).mean():.0%} of {len(set_pieces)} tagged set pieces "
          f"start inside a dead span (+-{v04.MATCH_TOL_S} s); "
          f"{m.any(axis=0).mean():.0%} of dead spans hold one (lower bound: fouls are rarely tagged)")

    # ---------- STAGE 2
    e = control_evidence(signals, pos, team)
    n_phase = 0
    if use_tags:
        prior, n_phase = tag_prior(tags, names, n)
        e = e + TAG_WEIGHT * prior * ~ev["tracked"]
    alive = scope & ~dead
    print(f"\n== Stage 2: control evidence (H = {names[0]}, A = {names[1]}) ==")
    print(f"  alive frames with ball evidence |e| > 0.1: {(np.abs(e[alive]) > 0.1).mean():.0%}; "
          f"ball tracked on {ev['tracked'][alive].mean():.0%} of alive frames"
          + (f"; {n_phase} phase tags as a weak prior" if use_tags else "; no tag prior"))

    # ---------- DECODER
    labels, stats = decode(e, dead, restart_flags(dead, spans, n))
    labels[~scope] = "X"          # ungraded frames: ignored by the scorer
    poss =possession_table(labels, scope)
    print(f"\n== Decoded possession (scope only) ==")
    counts = pd.Series(labels[scope]).value_counts(normalize=True)
    print("  label share:", {k: f"{v:.0%}" for k, v in counts.items()})
    print(f"  possessions: {len(poss)} ({(poss['team'] == 'H').sum()} H / {(poss['team'] == 'A').sum()} A), "
          f"median {poss['duration_s'].median():.1f} s, shortest {poss['duration_s'].min():.1f} s, "
          f"{scope.sum() / VIDEO_FPS / 60 / max(len(poss), 1):.2f} min per possession")
    q = np.quantile(poss["duration_s"], [0.1, 0.25, 0.5, 0.75, 0.9])
    print("  duration quantiles 10/25/50/75/90%:", " / ".join(f"{v:.1f}s" for v in q))
    seq = labels[scope]
    direct = sum(1 for i in np.flatnonzero(seq[1:] != seq[:-1]) if {seq[i], seq[i + 1]} == {"H", "A"})
    print(f"  direct H<->A changes: {direct}; label changes per minute: "
          f"{(seq[1:] != seq[:-1]).sum() / (scope.sum() / VIDEO_FPS / 60):.2f}")
    if stats["restart_prior_used"]:
        print(f"  restarts after ball out: {stats['restart_prior_used']}, decoded to the team that did "
              f"not have it before in {stats['restart_prior_followed'] / stats['restart_prior_used']:.0%}"
              f" (restart prior {'off' if RESTART_PRIOR == 0 else RESTART_PRIOR})")

    # ---------- outputs
    sub = write_submission(SUBMISSION_FILE, labels)
    if not official_validation(sub):
        raise SystemExit(f"{sub} FAILED the official validation -- do not submit it")
    spans.to_csv(OUT_DIR / "dead_spans.csv", index=False)
    poss.to_csv(OUT_DIR / "possessions.csv", index=False)
    fig = plot_timeline(e, labels, scope)
    fig.savefig(OUT_DIR / "timeline.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {sub} ({len(labels)} rows) and diagnostics to {OUT_DIR}")
    if show:
        plt.show()
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--show", action="store_true", help="open the figure (it is always saved)")
    parser.add_argument("--no-show", action="store_true", help=argparse.SUPPRESS)  # old flag, now the default
    parser.add_argument("--refresh", action="store_true", help="rebuild the tracking cache")
    parser.add_argument("--no-tags", action="store_true", help="no Smart Tagging prior")
    a = parser.parse_args()
    main(show=a.show and not a.no_show, refresh=a.refresh, use_tags=not a.no_tags)
