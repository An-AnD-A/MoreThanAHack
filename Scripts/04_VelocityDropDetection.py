"""Find drops in player/ball velocity from the ATD alone, then check them against Smart Tagging.

Script 03 starts from the tags and looks at speed around them. This one goes the other way:
it detects stoppages BLIND from the tracking data (no tags involved), and only afterwards asks
whether the detected drops line up with the tagged SET PIECES.

Run with:
    uv run python Scripts/04_VelocityDropDetection.py            # tables + figures
    uv run python Scripts/04_VelocityDropDetection.py --no-show  # save figures, don't open them

Outputs go to data/derived/velocity_drops/ (inside data/, so git ignores them):
    drops.csv          one row per detected drop, ATD features + what it matched in the tags
    set_pieces.csv     one row per tagged set piece, whether / how a drop caught it
    timeline.png       smoothed speed over the match, drops shaded, tag starts marked
    timing.png         where the tag start falls inside the matched drop
    sweep.png          recall / precision vs detection threshold, against chance


HOW IT WORKS
============

Part A uses the ATD only. Part B is the only place Smart Tagging is read.

A1. Per-frame signals (ATD)
    * Player speed = the provider's per-player `s` channel (m/s; see 03 for why it is m/s
      despite the "kph" label), averaged over tracked players; NaN with < 10 tracked.
    * Ball speed is differentiated from position over +-3 frames; > 40 m/s is a glitch.
    * Ball tracked (0/1) and the number of tracked players are kept as signals too.

A2. Drop detection (ATD)
    * Smooth player speed over 3 s, and compare it with the LOCAL tempo: a 120 s rolling
      median. ratio = smoothed / local. Using a ratio rather than a fixed m/s cut keeps the
      detector fair across slow and frantic spells, and as legs tire late in the match.
    * Hysteresis: a drop starts when ratio < 0.6 and is extended while ratio < 0.8, so
      one stoppage isn't chopped into pieces by a single jog. Gaps < 2 s are merged and
      drops < 4 s are discarded.
    * Each drop is described with ATD-only features: duration, depth, ball-tracked share,
      and where the ball was at the restart.
        - DEAD-BALL flag: the ball is tracked < 50% of the drop. The tracker loses the ball
          when it is out of play or in someone's hands; in open-play lulls it stays tracked.
        - RESTART ZONE from the ball position just before play resumes: centre spot ->
          kick-off, near a corner flag -> corner, in/near the 6-yard box -> goal kick, on a
          touchline -> throw-in, anywhere else -> free kick. Two of these rules were widened
          after looking at mis-labelled restarts (touchline band 2.5 -> 4 m; ball last seen
          beyond the goal line -> corner/goal kick), so the type-agreement figure is mildly
          optimistic. The detection itself was not tuned on the tags.
        - BREAK: drops > 90 s or with < 10 players tracked on average (pre-match, half-time,
          broadcast-style tracking gaps). They are reported but kept out of the scoring.

B. Validation against Smart Tagging
    * Scope: frames from each half's first tag to its last tag (pre-match / half-time out).
    * A tagged set piece is CAUGHT if its start_frame lies inside a drop, +-3 s. 03 showed the
      tag start marks the stoppage itself, so it should fall inside the drop, not near it.
    * Recall    = share of tagged set pieces caught (overall and per type).
      Precision = share of drops that contain a set-piece tag. This is a LOWER bound: Smart
      Tagging only tags some restarts (e.g. very few free kicks), so a foul, injury or
      substitution produces a genuine drop that counts as a false positive here. Drops that
      contain a goal are listed separately (the celebration is a real stoppage too).
    * Chance: the drops cover a sizeable share of the match, so some tags would land in one
      by luck. We rotate all tag times together by a random offset (keeping their spacing),
      2 000 times, and recompute recall/precision. p = share of rotations doing at least as
      well as the real alignment.
    * Timing: where the tag start sits relative to the drop's start and end.
    * Type agreement: for caught set pieces, does the ATD restart zone match the tag's Type?
    * Sweep: the whole evaluation repeated for a range of thresholds, to show the result is
      not an artefact of the one threshold chosen.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from morethanahack.helpers.config import DATA_ROOT, DEFAULT_SOURCE
from morethanahack.helpers.explorer import pitch_dimensions, team_metadata
from morethanahack.helpers.loader import (VIDEO_FPS, check_alignment, load, load_smart_tagging,
                                          print_report, video_frame_of)

# ---------------------------------------------------------------- params

MIN_PLAYERS = 10                 # tracked players needed for a frame's mean speed
PLAYER_MAX_SPEED = 12.0          # m/s, above this is a tracking error
BALL_DIFF_FRAMES = 3             # ball speed = displacement over +-3 frames
BALL_MAX_SPEED = 40.0            # m/s

SMOOTH_S = 3                     # smoothing of player speed
LOCAL_S = 120                    # window of the local-tempo rolling median
ENTER, EXIT = 0.6, 0.8           # ratio thresholds (hysteresis)
MERGE_GAP_S = 2
MIN_DROP_S = 4
DEAD_BALL_TRACKED = 0.5          # ball tracked < this share of the drop -> dead-ball flag
BREAK_S = 90                     # longer than this -> break, not a stoppage
BREAK_PLAYERS = 10               # fewer tracked players on average -> tracking gap

RESTART_LOOKBACK_S = 8           # restart position: ball in the last 8 s of the drop ...
RESTART_STILL = 2.0              # ... preferring frames where it is (nearly) still, m/s

MATCH_TOL_S = 3                  # tag start must be inside the drop, +- this
N_SHIFTS = 2000
SWEEP = np.round(np.arange(0.30, 0.91, 0.05), 2)
SEED = 7
OUT_DIR = DATA_ROOT / "derived" / "velocity_drops"

TAG_TO_ZONE = {"Throw-in": "throw-in", "Corner Kick": "corner", "Goal Kick": "goal kick",
               "Free Kick": "free kick", "Kick Off (After Goal)": "kick-off",
               "Kick Off (Start)": "kick-off"}

INK, MUTED, GRID = "#333333", "#888888", "#E5E5E5"
DROP_COLOUR, WEAK_COLOUR, TAG_COLOUR = "#D9734E", "#F0C2AE", "#2A9D8F"


def sec(seconds):
    return int(round(seconds * VIDEO_FPS))


def clock(frame):
    s = int(frame / VIDEO_FPS)
    return f"{s // 60:02d}:{s % 60:02d}"


def runs(mask):
    """(starts, ends) of the True runs in a boolean array; ends are exclusive."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


# ================================================================ A. ATD only

def load_frames(source=DEFAULT_SOURCE):
    """(kloppy dataset, frame table indexed by video frame). The slow part, shared with 05."""
    dataset = load(source)
    report = check_alignment(dataset, source)
    print_report(report)
    if not report["agrees"]:
        raise ValueError("ATD is not one record per video frame -- cannot index by frame")

    frames = dataset.to_df()
    frames.index = [video_frame_of(f) for f in frames["frame_id"]]
    return dataset, frames.sort_index()


def load_signals(source=DEFAULT_SOURCE):
    """Per-frame signals, row i = video frame i. Also returns pitch length/width in metres."""
    return signals_from_frames(*load_frames(source))


def signals_from_frames(dataset, frames):
    """Per-frame signals from an already-loaded frame table (see load_frames)."""
    player_ids = [p["player_id"] for t in team_metadata(dataset) for p in t["players"]]
    speed = frames[[f"{pid}_s" for pid in player_ids]].to_numpy(dtype=np.float64, copy=True)
    speed[speed > PLAYER_MAX_SPEED] = np.nan
    n_tracked = (~np.isnan(speed)).sum(axis=1)
    mean_speed = np.where(n_tracked >= MIN_PLAYERS,
                          np.nansum(speed, axis=1) / np.maximum(n_tracked, 1), np.nan)

    pitch = pitch_dimensions(dataset)
    mx, my = pitch["metres_per_unit"]
    bx = frames["ball_x"].to_numpy(dtype=np.float64) * mx
    by = frames["ball_y"].to_numpy(dtype=np.float64) * my
    k = BALL_DIFF_FRAMES
    ball_speed = np.full(len(bx), np.nan)
    ball_speed[k:-k] = np.hypot(bx[2 * k:] - bx[:-2 * k], by[2 * k:] - by[:-2 * k]) / (2 * k / VIDEO_FPS)
    ball_speed[ball_speed > BALL_MAX_SPEED] = np.nan

    signals = pd.DataFrame({"speed": mean_speed, "n_tracked": n_tracked, "ball_x": bx, "ball_y": by,
                            "ball_speed": ball_speed, "ball_tracked": (~np.isnan(bx)).astype(float)})
    return signals, (pitch["pitch_length_m"], pitch["pitch_width_m"])


def speed_ratio(signals):
    """3 s smoothed player speed, the 120 s local median, and their ratio."""
    s = signals["speed"]
    smooth = s.rolling(sec(SMOOTH_S), center=True, min_periods=sec(1)).mean()
    local = s.rolling(sec(LOCAL_S), center=True, min_periods=sec(30)).median()
    return smooth.to_numpy(), local.to_numpy(), (smooth / local).to_numpy()


def detect_drops(ratio, enter=ENTER, exit_=EXIT):
    """[(start, end)] frame spans (end exclusive) where the speed ratio sinks, with hysteresis.

    Frames with no ratio (too few players tracked) count as low: a tracking blackout is
    itself a sign of a stoppage, and BREAK labelling deals with the long ones.
    """
    r = np.nan_to_num(ratio, nan=0.0)
    core = r < enter
    starts, ends = runs(r < exit_)
    spans = [(a, b) for a, b in zip(starts, ends) if core[a:b].any()]
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] < sec(MERGE_GAP_S):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if b - a >= sec(MIN_DROP_S)]


def restart_position(signals, a, b):
    """Ball position just before play resumes: the last RESTART_LOOKBACK_S of the drop.

    Prefer frames where the ball is tracked and nearly still (placed for the restart);
    otherwise take the last tracked position in that stretch, then the first just after.
    """
    lo = max(a, b - sec(RESTART_LOOKBACK_S))
    seg = signals.iloc[lo:b]
    tracked = seg[seg["ball_tracked"] == 1]
    still = tracked[tracked["ball_speed"] < RESTART_STILL]
    if len(still):
        return still["ball_x"].median(), still["ball_y"].median()
    if len(tracked):
        return tracked["ball_x"].iloc[-1], tracked["ball_y"].iloc[-1]
    after = signals.iloc[b:b + sec(2)]
    after = after[after["ball_tracked"] == 1]
    if len(after):
        return after["ball_x"].iloc[0], after["ball_y"].iloc[0]
    return np.nan, np.nan


def restart_zone(x, y, length, width):
    """Where on the pitch a restart happens -> the kind of set piece it most likely is."""
    if np.isnan(x):
        return "unknown"
    near_goal_line = x < 6 or x > length - 6
    near_touchline = y < 6 or y > width - 6
    central = abs(y - width / 2) < 12
    if np.hypot(x - length / 2, y - width / 2) < 4:
        return "kick-off"
    if x < 0 or x > length:              # last seen over the goal line, i.e. out of play
        return "goal kick" if central else "corner"
    if near_goal_line and near_touchline:
        return "corner"
    if (x < 8 or x > length - 8) and central:
        return "goal kick"
    if y < 4 or y > width - 4:           # ball in the thrower's hands, a little inside the line
        return "throw-in"
    return "free kick"


def describe_drops(spans, signals, smooth, ratio, pitch):
    """One row per drop with ATD-only features."""
    length, width = pitch
    bt = signals["ball_tracked"].to_numpy()
    nt = signals["n_tracked"].to_numpy()
    rows = []
    for a, b in spans:
        x, y = restart_position(signals, a, b)
        dur = (b - a) / VIDEO_FPS
        with np.errstate(all="ignore"):
            rows.append({
                "start_frame": a, "end_frame": b, "start_clock": clock(a), "end_clock": clock(b),
                "duration_s": dur,
                "min_speed": np.nanmin(smooth[a:b]) if np.isfinite(smooth[a:b]).any() else np.nan,
                "min_ratio": np.nanmin(ratio[a:b]) if np.isfinite(ratio[a:b]).any() else np.nan,
                "ball_tracked": bt[a:b].mean(),
                "players_tracked": nt[a:b].mean(),
                "restart_x": x, "restart_y": y,
                "zone": restart_zone(x, y, length, width),
                "dead_ball": bt[a:b].mean() < DEAD_BALL_TRACKED,
                "is_break": dur > BREAK_S or nt[a:b].mean() < BREAK_PLAYERS,
            })
    return pd.DataFrame(rows)


# ================================================================ B. validation against tags

def tagged_scope(tags, n):
    """True from each half's first tag to its last (validation scope only)."""
    mask = np.zeros(n, bool)
    for _, half in tags.groupby("Half"):
        mask[half["start_frame"].min():half["end_frame"].max() + 1] = True
    return mask


def match_matrix(tag_frames, drops):
    """(tags, drops) bool: tag start inside drop +- tolerance."""
    t = np.asarray(tag_frames)[:, None]
    a = drops["start_frame"].to_numpy()[None, :] - sec(MATCH_TOL_S)
    b = drops["end_frame"].to_numpy()[None, :] + sec(MATCH_TOL_S)
    return (t >= a) & (t <= b)


def scored_drops(drops, scope):
    """Drops that count for precision: mostly inside the tagged scope and not a break."""
    inside = np.array([scope[a:b].mean() > 0.5 for a, b in zip(drops["start_frame"], drops["end_frame"])])
    return drops[inside & ~drops["is_break"].to_numpy()].reset_index(drop=True)


def recall_precision(tag_frames, drops):
    if len(drops) == 0 or len(tag_frames) == 0:
        return np.nan, np.nan
    m = match_matrix(tag_frames, drops)
    return m.any(axis=1).mean(), m.any(axis=0).mean()


def rotations(tag_frames, scope, n_shifts, rng):
    """Tag times rotated together by random offsets inside the scope (spacing preserved)."""
    in_scope = np.flatnonzero(scope)
    pos = np.searchsorted(in_scope, tag_frames)
    offsets = rng.integers(sec(60), len(in_scope) - sec(60), size=n_shifts)
    return in_scope[(pos[None, :] + offsets[:, None]) % len(in_scope)]


def chance(tag_frames, drops, scope, rng, n_shifts=N_SHIFTS):
    """Recall/precision of rotated tags -> null distributions."""
    rp = np.array([recall_precision(t, drops) for t in rotations(tag_frames, scope, n_shifts, rng)])
    return rp[:, 0], rp[:, 1]


def p_upper(null, observed):
    return (np.sum(null >= observed) + 1) / (len(null) + 1)


# ---------------------------------------------------------------- figures

def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_timeline(smooth, local, drops, set_pieces, scope):
    halves = [(a, b) for a, b in zip(*runs(scope))]
    fig, axes = plt.subplots(len(halves), 1, figsize=(15, 3.4 * len(halves)), squeeze=False)
    minutes = np.arange(len(smooth)) / VIDEO_FPS / 60
    for ax, (lo, hi) in zip(axes[:, 0], halves):
        sl = slice(lo, hi)
        ax.plot(minutes[sl], smooth[sl], color=INK, linewidth=0.8, label="player speed, 3 s mean")
        ax.plot(minutes[sl], local[sl], color=MUTED, linewidth=1.2, linestyle="--",
                label=f"local tempo, {LOCAL_S} s median")
        for _, d in drops.iterrows():
            if d["end_frame"] < lo or d["start_frame"] > hi:
                continue
            ax.axvspan(d["start_frame"] / VIDEO_FPS / 60, d["end_frame"] / VIDEO_FPS / 60,
                       color=DROP_COLOUR if d["dead_ball"] else WEAK_COLOUR,
                       alpha=0.55 if d["dead_ball"] else 0.45, linewidth=0, zorder=0)
        sp = set_pieces[(set_pieces["start_frame"] >= lo) & (set_pieces["start_frame"] <= hi)]
        ax.scatter(sp["start_frame"] / VIDEO_FPS / 60, np.full(len(sp), -0.15), marker="|", s=120,
                   color=TAG_COLOUR, linewidths=2, clip_on=False, zorder=5,
                   label="Smart Tagging set-piece start")
        ax.set_xlim(minutes[lo], minutes[hi - 1])
        ax.set_ylim(-0.3, max(3.5, np.nanmax(smooth[sl]) * 1.05))
        ax.set_ylabel("mean player speed (m/s)")
        _style(ax)
    from matplotlib.patches import Patch
    handles, _ = axes[0, 0].get_legend_handles_labels()
    handles += [Patch(color=DROP_COLOUR, alpha=0.55, label="drop, dead ball (ball lost)"),
                Patch(color=WEAK_COLOUR, alpha=0.45, label="drop, ball still tracked")]
    axes[0, 0].legend(handles=handles, loc="upper left", ncol=5, fontsize=8, frameon=False,
                      bbox_to_anchor=(0, 1.16))
    axes[-1, 0].set_xlabel("video time (minutes)")
    fig.suptitle("Velocity drops detected from the ATD alone, with tagged set pieces underneath",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    return fig


def plot_timing(timing):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, col, title in ((axes[0], "tag_after_drop_start_s", "tag start - drop start"),
                           (axes[1], "tag_before_drop_end_s", "drop end - tag start")):
        vals = timing[col].dropna()
        ax.hist(vals, bins=np.arange(-MATCH_TOL_S, vals.max() + 2, 1.0), color=DROP_COLOUR,
                edgecolor="white", linewidth=1)
        ax.axvline(vals.median(), color=INK, linewidth=1.5)
        ax.text(vals.median(), ax.get_ylim()[1] * 0.95, f"  median {vals.median():.1f} s",
                va="top", fontsize=9, color=INK)
        ax.axvline(0, color=MUTED, linestyle=":", linewidth=1)
        ax.set_xlabel(f"{title} (s)")
        ax.set_title(title, loc="left", fontsize=10)
        _style(ax)
    axes[0].set_ylabel("caught set pieces")
    fig.suptitle("Where the tag start falls inside the drop that caught it  (> 0 = inside)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    return fig


def plot_sweep(sweep):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    series = (("recall", INK, "-", "recall"), ("chance_recall", INK, ":", "recall by chance"),
              ("precision", DROP_COLOUR, "-", "precision (lower bound)"),
              ("chance_precision", DROP_COLOUR, ":", "precision by chance"))
    for col, colour, ls, label in series:
        ax.plot(sweep["enter"], sweep[col], color=colour, linestyle=ls, linewidth=2, marker="o",
                markersize=4, label=label)
        ax.text(sweep["enter"].iloc[-1] + 0.01, sweep[col].iloc[-1], label, fontsize=8,
                color=INK, va="center")
    ax.axvline(ENTER, color=MUTED, linestyle="--", linewidth=1)
    ax.text(ENTER, 0.02, f" used ({ENTER})", fontsize=8, color=MUTED)
    ax.set_xlim(sweep["enter"].min() - 0.02, sweep["enter"].max() + 0.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"drop threshold: speed / local tempo  (exit = threshold + {EXIT - ENTER:.1f})")
    ax.set_ylabel("share")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("Alignment with Smart Tagging across thresholds - solid = real, dotted = chance",
                 loc="left", fontsize=11)
    _style(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- main

def main(show=True):
    rng = np.random.default_rng(SEED)

    # ---------- A. ATD only
    print("loading tracking (kloppy, ~40 s) ...")
    signals, pitch = load_signals()
    smooth, local, ratio = speed_ratio(signals)
    spans = detect_drops(ratio)
    drops = describe_drops(spans, signals, smooth, ratio, pitch)
    print(f"\n== A. ATD only: {len(drops)} drops "
          f"(ratio < {ENTER} to enter, < {EXIT} to stay, >= {MIN_DROP_S} s) ==")
    print(f"  breaks / tracking gaps: {drops['is_break'].sum()}, "
          f"dead-ball flagged: {drops['dead_ball'].sum()}")
    print(f"  open-play tempo: median smoothed speed {np.nanmedian(smooth):.2f} m/s; "
          f"in drops: {drops['min_speed'].median():.2f} m/s at the bottom "
          f"({drops['min_ratio'].median():.0%} of local tempo)")
    print("  restart zones:", drops.loc[~drops["is_break"], "zone"].value_counts().to_dict())

    # ---------- B. validation against Smart Tagging
    tags = load_smart_tagging(DEFAULT_SOURCE)
    set_pieces = tags[tags["code"] == "SET PIECES"].reset_index(drop=True)
    goals = tags[tags["code"] == "GOALS"]
    scope = tagged_scope(tags, len(signals))
    scored = scored_drops(drops, scope)
    t = set_pieces["start_frame"].to_numpy()

    m = match_matrix(t, scored)
    set_pieces["caught"] = m.any(axis=1)
    scored["set_piece_tags"] = m.sum(axis=0)
    scored["goal"] = match_matrix(goals["start_frame"].to_numpy(), scored).any(axis=0)
    scored["tag_types"] = [", ".join(set_pieces.loc[m[:, j], "Type"]) for j in range(len(scored))]

    recall, precision = set_pieces["caught"].mean(), scored["set_piece_tags"].gt(0).mean()
    null_r, null_p = chance(t, scored, scope, rng)
    covered = sum(min(b, len(scope)) - a for a, b in zip(scored["start_frame"], scored["end_frame"]))
    print(f"\n== B. Against Smart Tagging ({scope.sum() / VIDEO_FPS / 60:.0f} min tagged, "
          f"{len(set_pieces)} set pieces, {len(scored)} drops in scope, "
          f"drops cover {covered / scope.sum():.0%} of it) ==")
    print(f"  recall    {recall:.2f}  ({set_pieces['caught'].sum()}/{len(set_pieces)} set pieces "
          f"start inside a drop +-{MATCH_TOL_S} s)   chance {null_r.mean():.2f} "
          f"(99th pct {np.quantile(null_r, 0.99):.2f}), p = {p_upper(null_r, recall):.4f}")
    print(f"  precision {precision:.2f}  ({scored['set_piece_tags'].gt(0).sum()}/{len(scored)} drops "
          f"hold a set-piece tag)         chance {null_p.mean():.2f} "
          f"(99th pct {np.quantile(null_p, 0.99):.2f}), p = {p_upper(null_p, precision):.4f}")
    unexplained = scored[(scored["set_piece_tags"] == 0) & ~scored["goal"]]
    print(f"  + {int((scored['goal'] & (scored['set_piece_tags'] == 0)).sum())} more drops hold a goal; "
          f"{len(unexplained)} drops match no tag (untagged fouls/injuries/subs? see drops.csv)")

    print("\n  by dead-ball flag (ATD-only confidence):")
    for flag, g in scored.groupby("dead_ball"):
        mm = match_matrix(t, g)
        print(f"    dead_ball={str(flag):<5}  drops {len(g):3d}  precision "
              f"{mm.any(axis=0).mean():.2f}  recall on its own {mm.any(axis=1).mean():.2f}")

    by_type = set_pieces.groupby("Type")["caught"].agg(["size", "sum", "mean"]) \
        .rename(columns={"size": "tagged", "sum": "caught", "mean": "recall"}) \
        .sort_values("tagged", ascending=False)
    print("\n  recall by set-piece type:")
    print(by_type.to_string(float_format=lambda v: f"{v:.2f}"))

    # timing + type agreement for caught set pieces (nearest drop)
    rows = []
    for i in np.flatnonzero(set_pieces["caught"]):
        j = np.flatnonzero(m[i])[np.argmin(np.abs(scored.loc[m[i], "start_frame"].to_numpy() - t[i]))]
        d = scored.loc[j]
        rows.append({"id": set_pieces.loc[i, "id"], "drop_start_frame": d["start_frame"],
                     "drop_end_frame": d["end_frame"],
                     "restart_x": d["restart_x"], "restart_y": d["restart_y"],
                     "tag_after_drop_start_s": (t[i] - d["start_frame"]) / VIDEO_FPS,
                     "tag_before_drop_end_s": (d["end_frame"] - t[i]) / VIDEO_FPS,
                     "drop_duration_s": d["duration_s"], "zone": d["zone"],
                     "dead_ball": d["dead_ball"]})
    timing = pd.DataFrame(rows)
    set_pieces = set_pieces.merge(timing, on="id", how="left")
    print(f"\n  timing (caught set pieces): tag start is a median "
          f"{timing['tag_after_drop_start_s'].median():.1f} s after the drop starts and "
          f"{timing['tag_before_drop_end_s'].median():.1f} s before it ends; "
          f"{(timing['tag_after_drop_start_s'] >= 0).mean():.0%} after the start, "
          f"{(timing['tag_before_drop_end_s'] >= 0).mean():.0%} before the end")

    caught = set_pieces[set_pieces["caught"]].copy()
    caught["expected_zone"] = caught["Type"].map(TAG_TO_ZONE)
    known = caught[caught["zone"] != "unknown"]
    print(f"\n  restart zone (ATD) vs tag Type, {len(known)} caught set pieces with a ball position: "
          f"agreement {(known['zone'] == known['expected_zone']).mean():.0%}")
    print(pd.crosstab(known["Type"], known["zone"]).to_string())

    # threshold sweep
    sweep_rows = []
    for enter in SWEEP:
        d = scored_drops(describe_drops(detect_drops(ratio, enter, enter + (EXIT - ENTER)),
                                        signals, smooth, ratio, pitch), scope)
        r, p = recall_precision(t, d)
        cr, cp = chance(t, d, scope, rng, n_shifts=300)
        sweep_rows.append({"enter": enter, "drops": len(d), "recall": r, "precision": p,
                           "chance_recall": cr.mean(), "chance_precision": cp.mean()})
    sweep = pd.DataFrame(sweep_rows)
    print("\n  threshold sweep (recall/precision vs chance):")
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    drops_out = drops.merge(scored[["start_frame", "set_piece_tags", "tag_types", "goal"]],
                            on="start_frame", how="left")
    drops_out.to_csv(OUT_DIR / "drops.csv", index=False)
    set_pieces.to_csv(OUT_DIR / "set_pieces.csv", index=False)
    figs = {"timeline": plot_timeline(smooth, local, scored, set_pieces, scope),
            "timing": plot_timing(timing),
            "sweep": plot_sweep(sweep)}
    for name, fig in figs.items():
        fig.savefig(OUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {OUT_DIR}")

    if show:
        plt.show()
    return drops_out, set_pieces, sweep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-show", action="store_true", help="save figures without opening them")
    main(show=not parser.parse_args().no_show)
