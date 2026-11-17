"""Do set pieces coincide with a drop in intensity? Speed of players and ball around each tag.

Run with:
    uv run python Scripts/03_SetPieceIntensity.py            # tables + figures
    uv run python Scripts/03_SetPieceIntensity.py --no-show  # save figures, don't open them

Outputs go to data/derived/set_piece_intensity/ (inside data/, so git ignores them):
    set_piece_events.csv       one row per set piece, every window mean below
    profile.png                average speed curve around the tag, windows shaded
    by_type.png                baseline vs stoppage vs recovery, per set-piece type
    null_test.png              observed dip vs the dip at random open-play moments


HOW IT WORKS
============

1. Intensity per frame (ATD, one row per video frame)
   ---------------------------------------------------
   The ATD comes from the project loader -- loader.load() (kloppy, ~30 s) then .to_df() (~10 s)
   -- and loader.check_alignment() must confirm one record per video frame before anything is
   indexed by frame. Team membership comes from explorer.team_metadata(), metres-per-unit from
   explorer.pitch_dimensions().

   * Player speed = the provider's `s` channel ("<player_id>_s" in to_df). The meta xml labels
     it "kph", but it is m/s: differentiating the positions in metres reproduces it with a
     median ratio of 1.01, and its 99.99th percentile is 8.9 -- sprint speed in m/s, a jog in
     km/h.
   * Per frame we average over tracked players: all players (>= 10 tracked), and each team
     (>= 5 tracked). Frames with too few players are NaN rather than a mean of two people.
   * Ball speed is NOT taken from the data -- kloppy leaves `ball_speed` empty and the raw `s`
     is present on only ~10% of frames. It is
     differentiated from position over +-3 frames (0.24 s); jumps above 40 m/s are tracking
     glitches and are dropped.
   * "Ball tracked" (0/1) is kept as its own signal. During stoppages the ball is often out of
     play / in a player's hands and goes untracked, so ball speed there is only measured on the
     minority of frames where it IS tracked -- read the ball-speed numbers with that in mind.

2. Where the set piece actually is inside the tag
   ----------------------------------------------
   Smart Tagging gives each SET PIECES instance a fixed 35 s window (5 s for kick-offs) starting
   at `start_frame`. Averaging speed around `start_frame` (profile.png) shows:
     - open play runs at ~1.8 m/s per player,
     - speed sinks from about -15 s, bottoms out at ~1.0 m/s from 0 to +5 s,
     - and rebounds -- overshooting to ~2.4 m/s -- from about +8 s.
   So `start_frame` marks the stoppage / set-up, and most of the 35 s tag is the play that
   FOLLOWS the restart. Using the whole tag as "the set piece" would mix the dip with the
   rebound and hide it. Everything below is therefore anchored on `start_frame`:

       BASELINE   [-60 s, -20 s)   open play before the stoppage
       STOPPAGE   [-10 s,  +5 s)   ball dead, set-up, the restart itself
       RECOVERY   [+10 s, +35 s)   the play after the restart (≈ rest of the tag)

   The windows were chosen by looking at that pooled curve, i.e. from the same data that is then
   tested. The null test in step 5 uses the SAME windows on random anchors, so it is fair in
   shape, but the significance should still be read as slightly optimistic.

3. Keeping the baseline honest
   ---------------------------
   * Only frames IN PLAY count: from each half's "Kick Off (Start)" to the last tag of that
     half. Pre-match and half-time are excluded (so the two "Kick Off (Start)" events get no
     baseline and drop out of paired stats).
   * Baseline frames that fall inside ANOTHER stoppage are excluded: the [-20 s, +10 s) span
     around every set piece, and the whole stretch from a goal to the kick-off that follows it
     (celebrations are not open play).
   * A window mean needs >= 50% of its frames valid (10% for ball speed, which is sparse).

4. Per-event (paired) comparison
   ------------------------------
   For each set piece and each signal -- all players, taking team, defending team, ball speed,
   ball tracked -- we compute the mean in each window. Taking/defending come from the tag's
   `Team`; throw-ins and kick-offs are mostly "N/A", so the team split covers fewer events.
   Each event is its own control, so we look at  d = stoppage - baseline:
     * mean change and % change,
     * share of events with a dip (d < 0),
     * Cohen's dz = mean(d) / sd(d)   (|dz| ~0.2 small, ~0.5 medium, >= 0.8 large),
     * one-sided sign-flip permutation p-value: if set pieces had no effect, the sign of each d
       would be a coin flip; we flip signs at random 10 000 times and see how often the mean
       is as negative as observed. No scipy needed, no normality assumed.

5. Null test against random moments of open play
   ----------------------------------------------
   Intensity fluctuates all the time, so a dip after ANY moment is possible. We take every
   second of open play whose [-20 s, +10 s) span touches no tagged stoppage, compute the exact
   same stoppage-minus-baseline difference there, and bootstrap the mean over as many anchors
   as there are set pieces. p = share of random means at least as negative as the observed one.
   Untagged stoppages (fouls without a free-kick tag, injuries) remain in this null pool, which
   makes the test conservative.

6. Correlation across the whole match
   -----------------------------------
   The match is cut into 1-second bins (in play only). For each bin we know the mean speed and
   whether it is inside a set-piece stoppage window. The Pearson correlation between that 0/1
   indicator and speed is the point-biserial r -- "how much does being in a set piece explain
   intensity". It is computed for (a) the stoppage window and (b) the raw 35 s tag, to show the
   difference anchoring makes. Bins are autocorrelated, so no p-value is quoted for r; steps 4
   and 5 carry the inference.
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from morethanahack.helpers.config import DATA_ROOT, DEFAULT_SOURCE
from morethanahack.helpers.explorer import pitch_dimensions, team_metadata
from morethanahack.helpers.loader import (VIDEO_FPS, check_alignment, load, load_smart_tagging,
                                          print_report, video_frame_of)

# ---------------------------------------------------------------- params

BASELINE_S = (-60, -20)          # windows in seconds relative to the tag's start_frame
STOPPAGE_S = (-10, 5)
RECOVERY_S = (10, 35)
EXCLUDE_S = (-20, 10)            # span around every set piece that is never "open play"
PROFILE_S = (-60, 40)            # range of the averaged curve

MIN_PLAYERS_ALL = 10             # tracked players needed for a frame's all-player mean
MIN_PLAYERS_TEAM = 5             # ... and for a team mean
PLAYER_MAX_SPEED = 12.0          # m/s, above this is a tracking error
BALL_DIFF_FRAMES = 3             # ball speed = displacement over +-3 frames
BALL_MAX_SPEED = 40.0            # m/s (144 km/h); faster is a tracking jump
MIN_COVERAGE = {"ball_speed": 0.1}   # share of valid frames a window needs; default below
DEFAULT_COVERAGE = 0.5

N_PERM = 10_000
SEED = 7
OUT_DIR = DATA_ROOT / "derived" / "set_piece_intensity"

SIGNALS = {                      # column -> (label, unit)
    "all": ("all players", "m/s"),
    "taking": ("taking team", "m/s"),
    "defending": ("defending team", "m/s"),
    "ball_speed": ("ball speed (when tracked)", "m/s"),
    "ball_tracked": ("ball tracked", "share"),
}

INK, MUTED = "#333333", "#888888"
ROLE_COLOURS = {"all": INK, "taking": "#2A9D8F", "defending": "#8E6CC2"}
PHASE_COLOURS = {"baseline": "#B7BDC6", "stoppage": "#D9734E", "recovery": "#4C82B5"}


def sec(seconds):
    """Seconds -> frames."""
    return int(round(seconds * VIDEO_FPS))


def fmt_p(p):
    """A permutation p-value can't go below 1/(N_PERM+1), so report that floor as '<'."""
    return f"p < {1 / N_PERM:.0e}".replace("e-0", "e-") if p <= 1 / N_PERM + 1e-12 else f"p = {p:.4f}"


# ---------------------------------------------------------------- 1. tracking -> per-frame signals

def load_tracking(source=DEFAULT_SOURCE):
    """ATD via the project loader -> speed (n, players) in m/s, player teams, ball x/y in metres.

    Row i of every array is video frame i; check_alignment() guarantees that is possible.
    """
    dataset = load(source)
    report = check_alignment(dataset, source)
    print_report(report)
    if not report["agrees"]:
        raise ValueError("ATD is not one record per video frame -- cannot index by frame")

    frames = dataset.to_df()
    frames.index = [video_frame_of(f) for f in frames["frame_id"]]
    frames = frames.sort_index()

    teams = team_metadata(dataset)
    player_ids = [p["player_id"] for t in teams for p in t["players"]]
    speed = frames[[f"{pid}_s" for pid in player_ids]].to_numpy(dtype=np.float64, copy=True)
    speed[speed > PLAYER_MAX_SPEED] = np.nan

    mx, my = pitch_dimensions(dataset)["metres_per_unit"]
    return {
        "speed": speed,
        "team": np.array([t["name"] for t in teams for _ in t["players"]], dtype=object),
        "home": next(t["name"] for t in teams if t["ground"] == "home"),
        "away": next(t["name"] for t in teams if t["ground"] == "away"),
        "ball_x": frames["ball_x"].to_numpy(dtype=np.float64) * mx,
        "ball_y": frames["ball_y"].to_numpy(dtype=np.float64) * my,
    }


def _mean_of_tracked(speed, min_n):
    """Row-wise mean over non-NaN columns; NaN where fewer than min_n are tracked."""
    n = (~np.isnan(speed)).sum(axis=1)
    total = np.nansum(speed, axis=1)
    return np.where(n >= min_n, total / np.maximum(n, 1), np.nan)


def frame_signals(trk):
    """DataFrame indexed by video frame: all / <team name> / ball_speed / ball_tracked."""
    out = {"all": _mean_of_tracked(trk["speed"], MIN_PLAYERS_ALL)}
    for team in np.unique(trk["team"]):
        out[team] = _mean_of_tracked(trk["speed"][:, trk["team"] == team], MIN_PLAYERS_TEAM)

    bx, by, k = trk["ball_x"], trk["ball_y"], BALL_DIFF_FRAMES
    ball_speed = np.full(len(bx), np.nan)
    ball_speed[k:-k] = np.hypot(bx[2 * k:] - bx[:-2 * k], by[2 * k:] - by[:-2 * k]) / (2 * k / VIDEO_FPS)
    ball_speed[ball_speed > BALL_MAX_SPEED] = np.nan
    out["ball_speed"] = ball_speed
    out["ball_tracked"] = (~np.isnan(bx)).astype(float)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 2./3. tags, masks, windows

def in_play_mask(tags, n):
    """True from each half's kick-off to the end of that half's last tag."""
    mask = np.zeros(n, bool)
    for _, half in tags.groupby("Half"):
        kick_off = half.loc[half["Type"] == "Kick Off (Start)", "start_frame"]
        first = kick_off.min() if len(kick_off) else half["start_frame"].min()
        mask[first:half["end_frame"].max() + 1] = True
    return mask


def stoppage_mask(set_pieces, goals, n):
    """True wherever play is (probably) stopped: around every set piece, and goal -> kick-off."""
    mask = np.zeros(n, bool)
    for f in set_pieces["start_frame"]:
        mask[max(0, f + sec(EXCLUDE_S[0])):max(0, f + sec(EXCLUDE_S[1]))] = True
    kick_offs = set_pieces.loc[set_pieces["Type"] == "Kick Off (After Goal)", "start_frame"].values
    for g in goals["start_frame"]:
        later = kick_offs[kick_offs > g]
        if len(later):
            mask[g:later.min() + sec(EXCLUDE_S[1])] = True
    return mask


class WindowMean:
    """O(1) mean of a signal over [anchor+lo, anchor+hi) for many anchors, via cumulative sums."""

    def __init__(self, values, valid, coverage):
        ok = valid & ~np.isnan(values)
        self.total = np.concatenate([[0.0], np.cumsum(np.where(ok, values, 0.0))])
        self.count = np.concatenate([[0], np.cumsum(ok)])
        self.coverage = coverage

    def __call__(self, anchors, window_s):
        anchors = np.asarray(anchors)
        lo, hi = anchors + sec(window_s[0]), anchors + sec(window_s[1])
        n = len(self.count) - 1
        inside = (lo >= 0) & (hi <= n)
        lo_c, hi_c = np.clip(lo, 0, n), np.clip(hi, 0, n)
        count = self.count[hi_c] - self.count[lo_c]
        total = self.total[hi_c] - self.total[lo_c]
        enough = inside & (count >= self.coverage * (hi - lo)) & (count > 0)
        return np.where(enough, total / np.maximum(count, 1), np.nan)


def window_means(signals, in_play, open_play):
    """{signal: (baseline WindowMean, stoppage/recovery WindowMean)}."""
    out = {}
    for col in signals.columns:
        cov = MIN_COVERAGE.get(col, DEFAULT_COVERAGE)
        values = signals[col].to_numpy()
        out[col] = (WindowMean(values, open_play, cov), WindowMean(values, in_play, cov))
    return out


# ---------------------------------------------------------------- 4. per-event table + paired stats

def event_table(set_pieces, means, home, away):
    """One row per set piece with baseline / stoppage / recovery means of every signal."""
    rows = set_pieces[["id", "Type", "Team", "Half", "start_frame", "end_frame"]].reset_index(drop=True)
    anchors = rows["start_frame"].to_numpy()

    def phases(col):
        base_wm, play_wm = means[col]
        return base_wm(anchors, BASELINE_S), play_wm(anchors, STOPPAGE_S), play_wm(anchors, RECOVERY_S)

    per_signal = {col: phases(col) for col in ("all", "ball_speed", "ball_tracked")}
    team_phases = {team: phases(team) for team in (home, away)}
    team = rows["Team"].to_numpy()
    for role, pick in (("taking", lambda t: t), ("defending", lambda t: away if t == home else home)):
        stacked = []
        for i in range(3):
            stacked.append(np.array([team_phases[pick(t)][i][j] if t in (home, away) else np.nan
                                     for j, t in enumerate(team)]))
        per_signal[role] = tuple(stacked)

    for col in SIGNALS:
        base, stop, rec = per_signal[col]
        rows[f"{col}_baseline"], rows[f"{col}_stoppage"], rows[f"{col}_recovery"] = base, stop, rec
        rows[f"{col}_dip_pct"] = 100 * (stop - base) / base
    return rows


def paired_summary(base, stop, rng):
    """Paired stoppage-vs-baseline statistics; one-sided sign-flip permutation p for a dip."""
    ok = ~np.isnan(base) & ~np.isnan(stop)
    b, s = base[ok], stop[ok]
    d = s - b
    if len(d) < 3:
        return {"n": len(d)}
    flips = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    null = (flips * d).mean(axis=1)
    return {
        "n": len(d),
        "baseline": b.mean(),
        "stoppage": s.mean(),
        "change": d.mean(),
        "change_pct": 100 * d.mean() / b.mean(),
        "share_dipped": (d < 0).mean(),
        "cohens_dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
        "p_sign_flip": (np.sum(null <= d.mean()) + 1) / (N_PERM + 1),
    }


# ---------------------------------------------------------------- 5. null test

def null_anchors(in_play, blocked):
    """Every second of open play whose [-20 s, +10 s) span touches no stoppage and is in play."""
    n = len(in_play)
    bad = np.concatenate([[0], np.cumsum(blocked | ~in_play)])
    anchors = np.arange(sec(-BASELINE_S[0]), n - sec(RECOVERY_S[1]), sec(1))
    lo, hi = anchors + sec(EXCLUDE_S[0]), anchors + sec(EXCLUDE_S[1])
    return anchors[(bad[hi] - bad[lo]) == 0]


def null_test(means_for_signal, anchors, n_events, observed, rng):
    """Bootstrap the mean (stoppage - baseline) over n_events random open-play anchors."""
    base_wm, play_wm = means_for_signal
    d = play_wm(anchors, STOPPAGE_S) - base_wm(anchors, BASELINE_S)
    d = d[~np.isnan(d)]
    boot = rng.choice(d, size=(N_PERM, n_events)).mean(axis=1)
    return {
        "null_anchors": len(d),
        "null_mean": boot.mean(),
        "null_sd": boot.std(),
        "z": (observed - boot.mean()) / boot.std(),
        "p_vs_random": (np.sum(boot <= observed) + 1) / (N_PERM + 1),
        "boot": boot,
    }


# ---------------------------------------------------------------- 6. match-wide correlation

def binned_correlation(values, in_play, indicator, bin_frames=sec(1)):
    """Point-biserial r between a 0/1 indicator and a signal, on 1-second bins in play."""
    n = len(values) // bin_frames * bin_frames
    v = values[:n].reshape(-1, bin_frames)
    play = in_play[:n].reshape(-1, bin_frames).all(axis=1)
    flag = indicator[:n].reshape(-1, bin_frames).mean(axis=1) >= 0.5
    with np.errstate(invalid="ignore"):
        counts = (~np.isnan(v)).sum(axis=1)
        mean = np.where(counts > 0, np.nansum(v, axis=1) / np.maximum(counts, 1), np.nan)
    keep = play & ~np.isnan(mean)
    r = np.corrcoef(flag[keep].astype(float), mean[keep])[0, 1]
    return {"r": r, "inside": mean[keep & flag].mean(), "outside": mean[keep & ~flag].mean(),
            "share_inside": flag[keep].mean()}


def window_indicator(set_pieces, n, window_s=None):
    """0/1 per frame: inside any set piece's window (relative to start) or its full tag."""
    flag = np.zeros(n, bool)
    for _, e in set_pieces.iterrows():
        if window_s is None:
            lo, hi = e["start_frame"], e["end_frame"] + 1
        else:
            lo, hi = e["start_frame"] + sec(window_s[0]), e["start_frame"] + sec(window_s[1])
        flag[max(0, lo):max(0, min(n, hi))] = True
    return flag


# ---------------------------------------------------------------- profiles + figures

def event_profiles(anchors, values, in_play):
    """(events, offsets) matrix of a signal around each anchor; NaN out of range / out of play."""
    offsets = np.arange(sec(PROFILE_S[0]), sec(PROFILE_S[1]))
    idx = np.asarray(anchors)[:, None] + offsets[None, :]
    ok = (idx >= 0) & (idx < len(values))
    idx_c = np.clip(idx, 0, len(values) - 1)
    out = np.where(ok & in_play[idx_c], values[idx_c], np.nan)
    return offsets / VIDEO_FPS, out


def _mean_band(matrix):
    """Mean and 95% band across events, smoothed over 1 s for display."""
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns at the edges
        n = (~np.isnan(matrix)).sum(axis=0)
        mean = np.nanmean(matrix, axis=0)
        se = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    smooth = lambda a: pd.Series(a).rolling(sec(1), center=True, min_periods=1).mean().to_numpy()
    return smooth(mean), smooth(mean - 1.96 * se), smooth(mean + 1.96 * se)


def _shade_windows(ax, label=False):
    for (lo, hi), name in ((BASELINE_S, "baseline"), (STOPPAGE_S, "stoppage"), (RECOVERY_S, "recovery")):
        ax.axvspan(lo, hi, color=PHASE_COLOURS[name], alpha=0.14 if name != "stoppage" else 0.22,
                   linewidth=0, zorder=0)
        if label:
            ax.text((lo + hi) / 2, 1.02, name, transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=9, color=MUTED)
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.axvline(RECOVERY_S[1], color=MUTED, linestyle=":", linewidth=1)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_profile(events, signals, in_play, home, away):
    anchors = events["start_frame"].to_numpy()
    team = events["Team"].to_numpy()
    tagged = np.isin(team, [home, away])

    t, all_players = event_profiles(anchors, signals["all"].to_numpy(), in_play)
    home_p = event_profiles(anchors, signals[home].to_numpy(), in_play)[1]
    away_p = event_profiles(anchors, signals[away].to_numpy(), in_play)[1]
    taking = np.where((team == home)[:, None], home_p, away_p)[tagged]
    defending = np.where((team == home)[:, None], away_p, home_p)[tagged]
    ball_speed = event_profiles(anchors, signals["ball_speed"].to_numpy(), in_play)[1]
    ball_tracked = event_profiles(anchors, signals["ball_tracked"].to_numpy(), in_play)[1]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1, 1]})
    ax = axes[0]
    for name, matrix, n in (("all", all_players, len(anchors)),
                            ("taking", taking, tagged.sum()),
                            ("defending", defending, tagged.sum())):
        mean, lo, hi = _mean_band(matrix)
        ax.fill_between(t, lo, hi, color=ROLE_COLOURS[name], alpha=0.12, linewidth=0)
        ax.plot(t, mean, color=ROLE_COLOURS[name], linewidth=2,
                label=f"{SIGNALS[name][0]} (n={n})")
    ax.set_ylabel("mean player speed (m/s)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    _shade_windows(ax, label=True)

    for ax, matrix, ylabel, scale in ((axes[1], ball_speed, "ball speed (m/s)\nwhen tracked", 1),
                                      (axes[2], ball_tracked, "ball tracked (%)", 100)):
        mean, lo, hi = _mean_band(matrix)
        ax.fill_between(t, lo * scale, hi * scale, color=INK, alpha=0.10, linewidth=0)
        ax.plot(t, mean * scale, color=INK, linewidth=2)
        ax.set_ylabel(ylabel)
        _shade_windows(ax)

    for ax in axes:
        _style(ax)
    axes[-1].set_xlabel("seconds from Smart Tagging set-piece start  "
                        "(dashed = tag start, dotted = typical tag end)")
    fig.suptitle(f"Intensity around {len(anchors)} tagged set pieces - mean and 95% band",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    return fig


def plot_by_type(events):
    phases = ("baseline", "stoppage", "recovery")
    grouped = events.groupby("Type")
    types = [t for t, g in sorted(grouped, key=lambda kv: -len(kv[1]))]
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.26
    for i, phase in enumerate(phases):
        vals = [grouped.get_group(t)[f"all_{phase}"].mean() for t in types]
        ax.bar(np.arange(len(types)) + (i - 1) * width, vals, width * 0.92,
               color=PHASE_COLOURS[phase], label=phase)
    for j, ty in enumerate(types):
        dip = grouped.get_group(ty)["all_dip_pct"].dropna()
        stop = grouped.get_group(ty)["all_stoppage"].mean()
        if len(dip) and not np.isnan(stop):
            ax.text(j, stop + 0.05, f"{dip.median():+.0f}%", ha="center", va="bottom",
                    fontsize=9, color=INK)
    labels = [f"{t}\n(n={grouped.get_group(t)['all_dip_pct'].notna().sum()}"
              f"/{len(grouped.get_group(t))})" for t in types]
    ax.set_xticks(np.arange(len(types)), labels=labels, fontsize=9)
    ax.set_ylabel("mean player speed, all players (m/s)")
    ax.set_title("By set-piece type - label = median per-event change, stoppage vs baseline\n"
                 "n = events with a valid baseline / events tagged", loc="left", fontsize=11)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)  # headroom for the legend
    ax.legend(frameon=False, ncol=3, loc="upper left")
    _style(ax)
    fig.tight_layout()
    return fig


def plot_null(null, observed, n_events):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(null["boot"], bins=60, color="#B7BDC6", edgecolor="white", linewidth=0.5)
    ax.axvline(observed, color=PHASE_COLOURS["stoppage"], linewidth=2)
    ax.text(observed, ax.get_ylim()[1] * 0.95, f"  set pieces: {observed:+.2f} m/s\n"
            f"  {fmt_p(null['p_vs_random'])}", color=INK, va="top", fontsize=9)
    ax.set_xlabel(f"mean change in player speed, stoppage - baseline (m/s), over {n_events} anchors")
    ax.set_ylabel("bootstrap draws")
    ax.set_title(f"Same windows at random open-play moments ({null['null_anchors']} anchors, "
                 f"{N_PERM} draws)", loc="left", fontsize=11)
    _style(ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- main

def main(show=True):
    rng = np.random.default_rng(SEED)

    print("loading tracking (kloppy, ~40 s) ...")
    trk = load_tracking()
    signals = frame_signals(trk)
    n = len(signals)
    home, away = trk["home"], trk["away"]

    tags = load_smart_tagging(DEFAULT_SOURCE)
    set_pieces = tags[tags["code"] == "SET PIECES"].reset_index(drop=True)
    goals = tags[tags["code"] == "GOALS"]

    in_play = in_play_mask(tags, n)
    blocked = stoppage_mask(set_pieces, goals, n)
    open_play = in_play & ~blocked
    means = window_means(signals, in_play, open_play)

    print(f"{n} frames, {in_play.mean():.0%} in play, {open_play.mean():.0%} open play; "
          f"{len(set_pieces)} set pieces; home {home!r}, away {away!r}")
    print(f"windows (s from tag start): baseline {BASELINE_S}, stoppage {STOPPAGE_S}, "
          f"recovery {RECOVERY_S}")

    events = event_table(set_pieces, means, home, away)

    # 4. paired stats per signal
    summary = pd.DataFrame({SIGNALS[col][0]: paired_summary(events[f"{col}_baseline"].to_numpy(),
                                                            events[f"{col}_stoppage"].to_numpy(), rng)
                            for col in SIGNALS}).T
    print("\n== Stoppage vs baseline, per event (paired) ==")
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))

    rec = events[["all_baseline", "all_stoppage", "all_recovery"]].dropna().mean()
    print(f"\nall players, events with all three windows: baseline {rec.iloc[0]:.2f} -> "
          f"stoppage {rec.iloc[1]:.2f} -> recovery {rec.iloc[2]:.2f} m/s")

    by_type = events.groupby("Type").agg(
        n=("id", "size"),
        baseline=("all_baseline", "mean"),
        stoppage=("all_stoppage", "mean"),
        recovery=("all_recovery", "mean"),
        median_dip_pct=("all_dip_pct", "median"),
        share_dipped=("all_dip_pct", lambda s: (s.dropna() < 0).mean()),
        ball_tracked_base=("ball_tracked_baseline", "mean"),
        ball_tracked_stop=("ball_tracked_stoppage", "mean"),
    ).sort_values("n", ascending=False)
    print("\n== By type, all players (m/s) ==")
    print(by_type.to_string(float_format=lambda v: f"{v:.2f}"))

    # 5. null test on all-player speed
    observed = summary.loc[SIGNALS["all"][0], "change"]
    n_events = int(summary.loc[SIGNALS["all"][0], "n"])
    null = null_test(means["all"], null_anchors(in_play, blocked), n_events, observed, rng)
    print("\n== Null test: same windows at random open-play moments (all players) ==")
    print(f"observed mean change {observed:+.3f} m/s over {n_events} set pieces; random: "
          f"{null['null_mean']:+.3f} +/- {null['null_sd']:.3f} ({null['null_anchors']} anchors) "
          f"-> z = {null['z']:.1f}, {fmt_p(null['p_vs_random'])}")

    # 6. match-wide correlation
    print("\n== Match-wide point-biserial correlation, 1-second bins in play ==")
    for label, flag in (("stoppage window", window_indicator(set_pieces, n, STOPPAGE_S)),
                        ("full 35 s tag", window_indicator(set_pieces, n))):
        for col in ("all", "ball_tracked"):
            res = binned_correlation(signals[col].to_numpy(), in_play, flag)
            print(f"  {label:<16} vs {SIGNALS[col][0]:<13} r = {res['r']:+.3f}   "
                  f"inside {res['inside']:.2f} / outside {res['outside']:.2f}   "
                  f"({res['share_inside']:.1%} of bins inside)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events.to_csv(OUT_DIR / "set_piece_events.csv", index=False)
    figs = {"profile": plot_profile(events, signals, in_play, home, away),
            "by_type": plot_by_type(events),
            "null_test": plot_null(null, observed, n_events)}
    for name, fig in figs.items():
        fig.savefig(OUT_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {OUT_DIR}")

    if show:
        plt.show()
    return events, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-show", action="store_true", help="save figures without opening them")
    main(show=not parser.parse_args().no_show)
