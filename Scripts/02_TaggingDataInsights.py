"""Timeline of the Smart Tagging events: which event, which video frames, which team.

One row per event code. Each row has two lanes -- home on top, away underneath -- and
every event is a bar from its start_frame to its end_frame. Events not tagged to a team
("N/A", e.g. kick-offs) are grey and span both lanes. Goals are also drawn as dashed
lines across the whole chart.

Run with:
    uv run python Scripts/02_TaggingDataInsights.py
"""

import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from morethanahack.helpers.config import COLOURS
from morethanahack.helpers.loader import VIDEO_FPS, atd_paths, load_smart_tagging

NEUTRAL = "#9AA0A6"  # events with no team ("N/A")
LANE_H = 0.36        # height of one team lane; a row is 1.0 tall

# Display order: in possession, out of possession, transitions, set pieces/goals, then
# the zone/ball tags. Same order as the colour table in the Smart Tagging xml.
CODE_ORDER = [
    "BUILD UP", "PROGRESSION", "CREATING CHANCES",
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
    "ATTACKING TRANSITION", "DEFENSIVE TRANSITION",
    "SET PIECES", "GOALS",
    "LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX",
]


def home_away_names(source="tactical"):
    """(home, away) team names from the ATD meta xml. kloppy treats the first team as home."""
    meta, _ = atd_paths(source)
    names = [t.findtext("Name") for t in ET.parse(meta).getroot().iter("Team")]
    return names[0], names[1]


def add_ground(df, home, away):
    """Add a 'ground' column: 'home', 'away' or None for events without a team."""
    df = df.copy()
    df["ground"] = df["Team"].map({home: "home", away: "away"})
    return df


def code_rows(df):
    """Event codes in display order, top row first."""
    codes = [c for c in CODE_ORDER if c in set(df["code"])]
    return codes + sorted(set(df["code"]) - set(codes))  # any code not in CODE_ORDER goes last


def clock(frame):
    """Video frame -> 'mm:ss' of video time."""
    seconds = int(frame / VIDEO_FPS)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def plot_event_timeline(df, home, away, frame_range=None, ax=None, half_labels=True,
                        legend_y=-0.08):
    """Draw the timeline. frame_range=(first, last) zooms in on a window of video frames."""
    if frame_range is not None:
        lo, hi = frame_range
        df = df[(df["end_frame"] >= lo) & (df["start_frame"] <= hi)]

    codes = code_rows(df)
    row_of = {c: len(codes) - 1 - i for i, c in enumerate(codes)}  # first code at the top

    if ax is None:
        _, ax = plt.subplots(figsize=(16, 0.55 * len(codes) + 2))

    # Lane centre offsets within a row; neutral events cover both lanes.
    lanes = {"home": (+LANE_H / 2, LANE_H), "away": (-LANE_H / 2, LANE_H), None: (0, 2 * LANE_H)}
    colour = {"home": COLOURS["home"], "away": COLOURS["away"], None: NEUTRAL}

    # Neutral first so the team bars sit on top of it.
    for ground in (None, "home", "away"):
        sub = df[df["ground"].isna()] if ground is None else df[df["ground"] == ground]
        offset, height = lanes[ground]
        for code, g in sub.groupby("code"):
            spans = list(zip(g["start_frame"], g["end_frame"] - g["start_frame"]))
            ax.broken_barh(spans, (row_of[code] + offset - height / 2, height),
                           facecolors=colour[ground], edgecolors="white", linewidth=0.5,
                           alpha=0.85)

    # Goals as dashed lines across every row, in the scoring team's colour.
    for _, goal in df[df["code"] == "GOALS"].iterrows():
        ax.axvline(goal["start_frame"], color=colour[goal["ground"]], linestyle="--",
                   linewidth=1, alpha=0.6, zorder=0)

    # Halves: shade the gap between them and label each one along the top.
    halves = df.groupby("Half").agg(first=("start_frame", "min"), last=("end_frame", "max"))
    if len(halves) == 2:
        ax.axvspan(halves["last"].iloc[0], halves["first"].iloc[1], color="#000000",
                   alpha=0.05, linewidth=0)
    for half, (first, last) in halves.iterrows() if half_labels else ():
        x = (max(first, frame_range[0]) + min(last, frame_range[1])) / 2 if frame_range else (first + last) / 2
        ax.text(x, len(codes) - 0.45, half, ha="center", va="bottom", fontsize=10, color="#555555")

    # Axes: rows, light gridlines between them, frames along the bottom, minutes along the top.
    ax.set_yticks(range(len(codes)), labels=codes[::-1], fontsize=9)
    ax.set_ylim(-0.6, len(codes) + 0.1)  # headroom above the top row for the half labels
    for y in range(len(codes) - 1):
        ax.axhline(y + 0.5, color="#DDDDDD", linewidth=0.6, zorder=0)
    if frame_range is not None:
        ax.set_xlim(*frame_range)
    ax.set_xlabel("video frame")
    top = ax.secondary_xaxis("top", functions=(lambda f: f / VIDEO_FPS / 60,
                                               lambda m: m * 60 * VIDEO_FPS))
    top.set_xlabel("video time (minutes)")
    for side in ("right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.legend(handles=[Patch(color=COLOURS["home"], label=f"{home} (home) - upper lane"),
                       Patch(color=COLOURS["away"], label=f"{away} (away) - lower lane"),
                       Patch(color=NEUTRAL, label="no team (N/A)")],
              loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=3, frameon=False)
    return ax


def _event_at(df, codes, x, y):
    """The event under the mouse at (frame x, row y), or None. Prefers the team lane hovered."""
    row = round(y)
    if not 0 <= row < len(codes) or abs(y - row) > LANE_H:
        return None
    hits = df[(df["code"] == codes[len(codes) - 1 - row])
              & (df["start_frame"] <= x) & (df["end_frame"] >= x)]
    lane = "home" if y > row else "away"
    for candidates in (hits[hits["ground"] == lane], hits[hits["ground"].isna()]):
        if len(candidates):
            return candidates.iloc[0]
    return None


def _describe(event):
    """Tooltip text for one event: what, who, which frames, and any extra labels."""
    lines = [
        event["code"],
        f"team: {event['Team']}",
        f"frames {event['start_frame']}-{event['end_frame']} "
        f"({event['end_frame'] - event['start_frame'] + 1} frames)",
        f"time {clock(event['start_frame'])}-{clock(event['end_frame'])}  |  {event['Half']}",
    ]
    skip = {"id", "code", "start", "end", "start_frame", "end_frame", "Team", "Half", "ground"}
    lines += [f"{k}: {v}" for k, v in event.items() if k not in skip and isinstance(v, str)]
    return "\n".join(lines)


def interactive_timeline(df, home, away, window_minutes=5):
    """Timeline with sliders: 'position' scrolls through the match, 'window' sets the zoom.

    Hover a bar for its details. Left/right arrow keys step half a window.
    """
    from matplotlib.widgets import Slider

    codes = code_rows(df)
    first, last = int(df["start_frame"].min()), int(df["end_frame"].max())
    fps_min = 60 * VIDEO_FPS

    fig = plt.figure(figsize=(16, 0.55 * len(codes) + 3.5))
    ax = fig.add_axes((0.17, 0.25, 0.80, 0.65))
    plot_event_timeline(df, home, away, ax=ax, half_labels=False, legend_y=-0.1)

    pos_ax = fig.add_axes((0.17, 0.08, 0.72, 0.03))
    win_ax = fig.add_axes((0.17, 0.03, 0.30, 0.03))
    pos = Slider(pos_ax, "position", first, last, valinit=first, valstep=1)
    win = Slider(win_ax, "window (min)", 0.5, 30, valinit=window_minutes, valstep=0.5)

    def update(_=None):
        lo = int(pos.val)
        hi = lo + int(win.val * fps_min)
        ax.set_xlim(lo, hi)
        pos.valtext.set_text(f"{lo} ({clock(lo)})")
        mid = df[(df["start_frame"] <= (lo + hi) / 2)]
        half = mid["Half"].iloc[-1] if len(mid) else ""
        ax.set_title(f"Smart Tagging events - {home} vs {away}   |   frames {lo}-{hi}   "
                     f"({clock(lo)}-{clock(hi)})   {half}", loc="left", fontsize=12)
        fig.canvas.draw_idle()

    pos.on_changed(update)
    win.on_changed(update)

    tip = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points", fontsize=9,
                      bbox=dict(boxstyle="round", fc="white", ec="#888888", alpha=0.95),
                      zorder=10, visible=False)

    def on_move(event):
        x, y = event.xdata, event.ydata
        hit = None
        if event.inaxes is ax and x is not None and y is not None:
            hit = _event_at(df, codes, x, y)
        if hit is None or x is None or y is None:
            if tip.get_visible():
                tip.set_visible(False)
                fig.canvas.draw_idle()
            return
        tip.xy = (x, y)
        tip.set_text(_describe(hit))
        # Flip the tooltip to the left of the cursor in the right half of the view.
        lo, hi = ax.get_xlim()
        right = x > (lo + hi) / 2
        tip.set_position((-12 if right else 12, 12))
        tip.set_horizontalalignment("right" if right else "left")
        tip.set_visible(True)
        fig.canvas.draw_idle()

    def on_key(event):
        step = win.val * fps_min / 2
        if event.key == "right":
            pos.set_val(min(last, pos.val + step))
        elif event.key == "left":
            pos.set_val(max(first, pos.val - step))

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    # Hold on to the sliders: widgets stop responding if they are garbage collected.
    return fig, (pos, win)


if __name__ == "__main__":
    home, away = home_away_names()
    df = add_ground(load_smart_tagging(), home, away)

    print(f"{len(df)} events, home {home!r}, away {away!r}")
    print(df.groupby(["code", "ground"], dropna=False).size().unstack(fill_value=0))

    fig, sliders = interactive_timeline(df, home, away)
    plt.show()
