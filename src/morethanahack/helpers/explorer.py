"""Inspect the metadata of a loaded kloppy dataset.

Run with:
    uv run python -m morethanahack.helpers.explorer
"""

import math

from morethanahack.helpers.config import COLOURS
from morethanahack.helpers.loader import load, check_alignment, print_report


def _metadata(dataset_or_metadata):
    """Accept either a kloppy dataset or its .metadata object."""
    return getattr(dataset_or_metadata, "metadata", dataset_or_metadata)


def _jersey_sort_key(player):
    try:
        return (0, int(player.jersey_no))
    except (TypeError, ValueError):
        return (1, str(player.jersey_no))


def team_metadata(dataset_or_metadata):
    """Team-level metadata as plain dicts: name, id, home/away and the player list."""
    teams = []
    for team in _metadata(dataset_or_metadata).teams:
        players = [
            {
                "player_id": p.player_id,
                "jersey_no": p.jersey_no,
                "name": p.full_name,
                "starting": p.starting,
                "position": str(p.starting_position) if p.starting_position else None,
            }
            for p in sorted(team.players, key=_jersey_sort_key)
        ]
        teams.append(
            {
                "team_id": team.team_id,
                "name": team.name,
                "ground": team.ground.name.lower() if team.ground else None,
                "n_players": len(players),
                "players": players,
            }
        )
    return teams


def print_team_metadata(dataset_or_metadata):
    """Print each team (home/away) and every player listed for it."""
    for team in team_metadata(dataset_or_metadata):
        ground = (team["ground"] or "unknown").upper()
        print(f"\n{team['name']} ({ground})  id={team['team_id']}  players={team['n_players']}")
        print(f"  {'#':>3}  {'name':<30} {'starter':<8} {'position':<20} player_id")
        for p in team["players"]:
            starter = "yes" if p["starting"] else "no"
            print(
                f"  {str(p['jersey_no']):>3}  {str(p['name']):<30} {starter:<8} "
                f"{str(p['position'] or '-'):<20} {p['player_id']}"
            )


# ---------------------------------------------------------------- pitch

# Which pitch axis each kloppy dimension is measured along. Needed to turn normalised
# values back into metres: "length" scales by pitch_length, "width" by pitch_width.
# goal_height is vertical (off the ground) and kloppy always keeps it in metres.
_PITCH_FIELD_AXIS = {
    "goal_width": "width",
    "six_yard_width": "width",
    "six_yard_length": "length",
    "penalty_area_width": "width",
    "penalty_area_length": "length",
    "circle_radius": "length",
    "corner_radius": "length",
    "penalty_spot_distance": "length",
    "penalty_arc_radius": "length",
    "goal_height": None,
}


def pitch_dimensions(dataset_or_metadata):
    """Everything about the pitch, in native (dataset) units and in metres.

    Returns a dict with:
      unit, pitch_length/width (m), x/y range in native units, coordinate origin and
      vertical orientation, every kloppy dimension as {"native": ..., "metres": ...},
      and derived positions (centre, goals, penalty spots, boxes) in both unit systems.
    Positions use x along the length (goal to goal) and y along the width.
    """
    obj = dataset_or_metadata
    md = getattr(obj, "metadata", obj)
    pd = getattr(md, "pitch_dimensions", md)  # also accept a PitchDimensions directly
    cs = getattr(md, "coordinate_system", None)

    L, W = pd.pitch_length, pd.pitch_width
    x0, x1 = pd.x_dim.min, pd.x_dim.max
    y0, y1 = pd.y_dim.min, pd.y_dim.max
    # metres per native unit along each axis (1.0 if the dataset is already metric)
    mx, my = L / (x1 - x0), W / (y1 - y0)

    sizes = {}
    for field, axis in _PITCH_FIELD_AXIS.items():
        native = getattr(pd, field)
        scale = {"length": mx, "width": my, None: 1.0}[axis]
        sizes[field] = {"native": native, "metres": None if native is None else native * scale}
    m = {k: v["metres"] for k, v in sizes.items()}

    # Derived positions in metres, origin at a pitch corner.
    cy = W / 2
    pos_m = {
        "centre_spot": (L / 2, cy),
        "left_goal_posts": ((0.0, cy - m["goal_width"] / 2), (0.0, cy + m["goal_width"] / 2)),
        "right_goal_posts": ((L, cy - m["goal_width"] / 2), (L, cy + m["goal_width"] / 2)),
        "left_penalty_spot": (m["penalty_spot_distance"], cy),
        "right_penalty_spot": (L - m["penalty_spot_distance"], cy),
        # boxes as (x_min, y_min, x_max, y_max)
        "left_penalty_area": (0.0, cy - m["penalty_area_width"] / 2,
                              m["penalty_area_length"], cy + m["penalty_area_width"] / 2),
        "right_penalty_area": (L - m["penalty_area_length"], cy - m["penalty_area_width"] / 2,
                               L, cy + m["penalty_area_width"] / 2),
        "left_six_yard_box": (0.0, cy - m["six_yard_width"] / 2,
                              m["six_yard_length"], cy + m["six_yard_width"] / 2),
        "right_six_yard_box": (L - m["six_yard_length"], cy - m["six_yard_width"] / 2,
                               L, cy + m["six_yard_width"] / 2),
    }

    def to_native(v):
        """Metre point / tuple of points / box -> native units (same nesting)."""
        if isinstance(v[0], tuple):
            return tuple(to_native(p) for p in v)
        if len(v) == 4:
            return (x0 + v[0] / mx, y0 + v[1] / my, x0 + v[2] / mx, y0 + v[3] / my)
        return (x0 + v[0] / mx, y0 + v[1] / my)

    return {
        "unit": getattr(pd.unit, "value", str(pd.unit)),
        "standardized": pd.standardized,
        "pitch_length_m": L,
        "pitch_width_m": W,
        "x_range_native": (x0, x1),
        "y_range_native": (y0, y1),
        "metres_per_unit": (mx, my),
        "origin": str(getattr(cs, "origin", None)),
        "vertical_orientation": str(getattr(cs, "vertical_orientation", None)),
        "sizes": sizes,
        "positions": {k: {"metres": v, "native": to_native(v)} for k, v in pos_m.items()},
    }


def print_pitch_dimensions(dataset_or_metadata):
    """Pretty-print the output of pitch_dimensions()."""
    d = pitch_dimensions(dataset_or_metadata)
    print(f"\nPitch {d['pitch_length_m']} x {d['pitch_width_m']} m   unit={d['unit']}  "
          f"standardized={d['standardized']}")
    print(f"  native x range {d['x_range_native']}, y range {d['y_range_native']}  "
          f"(metres per unit: x {d['metres_per_unit'][0]:g}, y {d['metres_per_unit'][1]:g})")
    print(f"  origin {d['origin']}, vertical orientation {d['vertical_orientation']}")

    print(f"\n  {'dimension':<24} {'native':>10} {'metres':>8}")
    for k, v in d["sizes"].items():
        print(f"  {k:<24} {v['native']:>10.4f} {v['metres']:>8.2f}")

    fmt = lambda t: "(" + ", ".join(fmt(x) if isinstance(x, tuple) else f"{x:.3f}" for x in t) + ")"
    print(f"\n  {'position':<22} metres / native")
    for k, v in d["positions"].items():
        print(f"  {k:<22} {fmt(v['metres'])}")
        print(f"  {'':<22} {fmt(v['native'])}")
    return d


def _arc(cx, cy, r, t1, t2, n=100):
    """Points (metres) on a circular arc, angles in degrees."""
    import numpy as np

    t = np.radians(np.linspace(t1, t2, n))
    return cx + r * np.cos(t), cy + r * np.sin(t)


def plot_pitch(dataset_or_metadata, units="metres", ax=None, line_color="white",
               pitch_color="#3a7d44", annotate=False, figsize=(10.5, 6.8)):
    """Draw the pitch from the dataset's own dimensions and return the matplotlib Axes.

    units="metres": 0..pitch_length by 0..pitch_width (true-shaped circles).
    units="native": the dataset's own coordinates (e.g. 0..1) -- use this to overlay
                    tracking data without converting it. Circles become ellipses there,
                    as they would in the raw data.
    The y-axis is flipped when the coordinate system's origin is top-left
    (Metrica), so plotted data appears the right way round.
    annotate=True labels the main dimensions in metres.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    d = pitch_dimensions(dataset_or_metadata)
    m = {k: v["metres"] for k, v in d["sizes"].items()}
    L, W = d["pitch_length_m"], d["pitch_width_m"]

    if units == "metres":
        sx = sy = 1.0
        ox = oy = 0.0
    elif units == "native":
        sx, sy = 1 / d["metres_per_unit"][0], 1 / d["metres_per_unit"][1]
        ox, oy = d["x_range_native"][0], d["y_range_native"][0]
    else:
        raise ValueError("units must be 'metres' or 'native'")

    def tx(x): return ox + np.asarray(x) * sx
    def ty(y): return oy + np.asarray(y) * sy

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(pitch_color)
    def line(xs, ys):
        ax.plot(tx(xs), ty(ys), color=line_color, lw=1.5, zorder=1)

    def box(x_min, y_min, x_max, y_max):
        line([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min])

    def arc(cx, cy, r, t1, t2):
        xs, ys = _arc(cx, cy, r, t1, t2)
        line(xs, ys)

    def spot(x, y):
        ax.scatter(tx(x), ty(y), s=12, color=line_color, zorder=2)

    cy = W / 2
    pos = {k: v["metres"] for k, v in d["positions"].items()}

    # outline, halfway line, centre circle and spot
    box(0, 0, L, W)
    line([L / 2, L / 2], [0, W])
    arc(L / 2, cy, m["circle_radius"], 0, 360)
    spot(*pos["centre_spot"])

    for side in ("left", "right"):
        box(*pos[f"{side}_penalty_area"])
        box(*pos[f"{side}_six_yard_box"])
        spot(*pos[f"{side}_penalty_spot"])

        # penalty arc: the part of the circle around the spot that lies outside the box
        px = pos[f"{side}_penalty_spot"][0]
        gap = m["penalty_area_length"] - m["penalty_spot_distance"]
        r = m["penalty_arc_radius"]
        if r > gap:
            theta = np.degrees(np.arccos(gap / r))
            if side == "left":
                arc(px, cy, r, -theta, theta)
            else:
                arc(px, cy, r, 180 - theta, 180 + theta)

        # goal drawn as a small frame behind the line (depth is cosmetic, 2 m)
        (gx, gy0), (_, gy1) = pos[f"{side}_goal_posts"]
        depth = -2.0 if side == "left" else 2.0
        line([gx, gx + depth, gx + depth, gx], [gy0, gy0, gy1, gy1])

    # corner arcs
    cr = m["corner_radius"]
    for (cx_, cy_, t1) in ((0, 0, 0), (L, 0, 90), (L, W, 180), (0, W, 270)):
        arc(cx_, cy_, cr, t1, t1 + 90)

    if annotate:
        def label(x, y, text, rotation=0):
            ax.text(float(tx(x)), float(ty(y)), text, rotation=rotation, color=line_color,
                    fontsize=8, ha="center", va="center", zorder=3)

        label(L / 2, -1.5, f"length {L:g} m")
        label(-7.0, cy, f"width {W:g} m", rotation=90)
        label(-3.5, cy, f"goal {m['goal_width']:.2f} x {m['goal_height']:.2f} m", rotation=90)
        pa = pos["left_penalty_area"]
        label(pa[2] / 2, pa[3] + 1.5,
              f"box {m['penalty_area_length']:.1f} x {m['penalty_area_width']:.2f} m")
        sb = pos["left_six_yard_box"]
        label(sb[2] + 3, sb[1] - 1.2,
              f"6yd {m['six_yard_length']:.1f} x {m['six_yard_width']:.2f} m")
        label(m["penalty_spot_distance"], cy + 1.8, f"spot {m['penalty_spot_distance']:.1f} m")
        label(L / 2, cy + m["circle_radius"] + 1.5, f"r {m['circle_radius']:.2f} m")

    pad = 9.0 if annotate else 5.0
    ax.set_xlim(float(tx(-pad)), float(tx(L + pad)))
    ax.set_ylim(float(ty(-pad)), float(ty(W + pad)))
    if "TOP_TO_BOTTOM" in d["vertical_orientation"]:
        ax.invert_yaxis()
    if units == "metres":
        ax.set_aspect("equal")
    else:
        # keep the true pitch shape even though native units are not square:
        # aspect = metres per y-unit / metres per x-unit
        ax.set_aspect(sx / sy)
    ax.set_xlabel(f"x ({units})")
    ax.set_ylabel(f"y ({units})")
    return ax


# ---------------------------------------------------------------- frames

def _is_valid(point):
    """True when a kloppy Point has real x/y (untracked positions come through as NaN)."""
    return point is not None and not (math.isnan(point.x) or math.isnan(point.y))


def _get_frame(dataset, frame):
    """Accept an ATD frame_id (int) or a kloppy Frame."""
    if hasattr(frame, "players_data"):
        return frame
    frame = int(frame)
    # Fast path: ids are contiguous (see loader.check_alignment), so index directly.
    # kloppy's get_record_by_id scans every record, which is slow late in the match.
    records = dataset.records
    if records:
        i = frame - int(records[0].frame_id)
        if 0 <= i < len(records) and int(records[i].frame_id) == frame:
            return records[i]
    record = dataset.get_record_by_id(frame)
    if record is None:
        raise KeyError(f"no frame with frame_id {frame} in this dataset")
    return record


def frame_positions(dataset, frame, units="native"):
    """Player and ball positions for one frame, skipping anyone not tracked in it.

    frame: ATD frame_id (counts from 1; see loader.frame_id_of for video frames) or a Frame.
    units: "native" (dataset coordinates) or "metres" (origin at the pitch corner,
           same orientation as plot_pitch).
    Returns {"frame_id", "period", "timestamp", "players": [...], "ball": (x, y) or None}.
    """
    f = _get_frame(dataset, frame)
    if units == "native":
        def conv(p): return (p.x, p.y)
    elif units == "metres":
        d = pitch_dimensions(dataset)
        (x0, _), (y0, _) = d["x_range_native"], d["y_range_native"]
        mx, my = d["metres_per_unit"]
        def conv(p): return ((p.x - x0) * mx, (p.y - y0) * my)
    else:
        raise ValueError("units must be 'metres' or 'native'")

    players = [
        {
            "player_id": player.player_id,
            "jersey_no": player.jersey_no,
            "name": player.full_name,
            "ground": player.team.ground.name.lower() if player.team and player.team.ground else None,
            "xy": conv(data.coordinates),
        }
        for player, data in f.players_data.items()
        if _is_valid(data.coordinates)
    ]
    ball = conv(f.ball_coordinates) if _is_valid(f.ball_coordinates) else None
    return {
        "frame_id": f.frame_id,
        "period": f.period.id if f.period else None,
        "timestamp": f.timestamp,
        "players": players,
        "ball": ball,
    }


def plot_frame(dataset, frame, units="native", ax=None, show_jersey=True, **pitch_kwargs):
    """Draw the pitch and the tracked players and ball for one frame; return the Axes.

    frame: ATD frame_id or a kloppy Frame. Players not tracked in that frame are left off,
    and the ball is left off when its position is missing.
    Extra keyword arguments go to plot_pitch (pitch_color, line_color, annotate, ...).
    """
    pos = frame_positions(dataset, frame, units=units)
    if ax is None:
        ax = plot_pitch(dataset, units=units, **pitch_kwargs)
    else:
        plot_pitch(dataset, units=units, ax=ax, **pitch_kwargs)

    for ground in ("home", "away", None):
        group = [p for p in pos["players"] if p["ground"] == ground]
        if not group:
            continue
        xs, ys = zip(*(p["xy"] for p in group))
        ax.scatter(xs, ys, s=180, color=COLOURS.get(ground or "", "grey"), edgecolors="white",
                   linewidths=1.0, zorder=4, label=ground or "unknown")
        if show_jersey:
            for p, x, y in zip(group, xs, ys):
                ax.text(x, y, str(p["jersey_no"]), color="white", fontsize=7,
                        fontweight="bold", ha="center", va="center", zorder=5)

    if pos["ball"] is not None:
        ax.scatter(*pos["ball"], s=60, color=COLOURS["ball"], edgecolors=COLOURS["ball_edge"],
                   linewidths=1.0, zorder=6, label="ball")

    ball_note = "" if pos["ball"] is not None else "  (ball not tracked)"
    ax.set_title(f"frame {pos['frame_id']}  period {pos['period']}  {_clock(pos['timestamp'])}  "
                 f"players {len(pos['players'])}{ball_note}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    return ax


def _clock(timestamp):
    secs = timestamp.total_seconds() if hasattr(timestamp, "total_seconds") else timestamp
    return f"{int(secs // 60):02d}:{secs % 60:04.1f}" if secs is not None else "?"


def _frame_title(pos):
    ball_note = "" if pos["ball"] is not None else "  (ball not tracked)"
    return (f"frame {pos['frame_id']}  period {pos['period']}  {_clock(pos['timestamp'])}  "
            f"players {len(pos['players'])}{ball_note}")


class _FrameArtists:
    """The movable bits drawn on a pitch (player dots, jersey labels, ball, title).

    Created once, then draw(pos) moves them to a frame -- much faster than redrawing
    the pitch each time. Shared by animate_frames and FrameViewer.
    """

    GROUPS = ("home", "away", None)

    def __init__(self, ax, show_jersey=True):
        import numpy as np

        self.ax = ax
        self.show_jersey = show_jersey
        self.empty = np.empty((0, 2))
        colours = {"home": COLOURS["home"], "away": COLOURS["away"], None: "grey"}
        self.scatters = {
            g: ax.scatter([], [], s=180, color=colours[g], edgecolors="white", linewidths=1.0,
                          zorder=4, label=g or "unknown")
            for g in self.GROUPS
        }
        self.ball = ax.scatter([], [], s=60, color=COLOURS["ball"], edgecolors=COLOURS["ball_edge"],
                               linewidths=1.0, zorder=6, label="ball")
        self.labels = []
        self.title = ax.set_title("")

    def legend(self, groups=("home", "away")):
        handles = [self.scatters[g] for g in self.GROUPS if g in groups] + [self.ball]
        self.ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)

    def _label_pool(self, n):
        while len(self.labels) < n:  # grow on demand, reuse afterwards
            self.labels.append(self.ax.text(0, 0, "", color="white", fontsize=7,
                                            fontweight="bold", ha="center", va="center",
                                            zorder=5, visible=False))
        return self.labels

    def draw(self, pos):
        import numpy as np

        for g, sc in self.scatters.items():
            pts = [p["xy"] for p in pos["players"] if p["ground"] == g]
            sc.set_offsets(np.array(pts) if pts else self.empty)
        self.ball.set_offsets(np.array([pos["ball"]]) if pos["ball"] is not None else self.empty)

        players = pos["players"] if self.show_jersey else []
        for j, label in enumerate(self._label_pool(len(players))):
            if j < len(players):
                label.set_position(players[j]["xy"])
                label.set_text(str(players[j]["jersey_no"]))
                label.set_visible(True)
            else:
                label.set_visible(False)

        self.title.set_text(_frame_title(pos))
        return [*self.scatters.values(), self.ball, *self.labels, self.title]


def animate_frames(dataset, start, end, step=1, units="native", interval=None,
                   show_jersey=True, save_path=None, fps=None, **pitch_kwargs):
    """Animate players and ball over frame_ids start..end (inclusive) on top of the pitch.

    step:      use every step-th frame (e.g. 5 to skim a long passage).
    interval:  ms between animation frames; default plays in real time
               (1000 / dataset frame rate * step).
    save_path: if given, write the animation there. ".gif" uses Pillow (always available);
               ".mp4" needs ffmpeg on PATH. fps defaults to real time as well.
    Extra keyword arguments go to plot_pitch. Returns the FuncAnimation -- keep a
    reference to it, or it is garbage-collected and stops. In Jupyter, show it with
    IPython.display.HTML(anim.to_jshtml()).
    To step through frames by hand (frame by frame / a second at a time) use FrameViewer.
    """
    from matplotlib.animation import FuncAnimation

    if end < start:
        raise ValueError(f"end ({end}) is before start ({start})")
    positions = [frame_positions(dataset, fid, units=units) for fid in range(start, end + 1, step)]

    frame_rate = getattr(dataset.metadata, "frame_rate", None) or 25
    if interval is None:
        interval = 1000 / frame_rate * step
    if fps is None:
        fps = max(1, round(1000 / interval))

    ax = plot_pitch(dataset, units=units, **pitch_kwargs)
    fig = ax.get_figure(root=True)
    assert fig is not None  # plot_pitch always draws onto a figure

    artists = _FrameArtists(ax, show_jersey)
    # Legend only for groups that actually appear in the clip.
    artists.legend({p["ground"] for pos in positions for p in pos["players"]})

    anim = FuncAnimation(fig, lambda i: artists.draw(positions[i]), frames=len(positions),
                         interval=interval, blit=False)

    if save_path is not None:
        save_path = str(save_path)
        writer = "pillow" if save_path.lower().endswith(".gif") else "ffmpeg"
        anim.save(save_path, writer=writer, fps=fps)
    return anim


def _is_inline_backend():
    import matplotlib

    return "inline" in matplotlib.get_backend().lower()


class FrameViewer:
    """Step through the tracking data by hand: frame by frame or a second at a time.

    viewer = FrameViewer(dataset, start=11185, end=11435)   # keep the reference!

    In a matplotlib window (plain `python` script, or `%matplotlib tk` in a notebook):
      slider          scrub to any frame in start..end
      buttons         -1s  -1  play/pause  +1  +1s
      left / right    1 frame back / forward
      up / down, shift+left / shift+right
                      1 second (frame_rate frames, 25 here) forward / back
      space           play / pause (advances play_step frames per tick, in real time)
      home / end      jump to start / end

    With the inline backend (static images, e.g. the VS Code Interactive window by
    default) nothing on the figure is clickable, so call the methods instead -- each
    one redraws the figure below the cell:
      viewer.next()  viewer.prev()  viewer.next_second()  viewer.prev_second()
      viewer.step(n)  viewer.goto(frame_id)

    start/end are ATD frame_ids (counts from 1) and default to the whole dataset.
    Extra keyword arguments go to plot_pitch.
    """

    def __init__(self, dataset, start=None, end=None, units="native", show_jersey=True,
                 play_step=1, **pitch_kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider

        records = dataset.records
        if not records:
            raise ValueError("dataset has no frames")
        first, last = int(records[0].frame_id), int(records[-1].frame_id)
        self.start = first if start is None else max(first, int(start))
        self.end = last if end is None else min(last, int(end))
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) is before start ({self.start})")

        self.dataset = dataset
        self.units = units
        self.frame_rate = getattr(dataset.metadata, "frame_rate", None) or 25
        self.second = int(round(self.frame_rate))  # frames per second of play
        self.play_step = int(play_step)
        self.frame_id = self.start
        self.playing = False
        self._inline = _is_inline_backend()
        self._ready = False

        # Matplotlib binds left/right to the toolbar's back/forward by default; free them
        # so the arrow keys step frames instead.
        for name in ("keymap.back", "keymap.forward"):
            plt.rcParams[name] = [k for k in plt.rcParams[name] if k not in ("left", "right")]

        self.ax = plot_pitch(dataset, units=units, **pitch_kwargs)
        fig = self.ax.get_figure(root=True)
        assert fig is not None
        self.fig = fig
        self.fig.subplots_adjust(bottom=0.2)

        self.artists = _FrameArtists(self.ax, show_jersey)
        self.artists.legend()

        # widgets -- kept on self, otherwise they are garbage-collected and stop responding
        self.slider = Slider(self.fig.add_axes((0.12, 0.10, 0.76, 0.03)), "frame",
                             self.start, self.end, valinit=self.start, valstep=1, valfmt="%d")
        self.slider.on_changed(lambda v: self.goto(int(v)))

        controls = [
            ("-1s", self.prev_second),
            ("-1", self.prev),
            ("play", self.toggle_play),
            ("+1", self.next),
            ("+1s", self.next_second),
        ]
        w, gap = 0.1, 0.02
        x = 0.5 - (len(controls) * w + (len(controls) - 1) * gap) / 2
        self.buttons = []
        for i, (text, action) in enumerate(controls):
            button = Button(self.fig.add_axes((x + i * (w + gap), 0.02, w, 0.05)), text)
            button.on_clicked(lambda _event, action=action: action())
            self.buttons.append(button)
        self.play_button = self.buttons[2]

        self.timer = self.fig.canvas.new_timer(interval=int(1000 / self.frame_rate * self.play_step))
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._render()
        self._ready = True

    # ------------------------------------------------------------ navigation

    def goto(self, frame_id):
        """Show frame_id (clamped to start..end)."""
        fid = min(max(int(frame_id), self.start), self.end)
        self.frame_id = fid
        if int(self.slider.val) != fid:
            self.slider.set_val(fid)  # fires on_changed -> goto(fid) again -> renders once
            return
        self._render()

    def step(self, n):
        """Move n frames (negative = backwards)."""
        self.goto(self.frame_id + n)

    def next(self, n=1):
        self.step(n)

    def prev(self, n=1):
        self.step(-n)

    def next_second(self, n=1):
        self.step(n * self.second)

    def prev_second(self, n=1):
        self.step(-n * self.second)

    def toggle_play(self):
        if self._inline:
            print("play needs an interactive backend (e.g. %matplotlib tk); "
                  "use next() / next_second() here.")
            return
        self.playing = not self.playing
        if self.playing:
            if self.frame_id >= self.end:
                self.goto(self.start)
            self.timer.start()
        else:
            self.timer.stop()
        self.play_button.label.set_text("pause" if self.playing else "play")
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------ internals

    def _tick(self):
        if self.frame_id >= self.end:
            self.toggle_play()  # reached the end: stop
            return
        self.step(self.play_step)

    def _on_key(self, event):
        moves = {
            "right": 1, "left": -1,
            "up": self.second, "down": -self.second,
            "shift+right": self.second, "shift+left": -self.second,
        }
        if event.key in moves:
            self.step(moves[event.key])
        elif event.key == " ":
            self.toggle_play()
        elif event.key == "home":
            self.goto(self.start)
        elif event.key == "end":
            self.goto(self.end)

    def _render(self):
        pos = frame_positions(self.dataset, self.frame_id, units=self.units)
        self.artists.draw(pos)
        self.fig.canvas.draw_idle()
        if self._inline and self._ready:
            from IPython.display import display

            display(self.fig)

    def __repr__(self):
        return f"FrameViewer(frame {self.frame_id}, range {self.start}..{self.end})"


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = load(source="tactical")
    print_team_metadata(dataset)
    print_pitch_dimensions(dataset)
    plot_pitch(dataset, annotate=True)
    # plot_frame(dataset, 12001)
    # anim = animate_frames(dataset, 11185, 11435)  # 10 s of play at 25 fps
    # Tactical frames
    viewer = FrameViewer(dataset, 11092, 90651)  # arrows = 1 frame, up/down = 1 s
    plt.show()

    # Broadcast Frames
    viewer = FrameViewer(dataset, 11092, 90651)  # arrows = 1 frame, up/down = 1 s
