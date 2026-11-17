"""Dead-ball threshold experiments: re-run Stage 1 of 05 under parameter overrides, write one
submission per experiment, and score them all with the official scorer (via 06).

Run with:
    uv run python Scripts/07_DeadBallExperiments.py                  # all experiments
    uv run python Scripts/07_DeadBallExperiments.py --only E0,E1     # a subset (ID prefixes)
    uv run python Scripts/07_DeadBallExperiments.py --bootstrap 1000 # + spellF1 intervals

Outputs go to data/derived/experiments/deadball/ (inside data/, so git ignores them):
    <id>.csv       one full submission per experiment (same format as 05's possession.csv)
    summary.csv    overrides + public-window scores (D precision/recall, spellF1, macroF1, ...)
                   + label-free full-match diagnostics, one row per experiment

Only Stage 1 (dead balls) changes between experiments. Stage 2 evidence, the tag prior and the
decoder are computed once and shared, so every score difference comes from dead-ball detection.
05's production possession.csv is not touched.

Overrides are "module.NAME": value, where module is v04 (velocity-drop detector) or v05 (dead-ball
classification / refinement / clean-up). They are set for one experiment and restored afterwards.

CAVEAT: the public window is 5 minutes / 17 possessions (see 06). The label-free columns (dead
share, set-piece agreement, possession counts) are printed next to it so a change is only trusted
when both point the same way.
"""

import argparse
import hashlib
import importlib.util
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from morethanahack.helpers.config import DATA_ROOT, DEFAULT_SOURCE
from morethanahack.helpers.loader import VIDEO_FPS, load_smart_tagging


def _load_script(filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location("script_" + path.stem.split("_", 1)[1], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v05 = _load_script("05_PossessionAssignment.py")
v04 = v05.v04
MODULES = {"v04": v04, "v05": v05}

OUT_DIR = DATA_ROOT / "derived" / "experiments" / "deadball"
SCORE_SCRIPT = Path(__file__).with_name("06_ScoreSubmission.py")
SCOREBOARD = DATA_ROOT / "derived" / "evaluation" / "scoreboard.csv"
BASELINE_SHA = "9c7e0a1111bba44a8e99b7e09f9ad272bb7e524b5c2741b3df664ccfbbde6408"

# ---------------------------------------------------------------- experiments
# One factor group per experiment. Add rows here; E7 (combination) is filled in after E1-E6.
EXPERIMENTS = {
    "E0_baseline": {},
    "E1_loose_hysteresis": {"v04.ENTER": 0.7, "v04.EXIT": 0.9},
    "E2_strict_hysteresis": {"v04.ENTER": 0.5, "v04.EXIT": 0.7},
    "E3_short_stoppages": {"v04.SMOOTH_S": 2, "v04.MIN_DROP_S": 2.5, "v04.MERGE_GAP_S": 1.5},
    "E4_relaxed_dead_tests": {"v04.DEAD_BALL_TRACKED": 0.7, "v05.PARKED_SPEED": 1.0,
                              "v05.PARKED_S": 2.0},
    "E5_sensitive_out": {"v05.OUT_MARGIN_M": 0.5, "v05.OUT_MIN_S": 0.2},
    "E6_wider_edges": {"v05.START_LOOKBACK_S": 8, "v05.END_LOOKAHEAD_S": 4, "v05.MIN_ALIVE_S": 5},
    # E1 (set-piece tag recall 0.77 -> 0.89 at equal tag precision) + E5 (only group that moved
    # the public window); E3/E4 raise tag recall but lower tag precision, E6 changes nothing.
    "E7_combined": {"v04.ENTER": 0.7, "v04.EXIT": 0.9, "v05.OUT_MARGIN_M": 0.5, "v05.OUT_MIN_S": 0.2},
}


@contextmanager
def overrides(params):
    """Set module-level constants for one experiment, restore them afterwards."""
    saved = []
    try:
        for key, value in params.items():
            mod_name, attr = key.split(".", 1)
            mod = MODULES[mod_name]
            if not hasattr(mod, attr):
                raise AttributeError(f"{key}: no such parameter")
            saved.append((mod, attr, getattr(mod, attr)))
            setattr(mod, attr, value)
        yield
    finally:
        for mod, attr, value in reversed(saved):
            setattr(mod, attr, value)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(exp_id, params, shared):
    signals, pitch, scope, tags, e = (shared[k] for k in ("signals", "pitch", "scope", "tags", "e"))
    n = len(signals)
    with overrides(params):
        dead, spans, ev, n_lulls = v05.dead_spans(signals, pitch, scope)
        labels, _ = v05.decode(e, dead, v05.restart_flags(dead, spans, n))

    path = v05.write_submission(OUT_DIR / f"{exp_id}.csv", labels)
    poss = v05.possession_table(labels, scope)

    in_scope = list(zip(*v04.runs(dead & scope)))
    set_pieces = tags[(tags["code"] == "SET PIECES") & (tags["Type"] != "Kick Off (Start)")]
    m = v04.match_matrix(set_pieces["start_frame"].to_numpy(),
                         pd.DataFrame(in_scope, columns=["start_frame", "end_frame"]))
    return path, {
        "id": exp_id,
        "overrides": "; ".join(f"{k}={v}" for k, v in params.items()) or "-",
        "full_dead_share": (dead & scope).sum() / scope.sum(),
        "full_dead_spans": len(in_scope),
        "full_lulls_kept_alive": n_lulls,
        "full_possessions": len(poss),
        "full_median_poss_s": poss["duration_s"].median(),
        "sp_tag_recall": m.any(axis=1).mean() if m.size else np.nan,
        "sp_tag_precision": m.any(axis=0).mean() if m.size else np.nan,
        "sha256": sha256(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default="", help="comma-separated experiment ID prefixes")
    parser.add_argument("--bootstrap", type=int, default=0, help="bootstrap draws passed to 06")
    args = parser.parse_args()

    wanted = [p.strip() for p in args.only.split(",") if p.strip()]
    todo = {k: v for k, v in EXPERIMENTS.items()
            if not wanted or any(k == w or k.startswith(w + "_") for w in wanted)}
    if not todo:
        raise SystemExit(f"no experiment matches {wanted}; known: {list(EXPERIMENTS)}")

    # ---------- shared, Stage-1-independent inputs (computed once)
    signals, pos, team, pitch, names = v05.load_tracking()
    n = len(signals)
    tags = load_smart_tagging(DEFAULT_SOURCE)
    scope = v05.v03.in_play_mask(tags, n)
    tracked = ~np.isnan(signals["ball_x"].to_numpy())
    e = v05.control_evidence(signals, pos, team)
    prior, _ = v05.tag_prior(tags, names, n)
    e = e + v05.TAG_WEIGHT * prior * ~tracked
    shared = {"signals": signals, "pitch": pitch, "scope": scope, "tags": tags, "e": e}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths, rows = [], []
    for exp_id, params in todo.items():
        print(f"\n== {exp_id}: {params or 'baseline'} ==")
        path, row = run_experiment(exp_id, params, shared)
        if not v05.official_validation(path):
            raise SystemExit(f"{path} failed the official validation")
        if exp_id == "E0_baseline":
            ok = row["sha256"] == BASELINE_SHA
            print(f"  baseline SHA-256 {'matches' if ok else 'DIFFERS from'} the documented submission")
        paths.append(path)
        rows.append(row)

    # ---------- official scorer, all experiments in one call
    cmd = [sys.executable, str(SCORE_SCRIPT), *map(str, paths), "--no-detail",
           "--bootstrap", str(args.bootstrap)]
    code = subprocess.run(cmd).returncode
    if code != 0 or not SCOREBOARD.is_file():
        raise SystemExit(f"scoring failed (exit {code})")

    board = pd.read_csv(SCOREBOARD).rename(columns={"submission": "id"})
    keep = ["id", "precision_D", "recall_D", "f1_D", "spell_f1_mean", "macro_f1", "poss",
            "alive_dead_f1", "spell_bootstrap_low", "spell_bootstrap_high"]
    summary = pd.DataFrame(rows).merge(board[[c for c in keep if c in board]], on="id", how="left")
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    show = ["id", "precision_D", "recall_D", "f1_D", "spell_f1_mean", "macro_f1", "poss",
            "full_dead_share", "full_dead_spans", "full_possessions", "sp_tag_recall"]
    print("\n== Summary (public window: D precision/recall, spellF1, macroF1, poss; "
          "full_* and sp_tag_*: whole match, label-free) ==")
    print(summary[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nOverrides:")
    for _, r in summary.iterrows():
        print(f"  {r['id']}: {r['overrides']}")
    print(f"\nwrote {OUT_DIR / 'summary.csv'}")
    print("NOTE: public window = 5 min / 17 possessions; trust a change only if the full-match "
          "diagnostics agree.")


if __name__ == "__main__":
    main()
