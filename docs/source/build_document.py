"""Build the technical document docs/NOV4.pdf from docs/source/NOV4.html.

Run with:
    uv run python Scripts/06_ScoreSubmission.py --with-baselines --bootstrap 1000   # scoreboard
    uv run python docs/source/build_document.py

1. Draws the public-window score figure from data/derived/evaluation/scoreboard.csv (written by
   06) into docs/source/figures/scores.png. Only aggregate scores are plotted, no match data.
2. Prints the HTML to PDF with a headless Chrome / Edge (no extra Python dependency).
"""

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCOREBOARD = ROOT / "data" / "derived" / "evaluation" / "scoreboard.csv"
FIGURE = HERE / "figures" / "scores.png"
HTML = HERE / "NOV4.html"
PDF = ROOT / "docs" / "NOV4.pdf"

ACCENT, NEUTRAL, INK, MUTED, GRID = "#1f5fa8", "#b4bac3", "#1f2933", "#6b7280", "#e5e7eb"
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]


def score_figure():
    board = pd.read_csv(SCOREBOARD).sort_values("spell_f1_mean")
    fig, ax = plt.subplots(figsize=(7.2, 2.5), dpi=200)
    y = range(len(board))
    ours = board["submission"].eq("NOV4").to_numpy()
    colours = [ACCENT if o else NEUTRAL for o in ours]
    ax.barh(y, board["spell_f1_mean"], height=0.55, color=colours, zorder=2)
    lo = board["spell_f1_mean"] - board["spell_bootstrap_low"]
    hi = board["spell_bootstrap_high"] - board["spell_f1_mean"]
    ax.errorbar(board["spell_f1_mean"], y, xerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.9,
                capsize=2.5, zorder=3)
    for yi, (v, top, o) in enumerate(zip(board["spell_f1_mean"], board["spell_bootstrap_high"], ours)):
        ax.text(top + 0.012, yi, f"{v:.3f}", va="center", fontsize=8, color=INK,
                fontweight="bold" if o else "normal")
    ax.set_yticks(list(y), [s.replace("baseline_", "baseline ") for s in board["submission"]],
                  fontsize=8, color=INK)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("spellF1 on the public 5-minute window (bar) with 5-95% bootstrap interval "
                  "(1,000 draws)", fontsize=7.5, color=MUTED)
    ax.tick_params(axis="x", labelsize=7.5, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE)
    print(f"wrote {FIGURE}")


def print_pdf():
    browser = next((b for b in BROWSERS if Path(b).is_file() or shutil.which(b)), None)
    if browser is None:
        raise SystemExit("no Chrome / Edge found to print the PDF")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {PDF}")


if __name__ == "__main__":
    if not SCOREBOARD.is_file():
        sys.exit(f"{SCOREBOARD} missing -- run Scripts/06_ScoreSubmission.py --with-baselines first")
    score_figure()
    print_pdf()
