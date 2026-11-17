"""Score a submission CSV with the official scorer (archive/REPO/evaluation/run_evaluation.py).

The official scorer is run unchanged, as a command. It scores every *.csv in a results folder, so
this script copies the chosen CSV(s) into its own folder under data/ and points the scorer there.
Nothing is written into archive/.

Run with:
    uv run python Scripts/06_ScoreSubmission.py                          # default: 05's output
    uv run python Scripts/06_ScoreSubmission.py data/derived/x/my.csv    # any CSV(s)
    uv run python Scripts/06_ScoreSubmission.py --with-baselines         # + the 4 host baselines
    uv run python Scripts/06_ScoreSubmission.py --bootstrap 1000         # confidence intervals
    uv run python Scripts/06_ScoreSubmission.py --no-detail              # ranked table only

Outputs go to data/derived/evaluation/ (inside data/, so git ignores them):
    results/                    the CSVs that were scored (rebuilt on every run)
    scoreboard.csv              one row per submission, every metric
    reports/<name>.json         full metric dump per submission
    reports/<name>_transitions.csv   every transition: matched / missed / spurious, with frames

READ THIS BEFORE USING THE NUMBERS. Only the public ground truth (--gt public, 5 minutes,
17 possessions) is available to participants. It is a CORRECTNESS check -- alignment, format,
sane possession counts -- not a way to choose between versions: its noise band is several times
the ~0.04 full-match band (run --bootstrap 1000 to see it), and tuning on it overfits 5 minutes.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from morethanahack.helpers.config import DATA_ROOT, PROJECT_ROOT

EVAL_DIR = PROJECT_ROOT / "archive" / "REPO" / "evaluation"
SCORER = EVAL_DIR / "run_evaluation.py"
VALIDATOR = EVAL_DIR / "validate_submission.py"
HOST_BASELINES = EVAL_DIR / "results"
DEFAULT_SUBMISSION = DATA_ROOT / "derived" / "possession" / "submission" / "NOV4.csv"
OUT_DIR = DATA_ROOT / "derived" / "evaluation"


def stage_results(submissions, with_baselines):
    """Fresh results folder holding exactly the files to score. Returns its path."""
    results = OUT_DIR / "results"
    results.mkdir(parents=True, exist_ok=True)
    for old in results.glob("*.csv"):          # only this script's own staging copies ...
        old.unlink()
    for old in (OUT_DIR / "reports").glob("*"):  # ... and its own reports from the last run
        if old.suffix in (".json", ".csv"):
            old.unlink()

    names = set()
    for path in submissions:
        if path.name in names:
            raise SystemExit(f"two submissions named {path.name}; the scoreboard keys on the file name")
        names.add(path.name)
        shutil.copy2(path, results / path.name)
    if with_baselines:
        for base in sorted(HOST_BASELINES.glob("baseline_*.csv")):
            shutil.copy2(base, results / base.name)
    return results


def validate(path):
    """The hosts' format check, same as the scorer's. Prints a one-line verdict."""
    r = subprocess.run([sys.executable, str(VALIDATOR), str(path)], capture_output=True, text=True,
                       cwd=EVAL_DIR)
    first = (r.stdout or r.stderr).strip().splitlines()
    print(f"  {first[0] if first else 'no output'}" + ("" if r.returncode == 0 else
                                                       "\n  " + "\n  ".join(first[1:3])))
    return r.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submissions", nargs="*", type=Path, default=[DEFAULT_SUBMISSION],
                        help=f"CSV(s) to score (default: {DEFAULT_SUBMISSION.relative_to(PROJECT_ROOT)})")
    parser.add_argument("--with-baselines", action="store_true",
                        help="score the host baselines alongside, for reference")
    parser.add_argument("--gt", default="public", help="'public' (default), 'private', or a path")
    parser.add_argument("--bootstrap", type=int, default=0, help="bootstrap draws, 0 = off")
    parser.add_argument("--no-detail", action="store_true", help="ranked table only")
    args = parser.parse_args()

    submissions = [p.resolve() for p in args.submissions]
    missing = [str(p) for p in submissions if not p.is_file()]
    if missing:
        raise SystemExit("not found:\n  " + "\n  ".join(missing))
    if not SCORER.is_file():
        raise SystemExit(f"official scorer not found at {SCORER} -- is the archive submodule checked out?")

    print("== Format check (validate_submission.py) ==")
    if not all([validate(p) for p in submissions]):
        raise SystemExit("fix the failing file(s) first -- the scorer would skip them")

    results = stage_results(submissions, args.with_baselines)
    cmd = [sys.executable, str(SCORER), "--results", str(results), "--out", str(OUT_DIR),
           "--gt", args.gt, "--bootstrap", str(args.bootstrap)]
    if args.no_detail:
        cmd.append("--no-detail")
    print(f"\n== Official scorer (run_evaluation.py, --gt {args.gt}) ==\n", flush=True)
    code = subprocess.run(cmd, cwd=EVAL_DIR).returncode   # cwd: it imports possession_metrics
    if args.gt == "public":
        print("\nNOTE: public = 5 minutes / 17 possessions. Use it to check correctness, not to pick "
              "between versions.")
    sys.exit(code)


if __name__ == "__main__":
    main()
