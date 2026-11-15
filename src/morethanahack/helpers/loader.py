"""Load the ATD (FIFA/EPTS tracking data) and check it lines up with the video frames.

Run with:
    uv run python -m morethanahack.helpers.loader
"""

from pathlib import Path

from morethanahack.helpers.config import DATA_ROOT, DEFAULT_SOURCE, SOURCES, SUFFIXES

# Raw ATD line: "<frame_id>:<players>:<ball>" -- the frame id is everything before the first ":".
FIELD_SEP = ":"


# ---------------------------------------------------------------- paths

def atd_paths(source=DEFAULT_SOURCE, data_root=DATA_ROOT):
    """(meta_xml, raw_txt) for a source. Raises one error listing every missing file."""
    if source not in SOURCES:
        raise KeyError(f"no such source {source!r}; have {sorted(SOURCES)}")

    stem = SOURCES[source]["stem"]
    folder = Path(data_root) / source
    meta = folder / (stem + SUFFIXES["atd_meta"])
    raw = folder / (stem + SUFFIXES["atd_raw"])

    missing = [str(p) for p in (meta, raw) if not p.is_file()]
    if missing:
        raise FileNotFoundError("ATD files not found:\n  " + "\n  ".join(missing))
    return meta, raw


def expected_frames(source=DEFAULT_SOURCE):
    """Number of video frames in that source's recording."""
    return SOURCES[source]["n_video_frames"]


# ---------------------------------------------------------------- loading

def load(source=DEFAULT_SOURCE, data_root=DATA_ROOT):
    """The ATD as a kloppy TrackingDataset (slow: minutes and a few GB for the full match)."""
    from kloppy import metrica

    meta, raw = atd_paths(source, data_root)
    return metrica.load_tracking_epts(meta_data=str(meta), raw_data=str(raw))


# ---------------------------------------------------------------- frame mapping

def video_frame_of(frame_id):
    """ATD frame_id (counts from 1) -> video frame / submission row (counts from 0)."""
    return int(frame_id) - 1


def frame_id_of(video_frame):
    """Video frame -> ATD frame_id. Inverse of video_frame_of."""
    return int(video_frame) + 1


# ---------------------------------------------------------------- alignment

def _alignment_report(ids, expected):
    """Shared logic: given the frame ids (any order), check they are exactly 1..expected."""
    ids = sorted(ids)
    n = len(ids)
    contiguous = n > 0 and ids == list(range(ids[0], ids[0] + n))
    duplicates = n - len(set(ids))
    return {
        "n_records": n,
        "n_video_frames": expected,
        "frame_id_range": (ids[0], ids[-1]) if ids else (None, None),
        "contiguous": contiguous,
        "duplicates": duplicates,
        "first_video_frame": video_frame_of(ids[0]) if ids else None,
        "agrees": n == expected and contiguous and ids[0] == 1,
    }


def check_alignment(dataset, source=DEFAULT_SOURCE):
    """Check a loaded kloppy dataset has one record per video frame, ids 1..n, no gaps."""
    ids = [int(record.frame_id) for record in dataset.records]
    return _alignment_report(ids, expected_frames(source))


def frame_ids(raw_path, sep=FIELD_SEP):
    """Stream the frame id off each line of the raw ATD file."""
    with open(raw_path) as f:
        for line in f:
            if line.strip():
                yield int(line.split(sep, 1)[0])


def check_alignment_raw(source=DEFAULT_SOURCE, data_root=DATA_ROOT):
    """Same check straight off the .txt -- seconds instead of minutes, no kloppy needed."""
    _, raw = atd_paths(source, data_root)
    return _alignment_report(frame_ids(raw), expected_frames(source))


def print_report(report, label="alignment check"):
    print(f"\n{label}")
    print(f"  records {report['n_records']}, video frames {report['n_video_frames']}")
    lo, hi = report["frame_id_range"]
    print(f"  frame_id range {lo}..{hi}, contiguous: {report['contiguous']}, "
          f"duplicates: {report['duplicates']}")
    if report["agrees"]:
        print("  OK -- one record per video frame, numbered from 1. video_frame = frame_id - 1.")
    else:
        print("  MISMATCH -- not one record per video frame from 1. Report it; don't patch it.")


if __name__ == "__main__":
    # Cheap check first -- if this fails there is no point spending minutes on the full load.
    print_report(check_alignment_raw(), "raw alignment check")

    dataset = load()
    print(f"\n{len(dataset.records)} frames loaded")
    print_report(check_alignment(dataset), "dataset alignment check")
