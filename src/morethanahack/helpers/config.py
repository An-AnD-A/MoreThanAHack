from pathlib import Path

# config.py -> helpers -> morethanahack -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = PROJECT_ROOT / "data"
TACTICAL_DIR = DATA_ROOT / "tactical"
BROADCAST_DIR = DATA_ROOT / "broadcast"

DEFAULT_SOURCE = "tactical"

SUFFIXES = {
    "video": ".mp4",
    "atd_meta": "_atd.xml",
    "atd_raw": "_atd.txt",
    "homography": "_homography.csv",
    "smart_tagging": "_smart_tagging.xml",
}

SOURCES = {
    "tactical": {
        "stem": "BAR-ATH_tactical",
        "n_video_frames": 199319,
        # Smart Tagging is timed against the tactical video, so it belongs to this source only.
        "assets": ("video", "atd_meta", "atd_raw", "homography", "smart_tagging"),
    },
    "broadcast": {
        "stem": "BAR-ATH_broadcast",
        "n_video_frames": 164488,
        "assets": ("video", "atd_meta", "atd_raw", "homography"),
    },
}

# Colour scheme for plotting. Blue/orange stay distinguishable for colour-blind viewers
# and on both light and grass-green pitch backgrounds.
COLOURS = {
    "home": "#1F77B4",
    "away": "#FF7F0E",
    "ball": "#FFFFFF",
    "ball_edge": "#000000",  # outline so the ball shows on light pitches too
}

if __name__ == "__main__":
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"TACTICAL_DIR: {TACTICAL_DIR}")
    print(f"BROADCAST_DIR: {BROADCAST_DIR}")
    print(f"DEFAULT_SOURCE: {DEFAULT_SOURCE}")
    print(f"SUFFIXES: {SUFFIXES}")
    print(f"SOURCES: {SOURCES}")
    print(f"COLOURS: {COLOURS}")