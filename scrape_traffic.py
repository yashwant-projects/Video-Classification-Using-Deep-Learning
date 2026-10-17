import os
import glob
import subprocess

TRAFFIC_DIR = "Dataset/Traffic"

print("[*] Clearing old traffic videos...")
for f in glob.glob(os.path.join(TRAFFIC_DIR, "*.*")):
    try:
        os.remove(f)
    except Exception as e:
        print(f"Error deleting {f}: {e}")

print(f"[*] Cleared {TRAFFIC_DIR}. Starting yt-dlp download...")

# We will run 3 different searches to get a diverse set of 150 videos (50 each).
# ytsearch200 ensures we have a large pool of results to filter from.
queries = [
    "ytsearch300:dashcam traffic driving short",
    "ytsearch300:drone traffic highway short",
    "ytsearch300:city street traffic walking short"
]

for idx, query in enumerate(queries):
    print(f"\n[*] Running Search {idx+1}/3: {query}")
    cmd = [
        "yt-dlp",
        query,
        "--match-filter", "duration > 15 & duration < 240",
        "-f", "bestvideo[ext=mp4][height<=480]/bestvideo[ext=mp4]",
        "--max-downloads", "50",
        "--ignore-errors",
        "-o", f"{TRAFFIC_DIR}/traffic_v{idx}_%(autonumber)s.%(ext)s"
    ]
    subprocess.run(cmd)

print("[*] Dataset scraping complete!")
