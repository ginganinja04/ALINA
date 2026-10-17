"""Report detection frequency and runtime without ground-truth coordinates."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summaries = []
    for identity, group in sorted(groups.items()):
        completed = [row for row in group if row["status"] == "ok"]
        detected = sum(int(row["pixels"]) > 0 for row in completed)
        roi = [float(row["roi_ms"]) for row in completed]
        alina = [float(row["alina_ms"]) for row in completed]
        summaries.append({
            **dict(zip(keys, identity)),
            "frames": len(group), "completed": len(completed),
            "errors": len(group) - len(completed), "detected_frames": detected,
            "detection_rate_pct": 100 * detected / len(group),
            "mean_roi_ms": mean(roi) if roi else "",
            "mean_alina_ms": mean(alina) if alina else "",
            "mean_total_ms": mean([a + b for a, b in zip(roi, alina)]) if roi else "",
        })
    return summaries


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    source = Path(args.results)
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        parser.error("results contain no frames; run the batch experiment first")
    out = Path(args.output_dir) if args.output_dir else source.parent / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    overall = summarize(rows, ("method",))
    by_video = summarize(rows, ("video", "method"))
    write_csv(out / "overall.csv", overall)
    write_csv(out / "by_video.csv", by_video)

    labels = [row["method"] for row in overall]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, [row["detection_rate_pct"] for row in overall])
    ax.set(ylim=(0, 100), ylabel="Recorded frames with detections (%)",
           title="Detection frequency (not accuracy)")
    fig.tight_layout()
    fig.savefig(out / "detection_rate.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    roi = [row["mean_roi_ms"] if row["completed"] else np.nan for row in overall]
    alina = [row["mean_alina_ms"] if row["completed"] else np.nan for row in overall]
    ax.bar(labels, roi, label="ROI selection")
    ax.bar(labels, alina, bottom=roi, label="ALINA")
    ax.set(ylabel="Mean milliseconds per completed frame", title="Processing time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "processing_time.png", dpi=160)
    plt.close(fig)

    videos = sorted({row["video"] for row in by_video})
    lookup = {(row["video"], row["method"]): row for row in by_video}
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels))
    width = .8 / len(videos)
    for index, video in enumerate(videos):
        rates = [lookup.get((video, method), {}).get("detection_rate_pct", np.nan)
                 for method in labels]
        ax.bar(x + (index - (len(videos) - 1) / 2) * width, rates, width, label=video)
    ax.set_xticks(x, labels)
    ax.set(ylim=(0, 100), ylabel="Recorded frames with detections (%)",
           title="Detection frequency by video (not accuracy)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "detection_rate_by_video.png", dpi=160)
    plt.close(fig)
    print(f"Wrote overall.csv, by_video.csv and three charts to {out}")
    print("Detection means at least one output pixel; it does not establish correctness.")


if __name__ == "__main__":
    main()
