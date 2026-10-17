"""Render saved ALINA detections and predicted ROI overlays on original frames.

Example:
  python -m experiments.render_overlays --root outputs/final_dynamic \
    --video vidd_1 --methods hsv hough consensus --frames 00002 00004
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
from .roi_methods import MANUAL, METHODS, build
from .run_batch import VIDEOS


def load_coords(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 2), dtype=np.int32)
    values = np.loadtxt(path, dtype=np.int32)
    return np.atleast_2d(values)


def draw_roi(image: np.ndarray, corners: np.ndarray, color: tuple[int, int, int], label: str) -> None:
    points = np.round(corners).astype(np.int32)
    cv2.polylines(image, [points], True, color, 3, cv2.LINE_AA)
    x, y = points[1]
    cv2.putText(image, label, (int(x), int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="batch output root containing results.csv")
    ap.add_argument("--video", required=True, choices=VIDEOS)
    ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    ap.add_argument("--frames", nargs="+", required=True, help="frame stems, e.g. 00002 00004")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--no-manual", action="store_true", help="do not draw the manual reference ROI")
    args = ap.parse_args()
    root = Path(args.root)
    output = Path(args.output_dir) if args.output_dir else root / "overlays" / args.video
    predictors = {method: build(method, args.video) for method in args.methods}
    for stem in args.frames:
        image_path = VIDEOS[args.video] / f"{stem}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping missing image: {image_path}")
            continue
        for method, predictor in predictors.items():
            canvas = image.copy()
            roi = predictor.predict(image)
            draw_roi(canvas, roi.corners(), (0, 255, 0), "predicted ROI")
            if not args.no_manual:
                draw_roi(canvas, MANUAL[args.video].corners(), (255, 0, 255), "manual ROI")
            coords = load_coords(root / args.video / method / "coords" / f"{stem}.txt")
            # Red pixels are saved ALINA detections. Drawing individual points
            # avoids changing their shape or implying a segmentation mask.
            for x, y in coords:
                if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                    canvas[y, x] = (0, 0, 255)
            cv2.putText(canvas, f"{method}: {len(coords)} detected pixels", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, f"{method}: {len(coords)} detected pixels", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 1, cv2.LINE_AA)
            destination = output / method / f"{stem}_overlay.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destination), canvas)
            print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
