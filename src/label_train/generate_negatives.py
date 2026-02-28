"""
Generate negative samples from saved trainingBackup images.

For each image in images/trainingBackup (skipping anchors folder), this script:
- loads the image
- detects the yellow frame via FieldScanner.detect_yellow_frame
- builds the grid via FieldScanner.build_grid
- loads annotated centers from images/labeledImages/calibration/<base>.json (if present)
- treats grid cells without a nearby annotated center as negative samples
- saves negative crops to images/labeledImages/negative/free/

Usage:
    python generate_negatives.py

"""
import os
import glob
import json
import math
import cv2
import numpy as np

from map_scanner import FieldScanner

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUP_DIR = os.path.join(BASE_DIR, "images", "trainingBackup")
CALIB_DIR = os.path.join(BASE_DIR, "images", "labeledImages", "calibration")
NEG_DIR = os.path.join(BASE_DIR, "images", "labeledImages", "negative", "free")

os.makedirs(NEG_DIR, exist_ok=True)

scanner = FieldScanner()


def load_calibration_for(base):
    jf = os.path.join(CALIB_DIR, base + ".json")
    if not os.path.exists(jf):
        return []
    try:
        with open(jf, "r", encoding="utf-8") as f:
            pts = json.load(f)
            return pts
    except Exception:
        return []


def is_close_to_any(x, y, pts, thr):
    for p in pts:
        if "cx" in p and "cy" in p:
            dx = x - float(p["cx"])
            dy = y - float(p["cy"])
            if dx*dx + dy*dy <= thr*thr:
                return True
    return False


def main():
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    imgs = sorted(p for p in glob.glob(os.path.join(BACKUP_DIR, "*")) if os.path.splitext(p)[1].lower() in exts)
    count_saved = 0
    for img_path in imgs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        # skip anchor crops folder files
        if base.endswith("_ANCHOR"):
            continue
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] failed to read {img_path}")
            continue
        h, w = img.shape[:2]
        # try detect yellow frame using center click
        cx, cy = w//2, h//2
        fw, fh, frame = scanner.detect_yellow_frame(img, cx, cy)
        if frame is None:
            print(f"[SKIP] no anchor detected for {base}")
            continue
        anchor_fx, anchor_fy, _, _ = frame
        grid = scanner.build_grid(anchor_fx, anchor_fy, fw, fh, 0, 0, w, h)
        centers = [(tx + fw//2, ty + fh//2) for (tx, ty, fw, fh) in grid]
        annotated = load_calibration_for(base)
        # threshold: half of min(fw,fh)
        thr = min(fw, fh) * 0.6
        for i, (tx, ty, gw, gh) in enumerate(grid):
            cx_cell = tx + gw//2
            cy_cell = ty + gh//2
            if is_close_to_any(cx_cell, cy_cell, annotated, thr):
                continue
            # save crop
            x1, y1 = int(tx), int(ty)
            x2, y2 = int(tx + gw), int(ty + gh)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            fname = os.path.join(NEG_DIR, f"{base}_neg_{i}.png")
            cv2.imencode('.png', crop)[1].tofile(fname)
            count_saved += 1
        print(f"[DONE] {base}: saved negatives = {count_saved}")
    print(f"Total negatives saved: {count_saved}")


if __name__ == '__main__':
    main()
