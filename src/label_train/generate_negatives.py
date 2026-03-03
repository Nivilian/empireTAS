"""
Generate negative samples from saved trainingBackup images.

For each image in images/trainingBackup (skipping anchors folder), this script:
- loads the image
- detects the yellow frame via FieldScanner.detect_yellow_frame
- builds the grid via FieldScanner.build_grid
- saves each grid cell as a negative sample (we no longer rely on calibration JSONs)
- saves negative crops to images/labeledImages/negative/free/

Usage:
    python generate_negatives.py

"""
import os
import glob
import math
import cv2
import numpy as np
import shutil

from map_scanner import FieldScanner

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUP_DIR = os.path.join(BASE_DIR, "images", "trainingBackup")
NEG_DIR = os.path.join(BASE_DIR, "images", "labeledImages", "negative", "free")

os.makedirs(NEG_DIR, exist_ok=True)

scanner = FieldScanner()


def main():
    # remove legacy calibration dir if present (user requested deletion)
    cal_dir = os.path.join(BASE_DIR, "images", "labeledImages", "calibration")
    if os.path.isdir(cal_dir):
        try:
            shutil.rmtree(cal_dir)
            print(f"Removed legacy calibration directory: {cal_dir}")
        except Exception as ex:
            print(f"Failed to remove calibration dir: {ex}")
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
        # Save every grid cell as a negative candidate (no calibration filtering)
        for i, (tx, ty, gw, gh) in enumerate(grid):
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
