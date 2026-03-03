"""
MapScanner
==========
Scans the full isometric game map by sliding the viewport across the
2.8 × 2.7 screen-sized world, collecting tile information at each stop.

Usage
-----
    calibration = {
        "tile_a": {"game": (0, 0),   "screen": (px_x, px_y)},
        "tile_b": {"game": (20, 20), "screen": (px_x, px_y)},
    }
    scanner = MapScanner(window_title="MuMu安卓", calibration=calibration)
    results  = scanner.scan_full_map()
    # results: { "15/25": {"zone": 1097, "type": "大森林",
    #                       "status": "occupied", "alliance": "ABC"} }

Coordinate system
-----------------
The game uses an isometric (diamond) grid.
Tile (col, row) in game coords maps to screen pixel (px, py) via:

    px = origin_x + (col - row) * half_tile_w
    py = origin_y + (col + row) * half_tile_h

Inverse (pixel → tile):
    col = ( (px - origin_x) / half_tile_w + (py - origin_y) / half_tile_h ) / 2
    row = ( (py - origin_y) / half_tile_h - (px - origin_x) / half_tile_w ) / 2

Calibration derives origin_x/y and half_tile_w/h from two known points.
"""

import os
import sys
import time
import math
from collections import deque

import cv2
import numpy as np
import pyautogui
import win32gui

from vision_core import capture_game_ignore_ui, find_hwnd

__all__ = ["calibrate", "FieldScanner"]



# ---------------------------------------------------------------------------
# Calibration helper
# ---------------------------------------------------------------------------

def calibrate(tile_a_game, tile_a_screen, tile_b_game, tile_b_screen):
    """
    Derive isometric grid parameters from two (game_coord, screen_pixel) pairs.

    Returns a dict:
        { origin_x, origin_y, half_tile_w, half_tile_h }
    """
    ca, sa = tile_a_game, tile_a_screen   # (col_a, row_a), (px_a, py_a)
    cb, sb = tile_b_game, tile_b_screen

    # delta in game space
    d_col = cb[0] - ca[0]
    d_row = cb[1] - ca[1]

    # delta in screen space
    d_px = sb[0] - sa[0]
    d_py = sb[1] - sa[1]

    # Solve: d_px = (d_col - d_row) * half_w
    #        d_py = (d_col + d_row) * half_h
    iso_x = d_col - d_row   # coefficient for half_w
    iso_y = d_col + d_row   # coefficient for half_h

    half_tile_w = d_px / iso_x if iso_x != 0 else 1
    half_tile_h = d_py / iso_y if iso_y != 0 else 1

    origin_x = sa[0] - (ca[0] - ca[1]) * half_tile_w
    origin_y = sa[1] - (ca[0] + ca[1]) * half_tile_h

    return {
        "origin_x":    origin_x,
        "origin_y":    origin_y,
        "half_tile_w": half_tile_w,
        "half_tile_h": half_tile_h,
    }


# ---------------------------------------------------------------------------
# MapScanner class
# ---------------------------------------------------------------------------


# NOTE: MapScanner (full map scanning utilities) were trimmed from this
# refactor — this module now focuses on `FieldScanner` for grid/anchor
# detection and small helpers. The full MapScanner implementation can be
# restored later if needed.


# ---------------------------------------------------------------------------
# FieldScanner class
# ---------------------------------------------------------------------------

class FieldScanner:
    """
    Handles field / resource-tile scanning on the isometric map.

    Workflow
    --------
    1. Click the screen centre → detect the yellow selection diamond.
    2. If grid size is not yet calibrated, click one tile to the left,
       detect the second diamond, compute the tile-to-tile distance, and
       lock fw / fh permanently.
    3. BFS-expand the grid from the detected anchor tile to cover the full
       screen.
    4. Template-match resource types at every grid node.

    Parameters
    ----------
    window_title : str   Partial emulator window title.
    grid_fw      : int   Locked horizontal diagonal (px). Default 194.
    grid_fh      : int   Locked vertical diagonal  (px). Default 97.
    threshold    : float Match threshold for resource templates (default 0.3).
    """

    # Reference run: dist=194.74 → fw=194, fh=97
    DEFAULT_FW = 194
    DEFAULT_FH = 97

    # Known tile size candidates (fw, fh) — keep these hardcoded reference sizes
    KNOWN_TILE_SIZES = [
        (194, 97),
        (180, 90),
        (200,100),
    ]

    # (classification removed) — FieldScanner only provides grid detection/drawing

    @staticmethod
    def draw_grid(image, grid_cells, color=(55, 55, 55), thickness=1):
        """Draw isometric diamond grid on `image` for given `grid_cells` list.
        grid_cells: list of (tx, ty, gw, gh)
        """
        for (tx, ty, gw, gh) in grid_cells:
            top    = (tx + gw // 2, ty)
            right  = (tx + gw,      ty + gh // 2)
            bottom = (tx + gw // 2, ty + gh)
            left   = (tx,           ty + gh // 2)
            cv2.line(image, top,    right,  color, thickness)
            cv2.line(image, right,  bottom, color, thickness)
            cv2.line(image, bottom, left,   color, thickness)
            cv2.line(image, left,   top,    color, thickness)
        return image

    def __init__(self, window_title="MuMu安卓", grid_fw=None, grid_fh=None, threshold=0.18):
        self.window_title = window_title
        self.grid_fw      = grid_fw if grid_fw is not None else self.DEFAULT_FW
        self.grid_fh      = grid_fh if grid_fh is not None else self.DEFAULT_FH
        self.threshold    = threshold
        self._base        = os.path.dirname(os.path.abspath(__file__))

        # 仅保留截图和锚定grid相关功能，不加载分类器

    # ------------------------------------------------------------------ #
    #  Yellow diamond detection                                            #
    # ------------------------------------------------------------------ #

    def detect_yellow_frame(self, screen, click_x=None, click_y=None):
        """
        Detect the yellow tile-selection diamond (#FDEB04).

        If grid size is already known, uses fast synthetic-template matching
        on the HSV colour mask — robust against partial occlusion.
        Falls back to contour detection when grid size is unknown.

        Returns (tile_w, tile_h, (x, y, w, h)) in full-screen coords,
        or (None, None, None) if not found.
        """
        sh, sw     = screen.shape[:2]
        BASE_W, BASE_H = 1718, 966

        if click_x is None: click_x = sw // 2
        if click_y is None: click_y = sh // 2

        roi_hw = int(200 * sw / BASE_W)
        roi_hh = int(120 * sh / BASE_H)
        rx1 = max(0, click_x - roi_hw)
        ry1 = max(0, click_y - roi_hh)
        rx2 = min(sw, click_x + roi_hw)
        ry2 = min(sh, click_y + roi_hh)
        roi  = screen[ry1:ry2, rx1:rx2]
        print(f"  [TileFrame] ROI  x={rx1}-{rx2}  y={ry1}-{ry2}  ({rx2-rx1}x{ry2-ry1})")

        # HSV colour filter — #FDEB04 ≈ H28 S251 V253
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([22, 180, 180]), np.array([35, 255, 255]))

        debug_dir = os.path.join(self._base, "images", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "yellow_mask.png"), mask)
        cv2.imwrite(os.path.join(debug_dir, "roi_capture.png"), roi)
        print(f"  [TileFrame] debug images → {debug_dir}")

        # --- Synthetic template matching (occlusion-resistant) ---
        # Try locked grid first; if not set, try known candidate sizes.
        candidates = []
        size_sources = []
        if self.grid_fw and self.grid_fh:
            size_sources.append((self.grid_fw, self.grid_fh))
        else:
            size_sources.extend(self.KNOWN_TILE_SIZES)

        for fw, fh in size_sources:
            print(f"  [TileFrame] Synthetic template match (fw={fw}, fh={fh})")
            tpl  = np.zeros((fh, fw), dtype=np.uint8)
            pts  = np.array([[fw//2, 0], [fw-1, fh//2],
                              [fw//2, fh-1], [0, fh//2]], np.int32)
            cv2.polylines(tpl, [pts], isClosed=True, color=255, thickness=2)
            if mask.shape[0] >= fh and mask.shape[1] >= fw:
                res = cv2.matchTemplate(mask, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                print(f"  [TileFrame] Synthetic match score: {max_val:.3f} for {fw}x{fh}")
                if max_val > 0.15:
                    lx, ly = max_loc
                    return fw, fh, (rx1 + lx, ry1 + ly, fw, fh)
                candidates.append((max_val, fw, fh, max_loc))
        print("  [TileFrame] No synthetic match above threshold — falling back to contour method")

        # --- Contour fallback (used when grid_fw/fh not yet known) ---
        KSIZE  = 9; KHALF = KSIZE // 2
        kernel = np.ones((KSIZE, KSIZE), np.uint8)
        mask_d = cv2.dilate(mask, kernel, iterations=1)
        cnts, _ = cv2.findContours(mask_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            print("  [TileFrame] no yellow contours in ROI")
            return None, None, None

        MIN_W, MAX_W = 80, 360
        MIN_H, MAX_H = 45, 200
        print(f"  [TileFrame] {len(cnts)} contours, filtering...")
        candidates = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 300: continue
            x, y, w, h = cv2.boundingRect(c)
            aspect    = w / h if h > 0 else 0
            fill      = area / (w * h) if w * h > 0 else 1.0
            size_ok   = MIN_W <= w <= MAX_W and MIN_H <= h <= MAX_H
            asp_ok    = 1.1 <= aspect <= 2.8
            fill_ok   = fill < 0.80
            print(f"    contour  xy=({x},{y})  wh=({w},{h})  aspect={aspect:.2f}"
                  f"  fill={fill:.2f}  size={'OK' if size_ok else 'FAIL'}"
                  f"  asp={'OK' if asp_ok else 'FAIL'}  fill={'OK' if fill_ok else 'FAIL'}")
            if size_ok and asp_ok and fill_ok:
                candidates.append((area, x, y, w, h))

        if not candidates:
            print(f"  [TileFrame] no valid diamond")
            return None, None, None

        _, lx, ly, lw, lh = max(candidates, key=lambda c: c[0])
        lx += KHALF; ly += KHALF
        lw = max(1, lw - KSIZE); lh = max(1, lh - KSIZE)
        return lw, lh, (rx1 + lx, ry1 + ly, lw, lh)

    # ------------------------------------------------------------------ #
    #  Grid BFS                                                            #
    # ------------------------------------------------------------------ #

    def build_grid(self, anchor_fx, anchor_fy, fw, fh,
                   bound_x1=0, bound_y1=0, bound_x2=None, bound_y2=None):
        """
        BFS-expand an isometric diamond grid from one anchor tile corner.
        Only cells whose bounding rect overlaps the content area
        [bound_x1:bound_x2, bound_y1:bound_y2] are kept.
        Returns list of (tx, ty, fw, fh).
        """
        if bound_x2 is None: bound_x2 = 99999
        if bound_y2 is None: bound_y2 = 99999
        htw  = fw // 2
        hth  = fh // 2
        DIRS = [(+htw, -hth), (+htw, +hth), (-htw, +hth), (-htw, -hth)]
        visited, cells, queue = set(), [], deque()
        start = (anchor_fx, anchor_fy)
        visited.add(start)
        queue.append(start)
        while queue:
            tx, ty = queue.popleft()
            # Skip if cell is entirely outside the content area
            if tx + fw <= bound_x1 or ty + fh <= bound_y1 or tx >= bound_x2 or ty >= bound_y2:
                continue
            cells.append((tx, ty, fw, fh))
            for dx, dy in DIRS:
                nx, ny = tx + dx, ty + dy
                key = (nx, ny)
                if key not in visited:
                    # Allow one tile of expansion beyond bounds to avoid edge gaps
                    if (bound_x1 - fw) < nx < bound_x2 and (bound_y1 - fh) < ny < bound_y2:
                        visited.add(key)
                        queue.append((nx, ny))
        return cells

    # ------------------------------------------------------------------ #
    #  Main scan entry-point                                               #
    # ------------------------------------------------------------------ #

    def scan(self, templates=None, status_cb=None):
        """
        Full field scan.

        Parameters
        ----------
        templates : dict | None
            { display_name: absolute_image_path }  — resource templates.
            Defaults to 空粘土 and 被占领粘土.
        status_cb : callable | None
            Optional callback(str) for progress messages (e.g. overlay label).

        Returns
        -------
        dict  { name: [(tx, ty, fw, fh, score), ...] }
        """
        def _status(msg):
            print(f"  [FieldScanner] {msg}")
            if status_cb: status_cb(msg)

        if templates is None:
            res_dir   = os.path.join(self._base, "images", "resourceField")
            templates = {
                "空粘土":        os.path.join(res_dir, "空粘土.png"),
                "被联盟占领的粘土": os.path.join(res_dir, "被联盟占领的粘土.png"),
                "被散人占领的粘土": os.path.join(res_dir, "被散人占领的粘土.png"),
            }

        hwnd = find_hwnd(self.window_title)
        if not hwnd:
            _status("no window"); return {}

        # 1. Click screen centre → first yellow frame
        rect  = win32gui.GetClientRect(hwnd)
        cx, cy = (rect[2] - rect[0]) // 2, (rect[3] - rect[1]) // 2
        pt     = win32gui.ClientToScreen(hwnd, (cx, cy))
        pyautogui.click(pt[0], pt[1]); time.sleep(0.6)

        screen = capture_game_ignore_ui(self.window_title)
        if screen is None: _status("no capture"); return {}
        sh, sw = screen.shape[:2]

        tw1, th1, frame1 = self.detect_yellow_frame(screen, cx, cy)
        print(f"  [TileFrame1] tile_w={tw1}  tile_h={th1}  rect={frame1}")
        if frame1 is None: _status("no first frame"); return {}

        fx1, fy1, fw1, fh1 = frame1
        c1x, c1y = fx1 + fw1 // 2, fy1 + fh1 // 2

        # 2. If grid size unknown, click left-neighbour tile to calibrate
        if self.grid_fw is None or self.grid_fh is None:
            t2cx, t2cy  = c1x - fw1, c1y
            pt2 = win32gui.ClientToScreen(hwnd, (t2cx, t2cy))
            pyautogui.click(pt2[0], pt2[1]); time.sleep(0.6)

            screen2 = capture_game_ignore_ui(self.window_title)
            if screen2 is None: _status("no second capture"); return {}

            tw2, th2, frame2 = self.detect_yellow_frame(screen2, t2cx, t2cy)
            print(f"  [TileFrame2] tile_w={tw2}  tile_h={th2}  rect={frame2}")
            if frame2 is None: _status("no second frame"); return {}

            fx2, fy2, fw2, fh2 = frame2
            c2x, c2y = fx2 + fw2 // 2, fy2 + fh2 // 2

            dist = float(np.sqrt((c1x - c2x)**2 + (c1y - c2y)**2))
            print(f"  [Distance] c1=({c1x},{c1y})  c2=({c2x},{c2y})  dist={dist:.2f}")

            fw = int(round(dist))
            if fw % 2: fw -= 1
            fh = fw // 2
            self.grid_fw, self.grid_fh = fw, fh
            print(f"  [TileFrame] calibrated → fw={fw}  fh={fh}")

            anchor_fx = c2x - fw // 2
            anchor_fy = c2y - fh // 2
            work_screen = screen2
        else:
            fw, fh = self.grid_fw, self.grid_fh
            print(f"  [TileFrame] locked grid → fw={fw}  fh={fh}")
            anchor_fx = c1x - fw // 2
            anchor_fy = c1y - fh // 2
            work_screen = screen

        # 3. Build grid — confined to work_screen bounds (already border-cropped)
        wsh, wsw = work_screen.shape[:2]
        grid_cells = self.build_grid(anchor_fx, anchor_fy, fw, fh,
                                     0, 0, wsw, wsh)
        print(f"  [Grid] BFS generated {len(grid_cells)} cells  screen=({wsw}x{wsh})")

        # 4. Draw visualisation
        viz = work_screen.copy()
        # (anchor rectangle intentionally not drawn)
        cv2.putText(viz, f"ref {fw}x{fh}", (anchor_fx, max(0, anchor_fy - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for (tx, ty, gw, gh) in grid_cells:
            top    = (tx + gw // 2, ty)
            right  = (tx + gw,      ty + gh // 2)
            bottom = (tx + gw // 2, ty + gh)
            left   = (tx,           ty + gh // 2)
            cv2.line(viz, top,    right,  (55, 55, 55), 1)
            cv2.line(viz, right,  bottom, (55, 55, 55), 1)
            cv2.line(viz, bottom, left,   (55, 55, 55), 1)
            cv2.line(viz, left,   top,    (55, 55, 55), 1)

        # 5. Classification removed — return empty lists for template keys only.
        # Keep compatibility with callers that expect keys "空粘土" and "被占领粘土".
        results = {
            "空粘土": [],
            "被联盟占领的粘土": [],
            "被散人占领的粘土": [],
            # compatibility key used by main.test_field
            "被占领粘土": [],
        }

        print("\n--- Field Scan Result ---")
        for name, hits in results.items():
            print(f"  {name:10s}: {'yes' if hits else 'no'}  ({len(hits)} hits above {self.threshold})")
        print("-------------------------")

        cv2.imshow("Field Test", viz)
        cv2.waitKey(1)
        return results

    def browse_and_predict(self, target_title=None, model_path=None, torch_model_path=None,
                           show_win_name="Map Predictions", status_cb=None, conf_thresh=0.15):
        """Trigger center click to reveal yellow anchor, build grid, run classifier on each cell
        and show an overlay window with predicted terrain+occupation.

        Parameters
        - target_title: emulator window title (falls back to self.window_title)
        - model_path: path to baseline .npz model
        - torch_model_path: path to .pth torch model
        - status_cb: optional callable(str) to receive status updates
        - conf_thresh: minimal confidence to visualize
        """
        def _status(s):
            print(f"  [FieldScanner] {s}")
            if status_cb:
                try:
                    status_cb(s)
                except Exception:
                    pass

        title = target_title or self.window_title
        hwnd = find_hwnd(title)
        if not hwnd:
            _status("no window found")
            return None

        # click center (with small up-left shift similar to other callers)
        rect = win32gui.GetClientRect(hwnd)
        cx, cy = (rect[2] - rect[0]) // 2, (rect[3] - rect[1]) // 2
        try:
            fw, fh = self.grid_fw, self.grid_fh
            if fw and fh:
                s = math.hypot(fw / 2.0, fh / 2.0)
                shift = int(round(0.1 * s / math.sqrt(2)))
            else:
                shift = 0
        except Exception:
            shift = 0
        click_x = cx - shift
        click_y = cy - shift
        pt = win32gui.ClientToScreen(hwnd, (click_x, click_y))
        pyautogui.click(pt[0], pt[1])
        time.sleep(0.6)

        screen = capture_game_ignore_ui(title)
        if screen is None:
            _status("capture failed")
            return None
        img = screen.copy()

        # detect anchor, try fallback click sequence like other callers
        fw, fh = self.grid_fw, self.grid_fh
        _, _, anchor = self.detect_yellow_frame(img, cx, cy)
        if anchor is None:
            seq = [(-fw, 0), (0, -fh), (fw, 0), (fw, 0), (0, fh), (0, fh),
                   (-fw, 0), (-fw, 0), (0, -fh), (0, -fh), (fw, 0), (fw, 0)]
            cur_x, cur_y = cx, cy
            for dx, dy in seq:
                cur_x += dx; cur_y += dy
                ptc = win32gui.ClientToScreen(hwnd, (int(cur_x), int(cur_y)))
                pyautogui.click(ptc[0], ptc[1]); time.sleep(0.5)
                screen2 = capture_game_ignore_ui(title)
                if screen2 is None: continue
                _, _, anchor = self.detect_yellow_frame(screen2, cur_x, cur_y)
                if anchor:
                    img = screen2.copy()
                    break

        if anchor is None:
            h, w = img.shape[:2]
            anchor_fx = w // 2 - (fw // 2 if fw else self.DEFAULT_FW//2)
            anchor_fy = h // 2 - (fh // 2 if fh else self.DEFAULT_FH//2)
        else:
            anchor_fx, anchor_fy, _, _ = anchor
            try:
                if fw and fh:
                    s = math.hypot(fw / 2.0, fh / 2.0)
                    shift = int(round(0.1 * s / math.sqrt(2)))
                else:
                    shift = 0
            except Exception:
                shift = 0
            anchor_fx -= shift; anchor_fy -= shift

        h, w = img.shape[:2]
        grid_cells = self.build_grid(anchor_fx, anchor_fy, fw, fh, 0, 0, w, h)
        _status(f"built grid with {len(grid_cells)} cells")

        # load classifier: prefer torch, fallback to baseline
        clf = None
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            if torch_model_path is None:
                torch_model_path = os.path.join(self._base, "resourcefield_model.pth")
            if model_path is None:
                model_path = os.path.join(self._base, "resourcefield_model.npz")
            try:
                from label_train.torch_classifier import TorchResourceFieldClassifier
                tclf = TorchResourceFieldClassifier()
                if os.path.exists(torch_model_path) and tclf.load(torch_model_path):
                    clf = tclf
            except Exception:
                pass
            if clf is None and os.path.exists(model_path):
                from label_train.resourcefield_classifier import ResourceFieldClassifier
                rclf = ResourceFieldClassifier()
                if rclf.load(model_path):
                    clf = rclf
        except Exception:
            clf = None

        terrain_to_bgr = {
            'clay':   (40, 100, 150),
            'forest': (60, 200, 60),
            'boat':   (220, 120, 40),
            'copper': (60, 60, 220),
            'stone':  (200, 200, 200),
        }

        out = img.copy()
        for (tx, ty, gw, gh) in grid_cells:
            x1, y1, x2, y2 = int(tx), int(ty), int(tx+gw), int(ty+gh)
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue
            crop = img[y1:y2, x1:x2].copy()
            if crop.size == 0: continue
            pred_name, conf = (None, 0.0)
            if clf is not None:
                ch, cw = crop.shape[:2]
                if cw > 8 and ch > 8:
                    mask = np.zeros((ch, cw), dtype=np.uint8)
                    poly = np.array([[(cw // 2, 0), (cw - 1, ch // 2), (cw // 2, ch - 1), (0, ch // 2)]], dtype=np.int32)
                    cv2.fillPoly(mask, poly, 255)
                    bg = crop.mean(axis=(0,1)).astype(np.uint8)
                    bg_img = np.zeros_like(crop); bg_img[:,:] = bg
                    masked = np.where(mask[:,:,None] == 255, crop, bg_img)
                    pred_name, conf = clf.predict(masked)

            terrain_name = None; occ_name = 'free'
            if isinstance(pred_name, str) and pred_name:
                parts = pred_name.split('_')
                terrain_name = parts[0]
                if len(parts) > 1: occ_name = parts[1]

            if terrain_name in terrain_to_bgr and conf >= conf_thresh:
                color = terrain_to_bgr[terrain_name]
                pts = np.array([(x1 + gw//2, y1), (x2, y1 + gh//2), (x1 + gw//2, y2), (x1, y1 + gh//2)], dtype=np.int32)
                overlay = out.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.28, out, 1 - 0.28, 0, out)
                cv2.polylines(out, [pts], True, color, 2)
                cx = x1 + gw//2; cy = y1 + gh//2
                label_str = f"{terrain_name}, {occ_name} {int(conf*100)}%"
                cv2.putText(out, label_str, (cx - gw//4, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
                cv2.putText(out, label_str, (cx - gw//4, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imshow(show_win_name, out)
        cv2.waitKey(1)
        _status("browse_and_predict done")
        return out
