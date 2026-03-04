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

import datetime
import os
import sys
import time
import math
from collections import deque

import cv2
import numpy as np
import pyautogui
import win32gui

from vision_core import capture_game_ignore_ui, find_hwnd, get_ocr_engine

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
        self.global_list  = []   # accumulated across map_scan_full runs within one session

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
        # print(f"  [TileFrame] ROI  x={rx1}-{rx2}  y={ry1}-{ry2}  ({rx2-rx1}x{ry2-ry1})")

        # HSV colour filter — #FDEB04 ≈ H28 S251 V253
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([22, 180, 180]), np.array([35, 255, 255]))

        debug_dir = os.path.join(self._base, "images", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "yellow_mask.png"), mask)
        cv2.imwrite(os.path.join(debug_dir, "roi_capture.png"), roi)
        # print(f"  [TileFrame] debug images → {debug_dir}")

        # --- Synthetic template matching (occlusion-resistant) ---
        # Try locked grid first; if not set, try known candidate sizes.
        candidates = []
        size_sources = []
        if self.grid_fw and self.grid_fh:
            size_sources.append((self.grid_fw, self.grid_fh))
        else:
            size_sources.extend(self.KNOWN_TILE_SIZES)

        for fw, fh in size_sources:
            # print(f"  [TileFrame] Synthetic template match (fw={fw}, fh={fh})")
            tpl  = np.zeros((fh, fw), dtype=np.uint8)
            pts  = np.array([[fw//2, 0], [fw-1, fh//2],
                              [fw//2, fh-1], [0, fh//2]], np.int32)
            cv2.polylines(tpl, [pts], isClosed=True, color=255, thickness=2)
            if mask.shape[0] >= fh and mask.shape[1] >= fw:
                res = cv2.matchTemplate(mask, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                # print(f"  [TileFrame] Synthetic match score: {max_val:.3f} for {fw}x{fh}")
                if max_val > 0.15:
                    lx, ly = max_loc
                    return fw, fh, (rx1 + lx, ry1 + ly, fw, fh)
                candidates.append((max_val, fw, fh, max_loc))
        # print("  [TileFrame] No synthetic match above threshold — falling back to contour method")

        # --- Contour fallback (used when grid_fw/fh not yet known) ---
        KSIZE  = 9; KHALF = KSIZE // 2
        kernel = np.ones((KSIZE, KSIZE), np.uint8)
        mask_d = cv2.dilate(mask, kernel, iterations=1)
        cnts, _ = cv2.findContours(mask_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            # print("  [TileFrame] no yellow contours in ROI")
            return None, None, None

        MIN_W, MAX_W = 80, 360
        MIN_H, MAX_H = 45, 200
        # print(f"  [TileFrame] {len(cnts)} contours, filtering...")
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
            # print(f"    contour  xy=({x},{y})  wh=({w},{h})  aspect={aspect:.2f}"
            #       f"  fill={fill:.2f}  size={'OK' if size_ok else 'FAIL'}"
            #       f"  asp={'OK' if asp_ok else 'FAIL'}  fill={'OK' if fill_ok else 'FAIL'}")
            if size_ok and asp_ok and fill_ok:
                candidates.append((area, x, y, w, h))

        if not candidates:
            # print(f"  [TileFrame] no valid diamond")
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
            # print(f"  [FieldScanner] {msg}")
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
        # print(f"  [TileFrame1] tile_w={tw1}  tile_h={th1}  rect={frame1}")
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
            # print(f"  [TileFrame2] tile_w={tw2}  tile_h={th2}  rect={frame2}")
            if frame2 is None: _status("no second frame"); return {}

            fx2, fy2, fw2, fh2 = frame2
            c2x, c2y = fx2 + fw2 // 2, fy2 + fh2 // 2

            dist = float(np.sqrt((c1x - c2x)**2 + (c1y - c2y)**2))
            print(f"  [Distance] c1=({c1x},{c1y})  c2=({c2x},{c2y})  dist={dist:.2f}")

            fw = int(round(dist))
            if fw % 2: fw -= 1
            fh = fw // 2
            self.grid_fw, self.grid_fh = fw, fh
            # print(f"  [TileFrame] calibrated → fw={fw}  fh={fh}")

            anchor_fx = c2x - fw // 2
            anchor_fy = c2y - fh // 2
            work_screen = screen2
        else:
            fw, fh = self.grid_fw, self.grid_fh
            # print(f"  [TileFrame] locked grid → fw={fw}  fh={fh}")
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

    # ------------------------------------------------------------------
    # Location OCR helper (shared by alliance_detection + test_location)
    # ------------------------------------------------------------------
    def _read_location(self, title):
        """Return (zone_str, coord_str) via OCR. Never raises."""
        import re
        try:
            screen = capture_game_ignore_ui(title)
            if screen is None:
                return "????", "?/?"
            raw_h, raw_w = screen.shape[:2]
            tgt_h = int(round(raw_w * 966.0 / 1718.0))
            ct    = max(0, raw_h - tgt_h)
            img   = screen[ct:, :, :]
            ih, iw = img.shape[:2]
            scale  = iw / 1718.0

            def _ocr(patch, allowlist):
                if patch.size == 0: return ""
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                big  = cv2.resize(gray, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                reader = get_ocr_engine()
                if reader is None: return ""
                return "".join(reader.readtext(big, detail=0, allowlist=allowlist))

            # Zone (top-centre, 88×35 px)
            hw_a = int(round(88 * scale / 2))
            hh_a = max(1, int(round(35 * scale / 2)))
            ax1, ay2 = max(0, iw//2 - hw_a), min(ih, hh_a * 2)
            ax2 = min(iw, iw//2 + hw_a)
            raw_a = _ocr(img[0:ay2, ax1:ax2], '0123456789')
            z = re.findall(r'\d{4}', raw_a)
            zone_str = z[0] if z else "????"

            # Coords (bottom-left + right 814 + up 26, 61×20 px)
            hw_b = int(round(61 * scale / 2))
            hh_b = max(1, int(round(20 * scale / 2)))
            cx_b = int(round(814 * scale))
            cy_b = ih - int(round(26 * scale))
            bx1, by1 = max(0, cx_b - hw_b), max(0, cy_b - hh_b)
            bx2, by2 = min(iw, cx_b + hw_b), min(ih, cy_b + hh_b)
            raw_b = _ocr(img[by1:by2, bx1:bx2], '0123456789/')
            m = re.search(r'(\d{1,3})\s*/\s*(\d{1,3})', raw_b)
            coord_str = f"{m.group(1)}/{m.group(2)}" if m else "?/?"

            return zone_str, coord_str
        except Exception:
            return "????", "?/?"

    # ------------------------------------------------------------------
    # Alliance region template matching
    # ------------------------------------------------------------------
    # Search ROI reference dimensions (must be >= largest template 105x185)
    _ALLIANCE_ROI_W = 220   # reference px (1718-wide screen)
    _ALLIANCE_ROI_H = 280
    _ALLIANCE_KNOWN_THRESH = 0.7  # score threshold for a "known" match

    def _alliance_search_roi(self, img):
        """Return (x1,y1,x2,y2) of the alliance info-panel search area."""
        ih, iw = img.shape[:2]
        scale  = iw / 1718.0
        cx = int(round(1641 * scale))
        cy = ih - int(round(644 * scale))
        hw = int(round(self._ALLIANCE_ROI_W * scale / 2))
        hh = int(round(self._ALLIANCE_ROI_H * scale / 2))
        return (max(0, cx - hw), max(0, cy - hh),
                min(iw, cx + hw), min(ih, cy + hh))

    def _match_alliance_region(self, window_title):
        """Capture current screen, extract the alliance info-panel search area, and
        template-match against every image in images/alliance/.

        Templates are scaled proportionally to the current display resolution
        and matched with sliding-window matchTemplate + minMaxLoc.

        Returns
        -------
        (best_name, best_score, results)
        """
        import glob

        screen = capture_game_ignore_ui(window_title)
        if screen is None:
            return None, 0.0, []

        raw_h, raw_w = screen.shape[:2]
        tgt_h = int(round(raw_w * 966.0 / 1718.0))
        ct    = max(0, raw_h - tgt_h)
        img   = screen[ct:, :, :]

        ih, iw = img.shape[:2]
        scale  = iw / 1718.0

        x1, y1, x2, y2 = self._alliance_search_roi(img)
        search = img[y1:y2, x1:x2]
        if search.size == 0:
            return None, 0.0, []
        sh, sw = search.shape[:2]

        alliance_dir = os.path.join(self._base, "images", "alliance")
        tmpl_paths   = []
        for ext in ("png", "jpg", "jpeg"):
            tmpl_paths.extend(glob.glob(os.path.join(alliance_dir, f"*.{ext}")))
        if not tmpl_paths:
            return None, 0.0, []

        results = []
        for tp in tmpl_paths:
            name = os.path.splitext(os.path.basename(tp))[0]
            tmpl = cv2.imdecode(np.fromfile(tp, dtype=np.uint8), cv2.IMREAD_COLOR)
            if tmpl is None: continue
            th, tw = tmpl.shape[:2]
            # scale template to match game resolution
            tw_s = max(1, int(round(tw * scale)))
            th_s = max(1, int(round(th * scale)))
            # ensure template fits inside search area
            if tw_s > sw or th_s > sh:
                ratio = min(sw / tw_s, sh / th_s) * 0.95
                tw_s = max(1, int(round(tw_s * ratio)))
                th_s = max(1, int(round(th_s * ratio)))
            tmpl_r = cv2.resize(tmpl, (tw_s, th_s), interpolation=cv2.INTER_AREA)
            res    = cv2.matchTemplate(search, tmpl_r, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            results.append((name, float(max_val)))

        results.sort(key=lambda x: x[1], reverse=True)
        if results:
            return results[0][0], results[0][1], results
        return None, 0.0, []

    def alliance_detection(self, hwnd, client_cx, client_cy, status_cb=None, terrain=None, terrain_conf=1.0):
        """Two-step interaction on a detected alliance tile.

        Step 1: click the diamond centre (client_cx, client_cy).
        Step 2: wait 1 s, then click:
                  bottom midpoint of the full client ROI  +  up 70  +  left 47
                (offsets scaled relative to 1718 reference width).

        Parameters
        ----------
        hwnd       : win32 hwnd of the game window
        client_cx  : tile centre x in client coords
        client_cy  : tile centre y in client coords
        terrain    : optional tile type string appended to result (e.g. 'clay')
        """
        def _s(msg):
            # print(f"  [AllianceDet] {msg}")
            if status_cb:
                try: status_cb(msg)
                except Exception: 
                    return

        rect = win32gui.GetClientRect(hwnd)
        cw, ch = rect[2], rect[3]   # client width / height
        scale  = cw / 1718.0

        # Step 1 — click diamond centre
        pt1 = win32gui.ClientToScreen(hwnd, (client_cx, client_cy))
        _s(f"click 1: diamond centre  client=({client_cx},{client_cy})")
        pyautogui.click(pt1[0], pt1[1])

        # Step 2 — wait 0.3 s, then click bottom-midpoint of full ROI offset up/left
        time.sleep(0.3)
        x2 = cw // 2 - int(round(47 * scale))
        y2 = ch      - int(round(70 * scale))
        pt2 = win32gui.ClientToScreen(hwnd, (x2, y2))
        _s(f"click 2: ROI bottom-mid offset  client=({x2},{y2})")
        pyautogui.click(pt2[0], pt2[1])

        # --- identify which alliance is shown in the info panel ---
        time.sleep(0.5)   # wait for the panel to render
        title = self.window_title
        best_name, best_score, all_results = self._match_alliance_region(title)

        if best_score >= self._ALLIANCE_KNOWN_THRESH and best_name:
            # dismiss click first (closes the panel), then OCR the location bar
            cx_dismiss = cw // 2
            cy_dismiss = ch // 2
            pt_dismiss = win32gui.ClientToScreen(hwnd, (cx_dismiss, cy_dismiss))
            # _s(f"dismiss click: screen centre  client=({cx_dismiss},{cy_dismiss})")
            pyautogui.click(pt_dismiss[0], pt_dismiss[1])
            time.sleep(2.0)  # wait for dismissal

            zone_str, coord_str = self._read_location(title)
            result_line = f"{best_name} ({best_score:.4f}): [{zone_str}, {coord_str}]"
            coor = f"[{zone_str}, {coord_str}]"
            if terrain:
                result_line = f"{result_line}, {terrain} ({terrain_conf:.4f})"
            # print(f"  {result_line}")
            if status_cb:
                try: status_cb(result_line)
                except Exception: pass
            return result_line, coor
        else:
            guess = f"{best_name} {best_score:.4f}" if best_name else "no templates"
            # print(f"  [AllianceDet] unknown alliance  ({guess})")

            # dismiss even on unknown
            cx_dismiss = cw // 2
            cy_dismiss = ch // 2
            pt_dismiss = win32gui.ClientToScreen(hwnd, (cx_dismiss, cy_dismiss))
            pyautogui.click(pt_dismiss[0], pt_dismiss[1])
            return None, None

        # _s("alliance_detection done")

    def swap_to_left(self, target_title=None, status_cb=None):
        """Scroll the map view one page to the left.

        Start position: x = client_width - round(50/1718 * client_width),
                        y = client_height // 2.
        Drag direction: right → left, distance = 8 × fw / 1.3.
        """
        def _status(s):
            # print(f"  [SwapToLeft] {s}")
            if status_cb:
                try: status_cb(s)
                except Exception: pass

        title = target_title or self.window_title
        hwnd  = find_hwnd(title)
        if not hwnd:
            _status("no window found"); return

        fw = self.grid_fw or self.DEFAULT_FW
        drag_dist = int(round(8 * fw / 1.3))
        _status(f"fw={fw}  drag_dist={drag_dist}px")

        rect = win32gui.GetClientRect(hwnd)
        cw, ch = rect[2], rect[3]          # client width / height
        x_client = cw - int(round(50 * cw / 1718))
        y_client = ch // 2
        sx, sy = win32gui.ClientToScreen(hwnd, (x_client, y_client))
        _status(f"start screen=({sx},{sy})  drag left {drag_dist}px")

        pyautogui.moveTo(sx, sy, duration=0.3)
        time.sleep(0.15)
        pyautogui.drag(-drag_dist, 0, duration=1.8, button='left')
        _status("swap done")

    def swap_to_bottom(self, target_title=None, status_cb=None):
        """Scroll the map view downward by dragging from top-centre to bottom-centre.

        Drag distance = 8 × fw / 1.3 × 966 / 1718  (swap_to_left distance × aspect ratio)
        Track:  x = client_width // 2  (horizontal centre)
                start_y = centre - drag // 2,  end_y = centre + drag // 2
        """
        def _status(s):
            # print(f"  [SwapToBottom] {s}")
            if status_cb:
                try: status_cb(s)
                except Exception: pass

        title = target_title or self.window_title
        hwnd  = find_hwnd(title)
        if not hwnd:
            _status("no window found"); return

        fw = self.grid_fw or self.DEFAULT_FW
        drag_dist = int(round(8 * fw / 1.3 * 966 / 1718))
        _status(f"fw={fw}  drag_dist={drag_dist}px")

        rect = win32gui.GetClientRect(hwnd)
        cw, ch = rect[2], rect[3]
        x_client = cw // 2
        y_client  = ch // 2 + drag_dist // 2
        sx, sy = win32gui.ClientToScreen(hwnd, (x_client, y_client))
        _status(f"start screen=({sx},{sy})  drag up {drag_dist}px")

        pyautogui.moveTo(sx, sy, duration=0.3)
        time.sleep(0.15)
        pyautogui.drag(0, -drag_dist, duration=1.8, button='left')
        _status("swap done")

    def swap_to_bottom_right(self, target_title=None, status_cb=None):
        """Scroll the map view toward the top-left by dragging toward bottom-right.

        Long leg  (horizontal) = 8 × fw / 1.3
        Short leg (vertical)   = long_leg × 966 / 1718
        Drag vector: (+long_leg, +short_leg)  i.e. right + down.
        Start position: bottom-left corner + right 400 + up 700 (1718×966 reference)
        """
        def _status(s):
            # print(f"  [SwapToBottomRight] {s}")
            if status_cb:
                try: status_cb(s)
                except Exception: pass

        title = target_title or self.window_title
        hwnd  = find_hwnd(title)
        if not hwnd:
            _status("no window found"); return

        fw = self.grid_fw or self.DEFAULT_FW
        long_leg  = 8 * fw / 1.3
        short_leg = long_leg * 966 / 1718
        dx = int(round(long_leg))
        dy = int(round(short_leg))
        _status(f"fw={fw}  long_leg={dx}  short_leg={dy}")

        rect = win32gui.GetClientRect(hwnd)
        cw, ch = rect[2], rect[3]
        # start: bottom-left + right 400 + up 700 (scaled to current resolution)
        x_client = int(round(400 * cw / 1718))
        y_client  = ch - int(round(750 * ch / 966))
        sx, sy = win32gui.ClientToScreen(hwnd, (x_client, y_client))
        _status(f"start screen=({sx},{sy})  drag (+{dx},+{dy})")

        for i in range(1, 3):
            _status(f"drag {i}/2 ...")
            pyautogui.moveTo(sx, sy, duration=0.3)
            time.sleep(0.15)
            pyautogui.drag(dx, dy, duration=1.8, button='left')
            time.sleep(0.4)
        _status("swap done")

    def test_alliance_region(self, target_title=None, status_cb=None):
        """Capture the 966/1718-cropped ROI and draw the alliance info-panel rectangle:
          - Origin: bottom-left of cropped image
          - Centre:  right 1635 px + up 644 px  (scaled to current resolution)
          - Rect:    width 73 px, height 123 px  (scaled)
        Displays the annotated image in a cv2 window.
        """
        def _s(s):
            print(f"  [TestAllianceRegion] {s}")
            if status_cb:
                try: status_cb(s)
                except Exception: pass

        title = target_title or self.window_title
        screen = capture_game_ignore_ui(title)
        if screen is None:
            _s("capture failed"); return

        # crop to 966/1718 ratio (bottom region)
        raw_h, raw_w = screen.shape[:2]
        tgt_h = int(round(raw_w * 966.0 / 1718.0))
        ct    = max(0, raw_h - tgt_h)
        img   = screen[ct:, :, :].copy()

        ih, iw = img.shape[:2]
        scale  = iw / 1718.0

        # centre relative to bottom-left of cropped image — use shared ROI helper
        x1, y1, x2, y2 = self._alliance_search_roi(img)
        search = img[y1:y2, x1:x2]

        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cx_draw = (x1 + x2) // 2
        cy_draw = (y1 + y2) // 2
        cv2.drawMarker(vis, (cx_draw, cy_draw), (0, 255, 255),
                       cv2.MARKER_CROSS, int(round(20*scale)), 2)
        label = f"alliance-region  ({x1},{y1})-({x2},{y2})  ({x2-x1}×{y2-y1})"
        cv2.putText(vis, label, (max(0, x1), max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(vis, label, (max(0, x1), max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("Test Alliance Region", vis)
        cv2.waitKey(1)
        _s(f"search area: ({x2-x1}×{y2-y1})  scale={scale:.4f}")

        # save the search patch for visual debugging
        debug_dir = os.path.join(self._base, "images", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imencode(".png", search)[1].tofile(os.path.join(debug_dir, "alliance_patch.png"))
        _s("patch saved → images/debug/alliance_patch.png")
        cv2.imshow("Alliance Patch (extracted)", search)
        cv2.waitKey(1)

        # --- template matching against images/alliance/ ---
        import glob
        THRESH = self._ALLIANCE_KNOWN_THRESH
        sh, sw = search.shape[:2]
        alliance_dir = os.path.join(self._base, "images", "alliance")
        tmpl_paths = []
        for ext in ("png", "jpg", "jpeg"):
            tmpl_paths.extend(glob.glob(os.path.join(alliance_dir, f"*.{ext}")))
        if not tmpl_paths:
            _s("no templates in images/alliance/"); return
        scores = []
        best_loc_info = None   # (name, loc, tw_s, th_s) for the best match
        for tp in tmpl_paths:
            name = os.path.splitext(os.path.basename(tp))[0]
            tmpl = cv2.imdecode(np.fromfile(tp, dtype=np.uint8), cv2.IMREAD_COLOR)
            if tmpl is None: continue
            th, tw = tmpl.shape[:2]
            tw_s = max(1, int(round(tw * scale)))
            th_s = max(1, int(round(th * scale)))
            if tw_s > sw or th_s > sh:
                ratio = min(sw / tw_s, sh / th_s) * 0.95
                tw_s = max(1, int(round(tw_s * ratio)))
                th_s = max(1, int(round(th_s * ratio)))
            tmpl_r = cv2.resize(tmpl, (tw_s, th_s), interpolation=cv2.INTER_AREA)
            res    = cv2.matchTemplate(search, tmpl_r, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            scores.append((name, float(max_val), max_loc, tw_s, th_s))
        scores.sort(key=lambda x: x[1], reverse=True)

        # draw best-match location on the saved debug patch
        if scores:
            bname, bval, bloc, btw, bth = scores[0]
            debug_vis = search.copy()
            cv2.rectangle(debug_vis, bloc, (bloc[0]+btw, bloc[1]+bth), (0,255,0), 2)
            cv2.putText(debug_vis, f"{bname} {bval:.3f}", (bloc[0], max(12, bloc[1]-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
            cv2.imencode(".png", debug_vis)[1].tofile(
                os.path.join(debug_dir, "alliance_patch_match.png"))

        sep = "-" * 36
        print(f"  [TestAllianceRegion] search=({sw}×{sh})  templates={len(scores)}")
        print(f"  [TestAllianceRegion] {sep}")
        for name, sc, loc, tw_s, th_s in scores[:5]:
            tag = "✓" if sc >= THRESH else " "
            print(f"  [TestAllianceRegion]  {tag} {name:<22s}  {sc:.4f}  loc={loc}  tmpl=({tw_s}×{th_s})")
        print(f"  [TestAllianceRegion] {sep}")
        if scores and scores[0][1] >= THRESH:
            print(f"  [TestAllianceRegion] matched: {scores[0][0]}  ({scores[0][1]:.4f})")
        else:
            best = f"{scores[0][0]} {scores[0][1]:.4f}" if scores else "none"
            print(f"  [TestAllianceRegion] unknown  (best: {best})")

    def test_alliance_click(self, target_title=None, status_cb=None):
        """Capture screen and draw a marker at the position where alliance_detection
        would place its second click (ROI bottom-midpoint, up 70, left 47).
        Displays the annotated image in a cv2 window for visual verification.
        """
        def _s(s):
            print(f"  [TestAllianceClick] {s}")
            if status_cb:
                try: status_cb(s)
                except Exception: pass

        title = target_title or self.window_title
        hwnd  = find_hwnd(title)
        if not hwnd:
            _s("no window found"); return

        screen = capture_game_ignore_ui(title)
        if screen is None:
            _s("capture failed"); return

        rect   = win32gui.GetClientRect(hwnd)
        cw, ch = rect[2], rect[3]
        scale  = cw / 1718.0

        x2 = cw // 2 - int(round(47 * scale))
        y2 = ch      - int(round(70 * scale))
        _s(f"2nd click client=({x2},{y2})")

        vis = screen.copy()
        # draw crosshair + circle at the target position
        cv2.circle(vis, (x2, y2), int(round(20 * scale)), (0, 255, 255), 2)
        cv2.line(vis, (x2 - int(round(25*scale)), y2), (x2 + int(round(25*scale)), y2), (0, 255, 255), 2)
        cv2.line(vis, (x2, y2 - int(round(25*scale))), (x2, y2 + int(round(25*scale))), (0, 255, 255), 2)
        cv2.putText(vis, f"2nd click ({x2},{y2})", (max(0, x2 - 80), max(20, y2 - 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, f"2nd click ({x2},{y2})", (max(0, x2 - 80), max(20, y2 - 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.imshow("Test Alliance Click", vis)
        cv2.waitKey(1)
        _s("shown in cv2 window")

    def test_location(self, target_title=None, status_cb=None):
        """Capture the 966/1718-cropped ROI and detect two regions:

        ROI-A  zone number : horizontally centred, top of cropped image, 88×26 px
        ROI-B  map coords  : bottom-left + right 814 + up 24, 61×18 px

        Prints a single line:  [0188, 39/39]
        """
        import re

        def _s(s):
            print(f"  [TestLocation] {s}")
            if status_cb:
                try: status_cb(s)
                except Exception: pass

        title  = target_title or self.window_title
        screen = capture_game_ignore_ui(title)
        if screen is None:
            _s("capture failed"); return

        # crop to 966/1718 ratio (bottom region)
        raw_h, raw_w = screen.shape[:2]
        tgt_h = int(round(raw_w * 966.0 / 1718.0))
        ct    = max(0, raw_h - tgt_h)
        img   = screen[ct:, :, :].copy()

        ih, iw = img.shape[:2]
        scale  = iw / 1718.0

        # --- ROI-A: zone number (top-centre) ---
        hw_a = int(round(88 * scale / 2))
        hh_a = max(1, int(round(35 * scale / 2)))
        cx_a = iw // 2
        cy_a = hh_a
        ax1, ay1 = max(0, cx_a - hw_a), 0
        ax2, ay2 = min(iw, cx_a + hw_a), min(ih, cy_a + hh_a)

        # --- ROI-B: map coordinates (bottom-left origin + right 814 + up 24) ---
        hw_b = int(round(61 * scale / 2))
        hh_b = max(1, int(round(20 * scale / 2)))
        cx_b = int(round(814 * scale))
        cy_b = ih - int(round(26 * scale))
        bx1, by1 = max(0, cx_b - hw_b), max(0, cy_b - hh_b)
        bx2, by2 = min(iw, cx_b + hw_b), min(ih, cy_b + hh_b)

        # draw overlay
        vis = img.copy()
        for (r1, r2, col, lbl) in [
            ((ax1, ay1, ax2, ay2), None, (0, 165, 255), "zone"),
            ((bx1, by1, bx2, by2), None, (0, 255, 128), "coord"),
        ]:
            x1, y1, x2, y2 = r1
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, lbl, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(vis, lbl, (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.imshow("Test Location", vis)
        cv2.waitKey(1)

        # OCR helper
        def _ocr(patch, allowlist, scale_f=4):
            if patch.size == 0:
                return ""
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            big  = cv2.resize(gray, (0, 0), fx=scale_f, fy=scale_f,
                              interpolation=cv2.INTER_CUBIC)
            reader = get_ocr_engine()
            if reader is None:
                return ""
            out = reader.readtext(big, detail=0, allowlist=allowlist)
            return "".join(out)

        try:
            # Zone number (4-digit)
            raw_a  = _ocr(img[ay1:ay2, ax1:ax2], '0123456789')
            zone_nums = re.findall(r'\d{4}', raw_a)
            zone_str  = zone_nums[0] if zone_nums else raw_a.strip() or "????"

            # Map coords  (dd/dd or ddd/ddd)
            raw_b   = _ocr(img[by1:by2, bx1:bx2], '0123456789/')
            coord_m = re.search(r'(\d{1,3})\s*/\s*(\d{1,3})', raw_b)
            coord_str = f"{coord_m.group(1)}/{coord_m.group(2)}" if coord_m else raw_b.strip() or "?/?"

            result = f"[{zone_str}, {coord_str}]"
            print(f"  [Location] {result}")
            if status_cb:
                try: status_cb(result)
                except Exception: pass
        except Exception as ex:
            _s(f"OCR error: {ex}")

    def map_scan_once(self, target_title=None, model_path=None, torch_model_path=None,
                      status_cb=None, onefield_list=None):
        """Scan current screen for alliance tiles (no swapping).
        For each alliance tile sorted by confidence:
          - run 2-click info flow
          - match alliance template
          - if matched: OCR location, dedup against onefield_list, then print+append
        Loop ends when all hits are processed.
        """
        title = target_title or self.window_title

        def _s(s):
            # print(f"  [MapScanOnce] {s}")
            if status_cb:
                try: status_cb(s)
                except Exception: pass

        hwnd = find_hwnd(title)
        if not hwnd:
            # _s("no window found") 
            return

        # _s("scanning...")
        result = self.browse_and_predict(
            target_title=title,
            model_path=model_path,
            torch_model_path=torch_model_path,
            status_cb=None,             # suppress per-tile noise
            auto_click_alliance=False,  # collect hits but don't auto-click
        )
        if result is None:
            # _s("scan failed")
            return

        _, _, hits_alliance = result
        if not hits_alliance:
            # _s("no alliance tiles found")
            return

        hits_alliance.sort(key=lambda t: t[0], reverse=True)
        # _s(f"{len(hits_alliance)} alliance tile(s) found, processing in order...")

        # Need crop_top to translate img coords back to client coords.
        # Re-derive it from a fresh capture (cheap).
        import win32gui as _w32
        raw = capture_game_ignore_ui(title)
        crop_top = 0
        if raw is not None:
            rh, rw = raw.shape[:2]
            tgt_h  = int(round(rw * 966.0 / 1718.0))
            crop_top = max(0, rh - tgt_h)

        for idx, (conf, img_cx, img_cy, terrain) in enumerate(hits_alliance):
            client_cx = int(img_cx)
            client_cy = int(img_cy) + crop_top
            _s(f"[{idx+1}/{len(hits_alliance)}] {terrain} conf={conf*100:.0f}%  client=({client_cx},{client_cy})")
            result, coor = self.alliance_detection(hwnd, client_cx, client_cy, status_cb=None, terrain=terrain, terrain_conf=float(conf))
            if result and coor:
                _already_seen = any(coor in entry for entry in (onefield_list or []) + self.global_list)
                if not _already_seen:
                    if onefield_list is not None:
                        print(result)
                        onefield_list.append(result)
                        self.global_list.append(result)
                # else: duplicate coords — skip silently
        _s("done")

    def map_scan_full(self, target_title=None, model_path=None, torch_model_path=None,
                     status_cb=None):
        """Full map sweep: navigate through 3 rows × 3 columns and scan each viewport.

        Sequence
        --------
        1.  swap_to_bottom_right
        2.  map_scan_once
        3.  swap_to_left
        4.  map_scan_once
        5.  swap_to_left
        6.  map_scan_once
        7.  swap_to_bottom_right + swap_to_bottom
        8.  map_scan_once
        9.  swap_to_left
        10. map_scan_once
        11. swap_to_left
        12. map_scan_once
        13. swap_to_bottom_right + 2× swap_to_bottom
        14. map_scan_once
        15. swap_to_left
        16. map_scan_once
        17. swap_to_left
        18. map_scan_once
        """
        title = target_title or self.window_title

        def _s(msg):
            # print(f"  [MapScanFull] {msg}")
            if status_cb:
                try: status_cb(msg)
                except Exception: pass

        onefield_list = []

        def _scan():
            self.map_scan_once(target_title=title,
                               model_path=model_path,
                               torch_model_path=torch_model_path,
                               status_cb=status_cb,
                               onefield_list=onefield_list)

        def _rbr():
            self.swap_to_bottom_right(target_title=title)

        def _left():
            self.swap_to_left(target_title=title)

        def _bot():
            self.swap_to_bottom(target_title=title)

        _s("step 1: swap_to_bottom_right");  _rbr()
        _s("step 2: map_scan_once");          _scan()
        _s("step 3: swap_to_left");           _left()
        _s("step 4: map_scan_once");          _scan()
        _s("step 5: swap_to_left");           _left()
        _s("step 6: map_scan_once");          _scan()
        _s("step 7: swap_to_bottom_right + swap_to_bottom"); _rbr(); _bot()
        _s("step 8: map_scan_once");          _scan()
        _s("step 9: swap_to_left");           _left()
        _s("step 10: map_scan_once");         _scan()
        _s("step 11: swap_to_left");          _left()
        _s("step 12: map_scan_once");         _scan()
        _s("step 13: swap_to_bottom_right + 2x swap_to_bottom"); _rbr(); _bot(); _bot()
        _s("step 14: map_scan_once");         _scan()
        _s("step 15: swap_to_left");          _left()
        _s("step 16: map_scan_once");         _scan()
        _s("step 17: swap_to_left");          _left()
        _s("step 18: map_scan_once");         _scan()

        # append this full-sweep results to the session global list (dedup across runs)
        added = 0
        for item in onefield_list:
            if item not in self.global_list:
                self.global_list.append(item)
                added += 1
        # print(f"  [MapScanFull] done  this_sweep={len(onefield_list)}  new={added}  total={len(self.global_list)}")

    def map_scan_global(self, target_title=None, model_path=None, torch_model_path=None,
                        status_cb=None):
        """Start a fresh global scan session: reset global_list, run map_scan_full,
        then print the complete accumulated list."""
        title = target_title or self.window_title

        def _s(msg):
            print(f"  [MapScanGlobal] {msg}")
            if status_cb:
                try: status_cb(msg)
                except Exception: pass

        # _s("new session — resetting global_list")
        self.global_list = []

        # prepare output txt file  e.g.  04Mar2026.txt
        _date_str = datetime.datetime.now().strftime("%d%b%Y")   # "04Mar2026"
        _txt_path = os.path.join(self._base, f"{_date_str}.txt")
        _header_written = False

        def _write_new_entries(prev_len):
            nonlocal _header_written
            new_items = self.global_list[prev_len:]
            if not new_items:
                return
            with open(_txt_path, "a", encoding="utf-8") as f:
                if not _header_written:
                    f.write(f"=== GLOBAL LIST ({len(self.global_list)} entries) ===\n")
                    _header_written = True
                for item in new_items:
                    f.write(f"  {item}\n")

        i = 0
        while i < 2 and len(self.global_list) < 100:  # Example loop, replace with actual condition
            _prev_len = len(self.global_list)
            self.map_scan_full(target_title=title,
                               model_path=model_path,
                               torch_model_path=torch_model_path,
                               status_cb=status_cb)
            _write_new_entries(_prev_len)
            i += 1
            self.test_field_change(target_title=title, status_cb=status_cb)
            time.sleep(7)

        print(f"\n=== GLOBAL LIST ({len(self.global_list)} entries) ===")
        for item in self.global_list:
            print(f"  {item}")
        print("==========================================")
        _s(f"map_scan_global done  →  {_txt_path}")

    def test_field_change(self, target_title=None, status_cb=None):
        """1. Click the centre of the zone-number ROI (top-centre of the cropped image).
           2. Wait 0.5 s.
           3. Show the full cropped image with a yellow marker at
              bottom-left + right 1255 + up 617 (reference 1718×966).
        """
        def _s(s):
            # print(f"  [TestFieldChange] {s}")
            if status_cb:
                try: status_cb(s)
                except Exception: pass

        title  = target_title or self.window_title
        hwnd   = find_hwnd(title)
        if not hwnd:
            _s("no window found"); return

        screen = capture_game_ignore_ui(title)
        if screen is None:
            _s("capture failed"); return

        # crop to 966/1718 ratio
        raw_h, raw_w = screen.shape[:2]
        tgt_h  = int(round(raw_w * 966.0 / 1718.0))
        ct     = max(0, raw_h - tgt_h)
        img    = screen[ct:, :, :].copy()
        ih, iw = img.shape[:2]
        scale  = iw / 1718.0

        # --- Step 1: click centre of zone ROI (ROI-A: top-centre, 88×35 px ref) ---
        hw_a   = int(round(88 * scale / 2))
        hh_a   = max(1, int(round(35 * scale / 2)))
        zone_cx_img = iw // 2          # centre x in cropped image
        zone_cy_img = hh_a             # centre y in cropped image
        # convert to client coords (add crop_top back)
        client_x = zone_cx_img
        client_y = zone_cy_img + ct
        pt = win32gui.ClientToScreen(hwnd, (client_x, client_y))
        _s(f"clicking zone ROI centre  client=({client_x},{client_y})")
        pyautogui.click(pt[0], pt[1])
        time.sleep(0.5)

        # --- Step 2: click at right 1255, up 617 from bottom-left ---
        mx = int(round(1255 * scale))
        my = ih - int(round(617 * scale))
        mx = max(0, min(iw - 1, mx))
        my = max(0, min(ih - 1, my))
        client_x2 = mx
        client_y2 = my + ct
        pt2 = win32gui.ClientToScreen(hwnd, (client_x2, client_y2))
        _s(f"clicking field-change pos  client=({client_x2},{client_y2})")
        pyautogui.click(pt2[0], pt2[1])

    def browse_and_predict(self, target_title=None, model_path=None, torch_model_path=None,
                           show_win_name="Map Predictions", status_cb=None, conf_thresh=0.15,
                           auto_click_alliance=True):
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
            # print(f"  [FieldScanner] {s}")
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

        def _crop_bottom(frame):
            """Crop to 966/1718 aspect ratio, keeping the bottom region."""
            _h, _w = frame.shape[:2]
            _th = int(round(_w * 966.0 / 1718.0))
            _ct = max(0, _h - _th)
            return (frame[_ct:, :, :].copy() if _ct > 0 else frame.copy()), _ct

        img, crop_top = _crop_bottom(screen)

        # detect anchor, try fallback click sequence like other callers
        fw, fh = self.grid_fw, self.grid_fh
        cy_img = max(0, cy - crop_top)
        _, _, anchor = self.detect_yellow_frame(img, cx, cy_img)
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
                screen2_c, _ = _crop_bottom(screen2)
                _, _, anchor = self.detect_yellow_frame(screen2_c, cur_x, max(0, cur_y - crop_top))
                if anchor:
                    img = screen2_c
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
            # 'copper': (60, 60, 220),
            # 'stone':  (200, 200, 200),
        }
        _active_terrains = frozenset(terrain_to_bgr)  # only these get labels + clicks

        # collect alliance hits for auto-click
        hits_alliance = []   # list of (conf, img_cx, img_cy, terrain_name)

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

            # only process clay / forest / boat
            if terrain_name not in _active_terrains or conf < conf_thresh:
                continue

            # collect alliance tiles
            if occ_name.lower() == 'alliance':
                hits_alliance.append((conf, x1 + gw // 2, y1 + gh // 2, terrain_name))

            color = terrain_to_bgr[terrain_name]
            pts = np.array([(x1 + gw//2, y1), (x2, y1 + gh//2), (x1 + gw//2, y2), (x1, y1 + gh//2)], dtype=np.int32)
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.28, out, 1 - 0.28, 0, out)
            cv2.polylines(out, [pts], True, color, 2)
            _cx = x1 + gw//2; _cy = y1 + gh//2
            label_str = f"{terrain_name}, {occ_name} {int(conf*100)}%"
            cv2.putText(out, label_str, (_cx - gw//4, _cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
            cv2.putText(out, label_str, (_cx - gw//4, _cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imshow(show_win_name, out)
        cv2.waitKey(1)

        # auto-click best alliance tile via alliance_detection
        found_alliance = False
        if auto_click_alliance and hits_alliance:
            found_alliance = True
            hits_alliance.sort(key=lambda t: t[0], reverse=True)
            best_conf, img_cx, img_cy, best_terrain = hits_alliance[0]
            # convert cropped-image coords back to client coords
            client_cx = int(img_cx)
            client_cy = int(img_cy) + crop_top
            _status(
                f"alliance hits={len(hits_alliance)}  best={best_terrain} "
                f"conf={best_conf*100:.0f}%  client=({client_cx},{client_cy})"
            )
            time.sleep(0.3)
            self.alliance_detection(hwnd, client_cx, client_cy, status_cb=_status)
        elif auto_click_alliance:
            _status("no alliance tiles found")

        _status("browse_and_predict done")
        return out, found_alliance, hits_alliance
