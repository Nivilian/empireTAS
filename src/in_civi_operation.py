import threading
import time
import cv2
import os
import math
import numpy as np
import win32gui
import pyautogui

from vision_core import capture_game_ignore_ui, find_image_and_location, find_hwnd


class InCiviOperation:
    """Detect the in-civilization UI 'prefer' (like) icon and optionally auto-click population.

    Usage:
      op = InCiviOperation(target_title)
      op.start()  # runs background thread
      op.stop()
      op.toggle()
      op.test_roi()  # show ROIs for tuning
    """
    def __init__(self, target_title, prefer_dir="baseUI/prefer", population_template="baseUI/inCivi/population.png", status_cb=None):
        self.target_title = target_title
        # directory containing prefer templates (like/respect images)
        self.prefer_dir = prefer_dir
        self.prefer_templates = []
        # load templates list lazily
        self.population_template = population_template
        self._running = False
        self._thread = None
        self._active_mode = None  # 'like' or 'respect' or None
        self.status_cb = status_cb

        # configurable timings
        self.check_interval = 5.0
        # default intervals (seconds) per mode
        self.intervals = {'like': 61.0, 'respect': 11.0}
        self.population_interval = self.intervals['like']

    def is_running(self):
        return self._running

    def start(self, status_cb=None):
        if self._running:
            return
        if status_cb is not None:
            self.status_cb = status_cb
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def toggle(self):
        if self._running:
            self.stop()
        else:
            self.start()

    def _click_at(self, cx, cy):
        hwnd = find_hwnd(self.target_title)
        if not hwnd:
            return False
        try:
            pt = win32gui.ClientToScreen(hwnd, (int(cx), int(cy)))
            pyautogui.click(pt[0], pt[1])
            return True
        except Exception:
            return False

    def _worker(self):
        # This worker performs: detect -> immediate click -> wait interval -> repeat.
        while self._running:
            screen = capture_game_ignore_ui(self.target_title)
            if screen is None:
                time.sleep(self.check_interval)
                continue
            h, w = screen.shape[:2]
            # Crop to bottom region with reference aspect ratio 1718:966 (width:height)
            ref_w = 1718.0
            ref_h = 966.0
            target_h = int(round(w * (ref_h / ref_w)))
            if target_h <= 0 or target_h > h:
                target_h = h
            crop_y0 = max(0, h - target_h)
            crop = screen[crop_y0:h, 0:w]
            ch, cw = crop.shape[:2]

            # Use crop width to scale reference offsets
            scale = float(cw) / ref_w if ref_w != 0 else 1.0
            # Preferred detection center offsets from bottom-left of the reference area
            pref_off_x = 1158.0
            pref_off_y = 947.0
            # compute coordinates inside the crop (origin top-left)
            pref_x = int(round(scale * pref_off_x))
            pref_y = int(round(ch - scale * pref_off_y))

            # Preferred click position offsets from bottom-left
            click_off_x = 1614.0
            click_off_y = 844.0
            click_x = int(round(scale * click_off_x))
            click_y = int(round(ch - scale * click_off_y))

            # define a fixed ROI around pref_x,pref_y: width=128, height=43 (scaled)
            ref_roi_w = 128.0
            ref_roi_h = 43.0
            roi_w = int(round(scale * ref_roi_w))
            roi_h = int(round(scale * ref_roi_h))
            roi_half_w = max(8, roi_w // 2)
            roi_half_h = max(6, roi_h // 2)
            roi_x = max(0, pref_x - roi_half_w)
            roi_y = max(0, pref_y - roi_half_h)
            roi_x2 = min(cw, pref_x + roi_half_w)
            roi_y2 = min(ch, pref_y + roi_half_h)
            search_area = crop[roi_y:roi_y2, roi_x:roi_x2]
            # try to find prefer icon within the scaled ROI (match against all templates in prefer_dir)
            like_found = False
            matched_mode = None
            try:
                if not self.prefer_templates:
                    # glob for images in prefer_dir
                    import glob
                    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", self.prefer_dir)
                    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
                    files = []
                    for p in patterns:
                        files.extend(glob.glob(os.path.join(base, p)))
                    self.prefer_templates = files
                for tpl in self.prefer_templates:
                    res, conf = find_image_and_location(search_area, tpl, threshold=0.6)
                    if res:
                        like_found = True
                        name = os.path.basename(tpl).lower()
                        matched_mode = 'respect' if 'respect' in name else 'like'
                        break
            except Exception:
                like_found = False
                matched_mode = None

            if not like_found:
                # update status and continue scanning after a short wait
                try:
                    if self.status_cb:
                        self.status_cb("Prefer: not found")
                except Exception:
                    pass
                time.sleep(self.check_interval)
                continue

            # matched: perform immediate click, set mode and compute next wait interval
            full_click_x = click_x
            full_click_y = crop_y0 + click_y
            clicked = False
            if 0 <= full_click_x < w and 0 <= full_click_y < h:
                clicked = self._click_at(full_click_x, full_click_y)
            else:
                # fallback template-based click if coords out of bounds
                res2, conf2 = find_image_and_location(screen, self.population_template, threshold=0.6)
                if res2:
                    px, py = res2
                    clicked = self._click_at(px + 10, py + 10)

            self._active_mode = matched_mode
            self.population_interval = self.intervals.get(matched_mode, self.intervals['like'])
            # update status: detected mode and whether click succeeded
            try:
                if self.status_cb:
                    self.status_cb(f"Detected: {matched_mode}  clicked={'yes' if clicked else 'no'}")
            except Exception:
                pass

            # wait until next scheduled click (do not perform detection during wait)
            next_time = time.time() + self.population_interval
            while self._running and time.time() < next_time:
                # update countdown status every 0.5s
                remaining = int(round(next_time - time.time()))
                try:
                    if self.status_cb:
                        self.status_cb(f"Detected: {self._active_mode}  next in {remaining}s  clicked={'yes' if clicked else 'no'}")
                except Exception:
                    pass
                time.sleep(0.5)

            # short sleep to avoid tight loop if running continuously
            time.sleep(self.check_interval)

    def test_roi(self):
        """Capture the game screen and show ROIs used for prefer detection and a coarse population search area."""
        screen = capture_game_ignore_ui(self.target_title)
        if screen is None:
            print("[InCiviOperation] No screen capture available")
            return
        h, w = screen.shape[:2]
        # crop bottom area with reference ratio
        ref_w = 1718.0
        ref_h = 966.0
        target_h = int(round(w * (ref_h / ref_w)))
        if target_h <= 0 or target_h > h:
            target_h = h
        crop_y0 = max(0, h - target_h)
        crop = screen[crop_y0:h, 0:w]
        ch, cw = crop.shape[:2]
        img = crop.copy()
        scale = float(cw) / ref_w if ref_w != 0 else 1.0
        pref_cx = int(round(scale * 1158.0))
        pref_cy = int(round(ch - scale * 947.0))
        click_cx = int(round(scale * 1614.0))
        click_cy = int(round(ch - scale * 844.0))
        # draw the scaled 128x43 ROI centered at (pref_cx, pref_cy)
        roi_w = int(round(scale * 128.0))
        roi_h = int(round(scale * 43.0))
        roi_x1 = max(0, pref_cx - roi_w // 2)
        roi_y1 = max(0, pref_cy - roi_h // 2)
        roi_x2 = min(cw, pref_cx + (roi_w - roi_w // 2))
        roi_y2 = min(ch, pref_cy + (roi_h - roi_h // 2))
        cv2.rectangle(img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0,255,0), 2)
        # draw a small rectangle marking the click target
        click_box_w = max(6, int(round(0.03 * cw)))
        click_box_h = max(6, int(round(0.03 * ch)))
        cx1 = max(0, click_cx - click_box_w // 2)
        cy1 = max(0, click_cy - click_box_h // 2)
        cx2 = min(cw, click_cx + (click_box_w - click_box_w // 2))
        cy2 = min(ch, click_cy + (click_box_h - click_box_h // 2))
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (255,0,0), 2)
        cv2.imshow("InCivi ROIs (bottom crop)", img)
        cv2.waitKey(1)
