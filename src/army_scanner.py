import os, re
import numpy as np
import cv2
import ctypes
import win32gui

from vision_core import imread_safe, find_all_matches, find_hwnd, get_ocr_engine


class ArmyScanner:
    """
    Handles all army-panel related vision tasks:
      - locating the army grid
      - identifying slot contents (unit type, status, region, count)
      - reading total army count from the panel header
      - scrolling the army list
      - debug overlay rendering
    Templates are loaded once on construction.
    """

    def __init__(self, base_dir=None):
        self._base = base_dir or os.path.dirname(os.path.abspath(__file__))

        # --- Grid anchor & slot-corner templates ---
        self._tpl_left  = imread_safe(os.path.join(self._base, "images", "army", "anchor_left.png"))
        self._tpl_right = imread_safe(os.path.join(self._base, "images", "army", "anchor_right.png"))
        self._tpl_tr    = imread_safe(os.path.join(self._base, "images", "army", "slot_topright.png"))
        self._tpl_br    = imread_safe(os.path.join(self._base, "images", "army", "slot_bottomright.png"))

        # --- Status icon templates ---
        self.status_tpls = {
            "Moving":    imread_safe(os.path.join(self._base, "images", "army", "status", "armyMoving.png")),
            "Stationed": imread_safe(os.path.join(self._base, "images", "army", "status", "armyStationed.png")),
            "Waiting":   imread_safe(os.path.join(self._base, "images", "army", "status", "armyWaiting.png")),
            "InBattle":  imread_safe(os.path.join(self._base, "images", "army", "status", "armyInBattle.png")),
        }

        # --- Unit type icon templates ---
        self.type_tpls = {
            "Xiaowei":        imread_safe(os.path.join(self._base, "images", "army", "types", "连弩校尉.png")),
            "Nuwa":           imread_safe(os.path.join(self._base, "images", "army", "types", "女娲.png")),
            "Mammoth":        imread_safe(os.path.join(self._base, "images", "army", "types", "猛犸.png")),
            "PersianCavalry": imread_safe(os.path.join(self._base, "images", "army", "types", "波斯轻骑兵.png")),
            "Centaur":        imread_safe(os.path.join(self._base, "images", "army", "types", "半人马.png")),
            "Spearman":       imread_safe(os.path.join(self._base, "images", "army", "types", "标枪手.png")),
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _refine_slot_rect(self, screen, cell_left, cell_top, cell_right, cell_bottom):
        """
        Within the rough geometric cell, find slot_topright and slot_bottomright
        to compute a precise slot rectangle.
        Returns (left, top, slot_w, slot_h) or None if not found.
        """
        tpl_tr, tpl_br = self._tpl_tr, self._tpl_br
        cell_roi = screen[cell_top:cell_bottom, cell_left:cell_right]
        cell_h, cell_w = cell_roi.shape[:2]

        tr_zone = cell_roi[:cell_h // 2, :]
        tr_hits = find_all_matches(tr_zone, tpl_tr, threshold=0.65)
        if not tr_hits:
            return None
        tx, ty, tw, th = max(tr_hits, key=lambda h: h[0])
        top   = cell_top + ty
        right = cell_left + tx + tw

        br_zone = cell_roi[cell_h // 2:, :]
        br_hits = find_all_matches(br_zone, tpl_br, threshold=0.65)
        if br_hits:
            bx, by, bw, bh = max(br_hits, key=lambda h: h[0])
            bottom = cell_top + cell_h // 2 + by + bh
        else:
            bottom = cell_bottom

        slot_h = bottom - top
        if slot_h <= 0:
            return None
        slot_w = max(1, int(slot_h * (389 / 105)))
        left   = right - slot_w
        return (left, top, slot_w, slot_h)

    def _locate_grid(self, screen):
        """
        Find the army grid via left/right anchor templates.
        Returns (grid_x, grid_y, grid_w, grid_h) or None if not found.
        """
        tpl_left, tpl_right = self._tpl_left, self._tpl_right
        if tpl_left is None or tpl_right is None:
            print("Missing anchor_left.png or anchor_right.png")
            return None
        res_l = cv2.matchTemplate(screen, tpl_left,  cv2.TM_CCOEFF_NORMED)
        res_r = cv2.matchTemplate(screen, tpl_right, cv2.TM_CCOEFF_NORMED)
        _, val_l, _, loc_l = cv2.minMaxLoc(res_l)
        _, val_r, _, loc_r = cv2.minMaxLoc(res_r)
        if val_l < 0.7 or val_r < 0.7:
            print(f"Grid anchors not found (l={val_l:.2f} r={val_r:.2f})")
            return None
        grid_x = loc_l[0]
        grid_y = max(loc_l[1] + tpl_left.shape[0], loc_r[1] + tpl_right.shape[0])
        grid_w = (loc_r[0] + tpl_right.shape[1]) - grid_x
        grid_h = int(grid_w * (417 / 1198))
        return grid_x, grid_y, grid_w, grid_h

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_slots(self, screen, scan_zone="top"):
        """
        Locate the army grid and return a list of slot dicts plus grid_info.
        Each slot: {'image': crop, 'index': int, 'rect': (left, top, w, h)}
        scan_zone: "top" (default) | "bottom"
        """
        tpl_tr, tpl_br = self._tpl_tr, self._tpl_br
        if tpl_tr is None or tpl_br is None:
            print("Missing slot_topright.png or slot_bottomright.png")
            return [], {}

        grid = self._locate_grid(screen)
        if grid is None:
            return [], {}
        grid_x, grid_y, grid_w, grid_h = grid
        col_w  = grid_w // 3
        scan_h = int(grid_w * (380 / 1198))

        if scan_zone == "bottom":
            scan_y1 = max(0, grid_y + grid_h - scan_h)
            scan_y2 = min(screen.shape[0], grid_y + grid_h)
        else:
            scan_y1 = grid_y
            scan_y2 = min(screen.shape[0], grid_y + scan_h)
        zone_h = scan_y2 - scan_y1
        row_h  = zone_h // 3
        print(f"  [Grid] scan_zone={scan_zone}  scan_y1={scan_y1}  scan_y2={scan_y2}  row_h={row_h}")

        slots = []
        viz   = screen.copy()
        cv2.rectangle(viz, (grid_x, scan_y1), (grid_x + grid_w, scan_y2), (255, 100, 0), 1)
        cv2.rectangle(viz, (grid_x, grid_y),  (grid_x + grid_w, grid_y + grid_h), (0, 0, 255), 2)

        measured_row_h = row_h
        for row_idx in range(3):
            geo_top    = scan_y1 + row_idx * row_h
            geo_bottom = geo_top + row_h
            for col_idx in range(3):
                idx       = row_idx * 3 + col_idx
                geo_left  = grid_x + col_idx * col_w
                geo_right = geo_left + col_w

                refined = self._refine_slot_rect(screen, geo_left, geo_top, geo_right, geo_bottom)
                if refined:
                    left, top, slot_w, slot_h = refined
                    bottom, right = top + slot_h, left + slot_w
                else:
                    left, top     = geo_left, geo_top
                    right, bottom = geo_right, geo_bottom
                    slot_w, slot_h = col_w, row_h

                cell_img = screen[top:bottom, left:right]
                slots.append({'image': cell_img, 'index': idx,
                              'rect': (left, top, slot_w, slot_h)})

                color = (0, 255, 0) if refined else (0, 165, 255)
                cv2.rectangle(viz, (left, top), (right, bottom), color, 1)
                dw1 = int(slot_w * (1 / 4))
                dw2 = int(slot_w * (14 / 27))
                cv2.line(viz, (left + dw1, top), (left + dw1, bottom), (255, 0, 0), 1)
                cv2.line(viz, (left + dw1 + dw2, top), (left + dw1 + dw2, bottom), (0, 255, 255), 1)
                cv2.putText(viz, str(idx + 1), (left + 5, top + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if len(slots) >= 4:
            slot1_top = slots[0]['rect'][1]
            slot4_top = slots[3]['rect'][1]
            if slot4_top > slot1_top:
                measured_row_h = slot4_top - slot1_top
                ax = grid_x + grid_w + 18
                cv2.arrowedLine(viz, (ax, slot4_top), (ax, slot1_top), (0, 255, 255), 2, tipLength=0.08)
                cv2.arrowedLine(viz, (ax, slot1_top), (ax, slot4_top), (0, 255, 255), 2, tipLength=0.08)
                cv2.putText(viz, f"{measured_row_h}px", (ax + 6, (slot1_top + slot4_top) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Army ROI Debug", viz)
        cv2.waitKey(1)

        grid_info = {
            "grid_x": grid_x, "grid_y": grid_y,
            "grid_w": grid_w, "grid_h": grid_h,
            "slot_w": col_w,  "slot_h": measured_row_h,
        }
        return slots, grid_info

    def read_count(self, screen):
        """
        OCR the '当前 4 / 上限 28' text to the right of anchor_left.
        Returns (current, maximum) as ints, or (0, 0) on failure.

        Strategy: grab a wide strip from the anchor rightward to the screen edge,
        run OCR without an allowlist so Chinese chars are preserved, then extract
        the two numbers around '/' with a regex.
        """
        tpl = self._tpl_left
        if tpl is None: return 0, 0
        res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(res)
        if val < 0.6: return 0, 0
        ax, ay = loc[0] + tpl.shape[1], loc[1]
        # English: extend strip all the way to the right edge of the screen
        strip_w = screen.shape[1] - ax
        if strip_w <= 0: return 0, 0
        strip = screen[ay:ay + tpl.shape[0], ax:ax + strip_w]
        reader = get_ocr_engine()
        if reader is None: return 0, 0
        try:
            gray     = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
            enlarged = cv2.resize(gray, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            # English: no allowlist — keep Chinese so word order is intact
            results  = reader.readtext(enlarged, detail=0)
            raw      = " ".join(results)
            print(f"  [ArmyCount OCR] raw={repr(raw)}")
            # English: match 'N / M' or 'N/M' directly
            slash_match = re.search(r'(\d+)\s*/\s*(\d+)', raw)
            if slash_match:
                cur = int(slash_match.group(1))
                mx  = int(slash_match.group(2))
                print(f"  [ArmyCount] current={cur}  maximum={mx}")
                return cur, mx
            # English: fallback — two separate number groups (e.g. OCR drops the slash)
            nums = re.findall(r'\d+', raw)
            print(f"  [ArmyCount] fallback nums={nums}")
            if len(nums) >= 2:
                return int(nums[0]), int(nums[1])
            elif len(nums) == 1:
                return int(nums[0]), 0   # maximum unknown, do not assume equal
        except Exception as e:
            print(f"  [ArmyCount OCR error]: {e}")
        return 0, 0

    def scroll(self, window_title, grid_info, drag_px):
        """
        Drag upward inside the army grid by drag_px pixels.
        """
        import pyautogui, time
        hwnd = find_hwnd(window_title)
        if not hwnd: return
        gx     = grid_info["grid_x"]
        gy     = grid_info["grid_y"]
        gw     = grid_info["grid_w"]
        gh     = grid_info["grid_h"]
        slot_h = grid_info.get("slot_h", gh // 3)
        cx      = gx + gw // 2
        start_y = gy + gh - 5 - slot_h
        end_y   = start_y - drag_px
        start_screen = win32gui.ClientToScreen(hwnd, (cx, start_y))
        end_screen   = win32gui.ClientToScreen(hwnd, (cx, end_y))
        dpi            = ctypes.windll.user32.GetDpiForSystem()
        pixels_per_cm  = dpi / 2.54
        duration       = drag_px / (1.2 * pixels_per_cm)
        print(f"  [Scroll] drag_px={drag_px}  duration={duration:.2f}s")
        pyautogui.moveTo(start_screen[0], start_screen[1])
        pyautogui.mouseDown(button='left')
        time.sleep(0.5)
        pyautogui.moveTo(end_screen[0], end_screen[1], duration=duration)
        time.sleep(1.0)
        pyautogui.mouseUp(button='left')

    def identify_content(self, cell_img, slot_id=0):
        """
        Identify the unit type, status, region coords, and count from a slot crop.
        Uses self.status_tpls and self.type_tpls loaded at construction time.
        Returns a dict: {unit, status, region, count}
        """
        res = {"unit": "Empty", "status": "Idle", "region": [], "count": ""}
        if cell_img is None: return res

        ocr_dir = os.path.join(self._base, "images", "ocr")
        os.makedirs(ocr_dir, exist_ok=True)
        cv2.imencode('.png', cell_img)[1].tofile(
            os.path.join(ocr_dir, f"slot_{slot_id}.png"))

        h, w = cell_img.shape[:2]
        w1 = int(w * (1 / 4))
        w2 = int(w * (14 / 27))
        left_part   = cell_img[:, :w1]
        middle_part = cell_img[:, w1:w1 + w2]
        right_part  = cell_img[:, w1 + w2:]

        # 1. Unit type
        best_unit, best_val = "Empty", 0.0
        conf_lines = []
        for name, tpl in self.type_tpls.items():
            if tpl is None: continue
            th, tw = tpl.shape[:2]
            lh, lw = left_part.shape[:2]
            if tw > lw or th > lh:
                scale = min(lw / tw, lh / th) * 0.95
                tpl   = cv2.resize(tpl, (max(1, int(tw * scale)), max(1, int(th * scale))))
                th, tw = tpl.shape[:2]
            if lh < th or lw < tw:
                conf_lines.append(f"{name}=SKIP")
                continue
            val = cv2.minMaxLoc(cv2.matchTemplate(left_part, tpl, cv2.TM_CCOEFF_NORMED))[1]
            conf_lines.append(f"{name}={val:.3f}")
            if val > best_val:
                best_val, best_unit = val, name
        print(f"  [Type slot {slot_id}] {' | '.join(conf_lines)} => best={best_unit}({best_val:.3f})")
        if best_val >= 0.4:
            res["unit"] = best_unit

        # 2. Status
        for name, tpl in self.status_tpls.items():
            if tpl is not None:
                th, tw = tpl.shape[:2]
                if right_part.shape[0] < th or right_part.shape[1] < tw: continue
                val = cv2.minMaxLoc(cv2.matchTemplate(right_part, tpl, cv2.TM_CCOEFF_NORMED))[1]
                if val > (0.53 if name == "Moving" else 0.7):
                    res["status"] = name
                    break

        # 3. OCR region/coords/count
        if res["unit"] != "Empty":
            reader = get_ocr_engine()
            if reader:
                try:
                    mh          = middle_part.shape[0]
                    top_part    = middle_part[:int(mh * 0.55), :]
                    bottom_part = middle_part[int(mh * 0.55):, :]

                    def ocr_region(img, allowlist):
                        gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        enlarged = cv2.resize(gray, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                        results  = reader.readtext(enlarged, detail=0, allowlist=allowlist)
                        return " ".join(results)

                    top_text    = ocr_region(top_part,    '0123456789/区')
                    bottom_text = ocr_region(bottom_part, '0123456789xX*')
                    print(f"  [OCR slot {slot_id}] top={repr(top_text)}  bottom={repr(bottom_text)}")

                    top_nums    = re.findall(r'\d+', top_text)
                    coord_match = re.search(r'(\d+)\s*/\s*(\d+)', top_text)
                    if top_nums and coord_match:
                        coord_str    = coord_match.group(0)
                        before_coord = top_text[:top_text.index(coord_str)]
                        pre_nums     = re.findall(r'\d+', before_coord)
                        region_id    = int(pre_nums[0]) if pre_nums else int(top_nums[0])
                        if len(str(region_id)) == 5:
                            region_id = int(str(region_id)[:4])
                        res["region"] = [region_id,
                                         int(coord_match.group(1)),
                                         int(coord_match.group(2))]

                    count_matches = re.findall(r'[xX\*]\s*(\d+)', bottom_text)
                    if count_matches:
                        res["count"] = count_matches[-1]
                    else:
                        bot_nums = re.findall(r'\d+', bottom_text)
                        if bot_nums:
                            res["count"] = bot_nums[-1]
                except Exception as e:
                    print(f"  [OCR error slot {slot_id}]: {e}")
        return res

    def debug_roi_view(self, screen):
        """
        Show the extended-ROI area and slot_topright matches after NMS,
        colour-coded by score. Prints score histogram to console.
        """
        tpl_left  = self._tpl_left
        tpl_right = self._tpl_right
        tpl_tr    = self._tpl_tr
        viz = screen.copy()

        if tpl_left is not None and tpl_right is not None:
            res_l = cv2.matchTemplate(screen, tpl_left,  cv2.TM_CCOEFF_NORMED)
            res_r = cv2.matchTemplate(screen, tpl_right, cv2.TM_CCOEFF_NORMED)
            _, val_l, _, loc_l = cv2.minMaxLoc(res_l)
            _, val_r, _, loc_r = cv2.minMaxLoc(res_r)
            print(f"  [ROITest] anchor_left={val_l:.3f}  anchor_right={val_r:.3f}")

            grid_x = loc_l[0]
            grid_y = max(loc_l[1] + tpl_left.shape[0], loc_r[1] + tpl_right.shape[0])
            grid_w = (loc_r[0] + tpl_right.shape[1]) - grid_x
            grid_h = int(grid_w * (417 / 1198))

            ext_y1 = grid_y
            ext_y2 = min(screen.shape[0], grid_y + grid_h)
            cv2.rectangle(viz, (grid_x, ext_y1), (grid_x + grid_w, ext_y2), (0, 140, 255), 2)
            cv2.rectangle(viz, (grid_x, grid_y),  (grid_x + grid_w, grid_y + grid_h), (0, 0, 255), 2)
            cv2.putText(viz, f"L={val_l:.2f} R={val_r:.2f}",
                        (grid_x, max(8, grid_y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            roi_img = screen[ext_y1:ext_y2, grid_x:grid_x + grid_w]
            roi_ox, roi_oy = grid_x, ext_y1
        else:
            print("  [ROITest] anchor templates missing, scanning full screen")
            roi_img = screen
            roi_ox, roi_oy = 0, 0

        if tpl_tr is not None:
            th, tw  = tpl_tr.shape[:2]
            hits    = find_all_matches(roi_img, tpl_tr, threshold=0.40)
            print(f"  [ROITest] tpl_tr size=({tw}x{th})  NMS hits (>0.40): {len(hits)}")

            res_full = cv2.matchTemplate(roi_img, tpl_tr, cv2.TM_CCOEFF_NORMED)
            scored   = []
            for (x, y, w, h) in hits:
                ry = min(y, res_full.shape[0] - 1)
                rx = min(x, res_full.shape[1] - 1)
                scored.append((float(res_full[ry, rx]), x, y, w, h))
            scored.sort(reverse=True)

            top10 = [f"{s:.3f}" for s, *_ in scored[:10]]
            print(f"  [ROITest] top-10 NMS scores: {' '.join(top10)}")
            n85  = sum(s >= 0.85 for s, *_ in scored)
            n70  = sum(0.70 <= s < 0.85 for s, *_ in scored)
            nlow = sum(s < 0.70 for s, *_ in scored)
            print(f"  [ROITest] >=0.85: {n85}  0.70-0.85: {n70}  <0.70: {nlow}")

            for (s, x, y, w, h) in scored:
                sx, sy = roi_ox + x, roi_oy + y
                color = (0, 255, 0) if s >= 0.85 else ((0, 200, 255) if s >= 0.70 else (100, 100, 255))
                cv2.rectangle(viz, (sx, sy), (sx + w, sy + h), color, 2)
                cv2.putText(viz, f"{s:.2f}", (sx, max(0, sy - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        else:
            print("  [ROITest] slot_topright.png missing")

        cv2.imshow("ROI Debug", viz)
        cv2.waitKey(1)
