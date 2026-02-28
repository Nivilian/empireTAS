import win32gui, win32ui, win32con, ctypes
from PIL import Image
import numpy as np
import cv2, os, re, warnings

# English: Suppress unnecessary warnings from libraries
warnings.filterwarnings("ignore", category=UserWarning)

# --- OCR ENGINE INITIALIZATION ---
ocr_engine = None

def get_ocr_engine():
    """
    Initialize EasyOCR, preferring GPU if architecture is compatible, else CPU.
    """
    global ocr_engine
    if ocr_engine is None:
        try:
            import easyocr
            import torch
            use_gpu = False
            if torch.cuda.is_available():
                # English: Try a small tensor op to confirm GPU is actually functional
                try:
                    torch.tensor([1.0]).cuda()
                    use_gpu = True
                except Exception as gpu_err:
                    print(f"--- GPU test failed ({gpu_err}); falling back to CPU ---")
            ocr_engine = easyocr.Reader(['en'], gpu=use_gpu)
            device = f"GPU: {torch.cuda.get_device_name(0)}" if use_gpu else "CPU"
            print(f"--- OCR Engine Initialized on {device} ---")
        except Exception as e:
            print(f"OCR Init Error: {e}")
    return ocr_engine

def imread_safe(path):
    """
    Reads images using numpy to handle Chinese file paths.
    """
    try:
        if not os.path.exists(path): return None
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except: return None

def find_hwnd(window_title):
    found_windows = []
    def callback(h, extra):
        title = win32gui.GetWindowText(h)
        if window_title.lower() in title.lower() and win32gui.IsWindowVisible(h):
            found_windows.append((h, title))
    win32gui.EnumWindows(callback, None)
    return found_windows[0][0] if found_windows else 0

def capture_game_ignore_ui(window_title="MuMu"):
    hwnd = find_hwnd(window_title)
    if not hwnd: return None
    left, top, right, bot = win32gui.GetClientRect(hwnd)
    w, h = right - left, bot - top
    if w <= 0 or h <= 0: return None
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitMap)
    ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
    bmpinfo = saveBitMap.GetInfo()
    bmpstr = saveBitMap.GetBitmapBits(True)
    im = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX', 0, 1)
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

def find_image_and_location(target_img, template_name, threshold=0.8):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(base_dir, "images", template_name)
    template = imread_safe(template_path)
    if template is None: return None, 0
    tH, tW = template.shape[:2]
    best_match, max_val = None, -1
    for scale in np.linspace(0.8, 1.2, 7): 
        rw, rh = int(tW * scale), int(tH * scale)
        if rw > target_img.shape[1] or rh > target_img.shape[0]: continue
        res = cv2.matchTemplate(target_img, cv2.resize(template, (rw, rh)), cv2.TM_CCOEFF_NORMED)
        _, cv, _, cl = cv2.minMaxLoc(res)
        if cv > max_val: max_val, best_match = cv, (cl, rw, rh)
    if max_val >= threshold:
        (loc, w, h) = best_match
        return (loc[0] + w // 2, loc[1] + h // 2), max_val
    return None, max_val

def find_all_matches(screen, template, threshold=0.75):
    """Return all (x, y, w, h) template matches above threshold using NMS."""
    if template is None:
        return []
    th, tw = template.shape[:2]
    res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    locs = np.argwhere(res >= threshold)
    if len(locs) == 0:
        return []
    # English: Sort by score descending, greedily suppress nearby duplicates
    scored = sorted(locs, key=lambda p: -res[p[0], p[1]])
    kept = []
    suppressed = np.zeros(res.shape, dtype=bool)
    for y, x in scored:
        if suppressed[y, x]:
            continue
        kept.append((int(x), int(y), tw, th))
        y1, y2 = max(0, y - th), min(res.shape[0], y + th)
        x1, x2 = max(0, x - tw), min(res.shape[1], x + tw)
        suppressed[y1:y2, x1:x2] = True
    return kept

def debug_roi_view(screen):
    """
    Show the extended-ROI area and slot_topright matches after NMS,
    colour-coded by score. Prints score histogram to console.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    tpl_left  = imread_safe(os.path.join(base, "images", "army", "anchor_left.png"))
    tpl_right = imread_safe(os.path.join(base, "images", "army", "anchor_right.png"))
    tpl_tr    = imread_safe(os.path.join(base, "images", "army", "slot_topright.png"))

    viz = screen.copy()

    # --- locate grid anchors ---
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
        rough_ch = grid_h // 3

        ext_y1 = grid_y
        ext_y2 = min(screen.shape[0], grid_y + grid_h)

        # English: orange = search ROI, red = estimated 3-row grid area
        cv2.rectangle(viz, (grid_x, ext_y1), (grid_x + grid_w, ext_y2), (0, 140, 255), 2)
        cv2.rectangle(viz, (grid_x, grid_y), (grid_x + grid_w, grid_y + grid_h), (0, 0, 255), 2)
        cv2.putText(viz, f"L={val_l:.2f} R={val_r:.2f}", (grid_x, max(8, grid_y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        roi_img = screen[ext_y1:ext_y2, grid_x:grid_x + grid_w]
        roi_ox, roi_oy = grid_x, ext_y1
    else:
        print("  [ROITest] anchor templates missing, scanning full screen")
        roi_img = screen
        roi_ox, roi_oy = 0, 0

    # --- NMS-filtered matches only (no pixel-level flood) ---
    if tpl_tr is not None:
        th, tw = tpl_tr.shape[:2]
        hits = find_all_matches(roi_img, tpl_tr, threshold=0.40)
        print(f"  [ROITest] tpl_tr size=({tw}x{th})  NMS hits (>0.40): {len(hits)}")

        # English: score each NMS hit from the raw result map
        res_full = cv2.matchTemplate(roi_img, tpl_tr, cv2.TM_CCOEFF_NORMED)
        scored = []
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
            if s >= 0.85:
                color = (0, 255, 0)       # green
            elif s >= 0.70:
                color = (0, 200, 255)     # yellow
            else:
                color = (100, 100, 255)   # light red — likely false positives
            cv2.rectangle(viz, (sx, sy), (sx + w, sy + h), color, 2)
            cv2.putText(viz, f"{s:.2f}", (sx, max(0, sy - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    else:
        print("  [ROITest] slot_topright.png missing")

    cv2.imshow("ROI Debug", viz)

    # --- Show individual slot crops as a tiled grid (geometry-based, same as get_army_grid_slots) ---
    if tpl_left is not None and tpl_right is not None and val_l >= 0.7 and val_r >= 0.7:
        tpl_tr_d = imread_safe(os.path.join(base, "images", "army", "slot_topright.png"))
        tpl_br_d = imread_safe(os.path.join(base, "images", "army", "slot_bottomright.png"))
        scan_h = int(grid_w * (380 / 1198))
        col_w  = grid_w // 3
        zone_y1 = grid_y
        zone_y2 = min(screen.shape[0], grid_y + scan_h)
        zone_h  = zone_y2 - zone_y1
        row_h   = zone_h // 3

        crops = []
        for row_idx in range(3):
            geo_top    = zone_y1 + row_idx * row_h
            geo_bottom = geo_top + row_h
            for col_idx in range(3):
                geo_left  = grid_x + col_idx * col_w
                geo_right = geo_left + col_w
                # English: refine crop with tr/br if available
                if tpl_tr_d is not None and tpl_br_d is not None:
                    refined = _refine_slot_rect(screen,
                                                geo_left, geo_top, geo_right, geo_bottom,
                                                tpl_tr_d, tpl_br_d)
                else:
                    refined = None
                if refined:
                    left, top, sw, sh = refined
                    crop = screen[top:top+sh, left:left+sw].copy()
                else:
                    crop = screen[geo_top:geo_bottom, geo_left:geo_right].copy()
                # English: label with slot number
                cv2.putText(crop, str(row_idx * 3 + col_idx + 1), (4, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                crops.append(crop)

        # English: Tile into 3×3 mosaic
        tile_w, tile_h = 260, 90
        resized = [cv2.resize(c, (tile_w, tile_h)) for c in crops]
        rows_img = [np.hstack(resized[r*3:(r+1)*3]) for r in range(3)]
        mosaic = np.vstack(rows_img)
        cv2.imshow("Slot Crops (3x3)", mosaic)

    cv2.waitKey(1)


def _refine_slot_rect(screen, cell_left, cell_top, cell_right, cell_bottom,
                      tpl_tr, tpl_br):
    """
    Within the rough geometric cell, find slot_topright and slot_bottomright
    to compute a precise slot rectangle.
    Returns (left, top, slot_w, slot_h) or None if not found.
    """
    cell_roi = screen[cell_top:cell_bottom, cell_left:cell_right]
    cell_h, cell_w = cell_roi.shape[:2]

    # Search tr in top half of cell
    tr_zone = cell_roi[:cell_h // 2, :]
    tr_hits = find_all_matches(tr_zone, tpl_tr, threshold=0.65)
    if not tr_hits:
        return None
    # pick rightmost tr hit (closest to the slot's right edge)
    tx, ty, tw, th = max(tr_hits, key=lambda h: h[0])
    top   = cell_top + ty
    right = cell_left + tx + tw

    # Search br in bottom half of cell, x-aligned with tr
    br_zone = cell_roi[cell_h // 2:, :]
    br_hits = find_all_matches(br_zone, tpl_br, threshold=0.65)
    if br_hits:
        bx, by, bw, bh = max(br_hits, key=lambda h: h[0])
        bottom = cell_top + cell_h // 2 + by + bh
    else:
        bottom = cell_bottom  # geometry fallback

    slot_h = bottom - top
    if slot_h <= 0:
        return None
    slot_w = max(1, int(slot_h * (389 / 105)))
    left   = right - slot_w
    return (left, top, slot_w, slot_h)


def get_army_grid_slots(screen, scan_zone="top"):
    """
    1. Locate grid via anchors (geometry).
    2. Divide scan zone (380/1198) into 9 rough cells.
    3. Refine each cell with slot_topright / slot_bottomright to get precise rect.
    scan_zone: "top" (default) | "bottom"
    grid_h (417/1198) is kept in grid_info for scroll distance computation.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    tpl_left  = imread_safe(os.path.join(base, "images", "army", "anchor_left.png"))
    tpl_right = imread_safe(os.path.join(base, "images", "army", "anchor_right.png"))
    tpl_tr    = imread_safe(os.path.join(base, "images", "army", "slot_topright.png"))
    tpl_br    = imread_safe(os.path.join(base, "images", "army", "slot_bottomright.png"))

    if tpl_left is None or tpl_right is None:
        print("Missing anchor_left.png or anchor_right.png"); return [], {}
    if tpl_tr is None or tpl_br is None:
        print("Missing slot_topright.png or slot_bottomright.png"); return [], {}

    # --- Locate grid via left/right anchors ---
    res_l = cv2.matchTemplate(screen, tpl_left,  cv2.TM_CCOEFF_NORMED)
    res_r = cv2.matchTemplate(screen, tpl_right, cv2.TM_CCOEFF_NORMED)
    _, val_l, _, loc_l = cv2.minMaxLoc(res_l)
    _, val_r, _, loc_r = cv2.minMaxLoc(res_r)
    if val_l < 0.7 or val_r < 0.7:
        print(f"Grid anchors not found (l={val_l:.2f} r={val_r:.2f})"); return [], {}

    grid_x = loc_l[0]
    grid_y = max(loc_l[1] + tpl_left.shape[0], loc_r[1] + tpl_right.shape[0])
    grid_w = (loc_r[0] + tpl_right.shape[1]) - grid_x
    grid_h = int(grid_w * (417 / 1198))   # full height — for scroll geometry only
    col_w  = grid_w // 3

    # --- Determine scan zone ---
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

    for row_idx in range(3):
        geo_top    = scan_y1 + row_idx * row_h
        geo_bottom = geo_top + row_h
        for col_idx in range(3):
            idx       = row_idx * 3 + col_idx
            geo_left  = grid_x + col_idx * col_w
            geo_right = geo_left + col_w

            # English: Try to refine using tr/br; fall back to geometry
            refined = _refine_slot_rect(screen,
                                        geo_left, geo_top, geo_right, geo_bottom,
                                        tpl_tr, tpl_br)
            if refined:
                left, top, slot_w, slot_h = refined
                bottom = top + slot_h
                right  = left + slot_w
            else:
                left, top     = geo_left, geo_top
                right, bottom = geo_right, geo_bottom
                slot_w, slot_h = col_w, row_h

            cell_img = screen[top:bottom, left:right]
            slots.append({'image': cell_img, 'index': idx,
                          'rect': (left, top, slot_w, slot_h)})

            # English: debug overlay
            color = (0, 255, 0) if refined else (0, 165, 255)  # green=refined, orange=fallback
            cv2.rectangle(viz, (left, top), (right, bottom), color, 1)
            dw1 = int(slot_w * (1/4))
            dw2 = int(slot_w * (14/27))
            cv2.line(viz, (left + dw1, top), (left + dw1, bottom), (255, 0, 0), 1)
            cv2.line(viz, (left + dw1 + dw2, top), (left + dw1 + dw2, bottom), (0, 255, 255), 1)
            cv2.putText(viz, str(idx + 1), (left + 5, top + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # English: Draw row-height arrow (slot1 top → slot4 top)
    measured_row_h = row_h
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

def read_army_count(screen):
    """
    OCR the '当前 62 / 上限 64' text to the right of anchor_left.
    Returns (current, maximum) as ints, or (0, 0) on failure.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    tpl = imread_safe(os.path.join(base, "images", "army", "anchor_left.png"))
    if tpl is None: return 0, 0
    res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
    _, val, _, loc = cv2.minMaxLoc(res)
    if val < 0.6: return 0, 0
    # English: Crop a strip to the right of the anchor, same height
    ax, ay = loc[0] + tpl.shape[1], loc[1]
    strip = screen[ay:ay + tpl.shape[0], ax:ax + int(tpl.shape[1] * 5)]
    reader = get_ocr_engine()
    if reader is None: return 0, 0
    try:
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        results = reader.readtext(enlarged, detail=0, allowlist='0123456789/')
        raw = " ".join(results)
        nums = re.findall(r'\d+', raw)
        print(f"  [ArmyCount OCR] raw={repr(raw)} nums={nums}")
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
        elif len(nums) == 1:
            return int(nums[0]), int(nums[0])
    except Exception as e:
        print(f"  [ArmyCount OCR error]: {e}")
    return 0, 0


def perform_army_scroll(window_title, grid_info, drag_px):
    """
    Drag upward inside the army grid by the given drag_px pixels.
    drag_px is computed externally as (slot4_top - global_slot1_top) * 1.1
    """
    import pyautogui, time
    hwnd = find_hwnd(window_title)
    if not hwnd: return
    gx = grid_info["grid_x"]
    gy = grid_info["grid_y"]
    gw = grid_info["grid_w"]
    gh = grid_info["grid_h"]
    # English: Drag from one slot-height above the grid bottom, upward by drag_px
    slot_h  = grid_info.get("slot_h", gh // 3)
    cx = gx + gw // 2
    start_y = gy + gh - 5 - slot_h
    end_y   = start_y - drag_px
    start_screen = win32gui.ClientToScreen(hwnd, (cx, start_y))
    end_screen   = win32gui.ClientToScreen(hwnd, (cx, end_y))
    dpi = ctypes.windll.user32.GetDpiForSystem()
    pixels_per_cm = dpi / 2.54
    duration = drag_px / (1.2 * pixels_per_cm)   # seconds at 1.2 cm/s
    print(f"  [Scroll] drag_px={drag_px}  duration={duration:.2f}s")
    pyautogui.moveTo(start_screen[0], start_screen[1])
    pyautogui.mouseDown(button='left')
    time.sleep(0.5)
    pyautogui.moveTo(end_screen[0], end_screen[1], duration=duration)
    time.sleep(1.0)
    pyautogui.mouseUp(button='left')


def identify_army_content(cell_img, status_tpls, type_tpls, slot_id=0):
    res = {"unit": "Empty", "status": "Idle", "region": [], "count": ""}
    if cell_img is None: return res

    # English: Save slot image to images/ocr/ for debugging (overwrite if exists)
    base = os.path.dirname(os.path.abspath(__file__))
    ocr_dir = os.path.join(base, "images", "ocr")
    os.makedirs(ocr_dir, exist_ok=True)
    cv2.imencode('.png', cell_img)[1].tofile(os.path.join(ocr_dir, f"slot_{slot_id}.png"))

    h, w = cell_img.shape[:2]
    # English: Horizontal split ratios — left: 1/4 (unit icon), middle: 14/27 (region/coord/count), right: 2/7 (status icon)
    w1 = int(w * (1/4))        # left partition width  (≈25%)
    w2 = int(w * (14/27))      # middle partition width (≈48%)
    # right partition is the remainder

    left_part   = cell_img[:, :w1]
    middle_part = cell_img[:, w1:w1+w2]
    right_part  = cell_img[:, w1+w2:]

    # 1. Type — pick highest confidence match; scale template down if larger than search area
    best_unit, best_val = "Empty", 0.0
    conf_lines = []
    for name, tpl in type_tpls.items():
        if tpl is None: continue
        th, tw = tpl.shape[:2]
        lh, lw = left_part.shape[:2]
        # English: Scale template down proportionally so it fits inside left_part
        if tw > lw or th > lh:
            scale = min(lw / tw, lh / th) * 0.95
            tpl = cv2.resize(tpl, (max(1, int(tw * scale)), max(1, int(th * scale))))
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
    for name, tpl in status_tpls.items():
        if tpl is not None:
            th, tw = tpl.shape[:2]
            if right_part.shape[0] < th or right_part.shape[1] < tw: continue
            val = cv2.minMaxLoc(cv2.matchTemplate(right_part, tpl, cv2.TM_CCOEFF_NORMED))[1]
            if val > (0.53 if name == "Moving" else 0.7):
                res["status"] = name; break

    # 3. EasyOCR (CPU Mode) — split middle into top (region/coord) and bottom (count)
    if res["unit"] != "Empty":
        reader = get_ocr_engine()
        if reader:
            try:
                mh = middle_part.shape[0]
                # English: Top ~55% of middle cell contains "2963区 2/18"
                top_part    = middle_part[:int(mh * 0.55), :]
                # English: Bottom ~45% of middle cell contains "猛犸 x99"
                bottom_part = middle_part[int(mh * 0.55):, :]

                def ocr_region(img, allowlist):
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    enlarged = cv2.resize(gray, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    results = reader.readtext(enlarged, detail=0, allowlist=allowlist)
                    return " ".join(results)

                top_text    = ocr_region(top_part,    '0123456789/区')
                bottom_text = ocr_region(bottom_part, '0123456789xX*')
                print(f"  [OCR slot {slot_id}] top={repr(top_text)}  bottom={repr(bottom_text)}")

                # English: Region number: first 3-4 digit group in the top line
                # "区" may be misread (as 1 etc.), so we match the first large number then the /x/y pair
                top_nums = re.findall(r'\d+', top_text)
                coord_match = re.search(r'(\d+)\s*/\s*(\d+)', top_text)
                if top_nums and coord_match:
                    # English: region id is the number that appears BEFORE the x/y pair
                    coord_str = coord_match.group(0)
                    before_coord = top_text[:top_text.index(coord_str)]
                    pre_nums = re.findall(r'\d+', before_coord)
                    region_id = int(pre_nums[0]) if pre_nums else int(top_nums[0])
                    # English: strip the trailing misread char of region (e.g. "29631" → 2963)
                    # region id is always 4 digits for this game
                    if len(str(region_id)) == 5:
                        region_id = int(str(region_id)[:4])
                    coord_x = int(coord_match.group(1))
                    coord_y = int(coord_match.group(2))
                    res["region"] = [region_id, coord_x, coord_y]

                # English: Count: use the LAST x/X/* match to avoid e.g. "4X4 X2" → 4
                count_matches = re.findall(r'[xX\*]\s*(\d+)', bottom_text)
                if count_matches:
                    res["count"] = count_matches[-1]
                else:
                    # English: fallback — last number in bottom line
                    bot_nums = re.findall(r'\d+', bottom_text)
                    if bot_nums:
                        res["count"] = bot_nums[-1]
            except Exception as e:
                print(f"  [OCR error slot {slot_id}]: {e}")
    return res