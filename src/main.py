import sys, time, os, cv2, win32gui
import math
import numpy as np
import pygetwindow as gw
import pyautogui
import keyboard
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer
from vision_core import capture_game_ignore_ui, find_image_and_location, find_hwnd, imread_safe, find_all_matches
from army_scanner import ArmyScanner
from map_scanner import FieldScanner
import json
import glob

class EmpireOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowOpacity(0.8)
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        
        # English: Update label to show CPU mode for clarity
        self.label = QLabel("TAS ACTIVE (CPU Mode)")
        self.label.setStyleSheet("color: #00FF00; font-weight: bold; background-color: rgba(0,0,0,120); padding: 5px;")
        self.main_layout.addWidget(self.label)

        self.add_control_button("Quick Harvest", self.once_harv)
        self.add_control_button("Check Army", self.army_reader)
        self.add_control_button("Scroll Test", self.scroll_test)
        self.add_control_button("ROI Test", self.roi_test)
        self.add_control_button("Capture Test", self.run_vision_test)
        self.add_control_button("Test Field", self.test_field)
        self.add_control_button("Save Screenshot (x)", self.save_screenshot)
        self.add_control_button("SHUT DOWN (esc)", self.close_app, is_danger=True)

        self.target_title = "MuMu安卓" 
        self.ui_width = 360
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_position)
        self.timer.start(500)
        self.army_queue = []
        self.army_scanner = ArmyScanner()
        # English: Locked grid size from reference run: dist=194.74 → fw=194, fh=97
        self.field_scanner = FieldScanner(self.target_title, grid_fw=194, grid_fh=97, threshold=0.18)
        # load optional click calibration (normalized offsets)
        self._load_click_calibration()

        # English: Global ESC hotkey — works even when the overlay is not focused
        keyboard.add_hotkey('esc', self.close_app)
        # Global hotkey 'x' to save screenshot quickly
        keyboard.add_hotkey('x', self.save_screenshot)

    def add_control_button(self, text, callback, is_danger=False):
        btn = QPushButton(text)
        if is_danger:
            style = "background-color: rgba(150, 0, 0, 200); color: white; padding: 5px; font-weight: bold;"
        else:
            style = "background-color: rgba(50, 50, 50, 200); color: white; padding: 5px; font-weight: bold;"
        btn.setStyleSheet(style)
        btn.clicked.connect(callback)
        self.main_layout.addWidget(btn)

    def close_app(self): QApplication.quit()

    def click_multiple(self, template_name, threshold=0.7, timeout=5, roi=None):
        start_time = time.time()
        while time.time() - start_time < timeout:
            screen = capture_game_ignore_ui(self.target_title)
            if screen is None: continue
            search_area = screen if not roi else screen[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
            result, conf = find_image_and_location(search_area, template_name, threshold)
            if result:
                cx, cy = (result[0] + roi[0], result[1] + roi[1]) if roi else result
                print(f"  [Click] {template_name}  conf={conf:.3f}  pos=({int(cx)}, {int(cy)})")
                hwnd = find_hwnd(self.target_title)
                if hwnd:
                    screen_point = win32gui.ClientToScreen(hwnd, (int(cx), int(cy)))
                    pyautogui.click(screen_point[0], screen_point[1])
                    return True 
            time.sleep(0.2)
        return False

    def _load_click_calibration(self):
        """Load calibration JSONs exported by the labeler and compute normalized offset.
        The labeler exports per-image calibration files to images/labeledImages/calibration/<base>.json
        Each contains entries with cx,cy coordinates in image pixels. We compute (cx - w/2)/w
        and (cy - h/2)/h averaged across samples to produce a normalized offset.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        calib_dir = os.path.join(base_dir, "images", "labeledImages", "calibration")
        backup_dir = os.path.join(base_dir, "images", "trainingBackup")
        self.cal_nx = 0.0
        self.cal_ny = 0.0
        self.cal_count = 0
        if not os.path.isdir(calib_dir):
            return
        vals_x = []
        vals_y = []
        for jf in glob.glob(os.path.join(calib_dir, "*.json")):
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    pts = json.load(f)
            except Exception:
                continue
            base = os.path.splitext(os.path.basename(jf))[0]
            # find corresponding backup image to get image size
            candidates = [p for p in glob.glob(os.path.join(backup_dir, base + ".*"))]
            if not candidates:
                # try any image starting with base
                candidates = [p for p in glob.glob(os.path.join(backup_dir, base + "*"))]
            if not candidates:
                continue
            img_path = candidates[0]
            try:
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
            except Exception:
                continue
            for p in pts:
                if not ("cx" in p and "cy" in p):
                    continue
                cx = float(p["cx"])
                cy = float(p["cy"])
                nx = (cx - (w / 2.0)) / w
                ny = (cy - (h / 2.0)) / h
                vals_x.append(nx)
                vals_y.append(ny)
        if vals_x:
            self.cal_nx = float(np.mean(vals_x))
            self.cal_ny = float(np.mean(vals_y))
            self.cal_count = len(vals_x)
            # persist a quick summary for debugging
            outp = {
                "nx": self.cal_nx,
                "ny": self.cal_ny,
                "count": self.cal_count,
            }
            try:
                os.makedirs(os.path.join(base_dir, "images", "labeledImages", "calibration"), exist_ok=True)
                with open(os.path.join(base_dir, "images", "labeledImages", "calibration", "offset.json"), "w", encoding="utf-8") as fo:
                    json.dump(outp, fo, ensure_ascii=False, indent=2)
            except Exception:
                pass

    def once_harv(self):
        self.label.setText("Harvesting...")
        self.click_multiple("baseUI/inCivi/close.png", timeout=1)
        time.sleep(0.5)
        success = self.click_multiple("baseUI/inCivi/quickHarv.png")
        self.label.setText("DONE" if success else "FAILED")

    def army_reader(self):
        self.label.setText("Scanning...")
        time.sleep(0.3)
        self.click_multiple("baseUI/inCivi/close.png", timeout=0.5)
        time.sleep(0.3)
        screen = capture_game_ignore_ui(self.target_title)
        if screen is None: return
        h, w = screen.shape[:2]
        left_roi = (0, int(h * 0.30), int(w * 0.10), int(h * 0.40))
        if not self.click_multiple("army/armyButton.png", threshold=0.5, timeout=3, roi=left_roi): return
        time.sleep(1.0)

        screen = capture_game_ignore_ui(self.target_title)
        if screen is None: return

        # English: Read total army count from UI header
        current, maximum = self.army_scanner.read_count(screen)
        if current == 0:
            print("[army_reader] Could not read army count, defaulting to 9")
            current = 9
        if maximum == 0:
            maximum = current   # fallback if upper limit OCR failed
        print(f"\n--- ARMY QUEUE UPDATE --- ({current}/{maximum}) ---")

        self.army_queue = []
        armies_read = 0
        pass_num = 0
        grid_info = None
        global_slot1_top = None   # English: top edge of slot1 on first scan, fixed reference
        ref_slot_w = None         # English: reference slot width from first scan
        ref_slot_h = None         # English: reference slot height from first scan
        SIZE_TOL = 0.15           # English: 15% deviation triggers a warning
        # English: when total>9 and remaining<=3 after a pass, switch to bottom-zone scan
        use_bottom_scan = False

        while armies_read < current:
            screen = capture_game_ignore_ui(self.target_title)
            if screen is None: break
            scan_zone = "bottom" if use_bottom_scan else "top"
            use_bottom_scan = False   # English: consume the flag — reset after each scan
            slots, grid_info = self.army_scanner.get_slots(screen, scan_zone=scan_zone)
            if not slots:
                print("[army_reader] No slots found, aborting")
                break

            slot_by_idx = {s['index']: s for s in slots}

            if pass_num == 0:
                # English: Record slot1's top edge and slot dimensions as fixed references
                if 0 in slot_by_idx:
                    global_slot1_top = slot_by_idx[0]['rect'][1]
                    ref_slot_w, ref_slot_h = slot_by_idx[0]['rect'][2], slot_by_idx[0]['rect'][3]
                    print(f"  [army_reader] global_slot1_top={global_slot1_top}px  ref_size=({ref_slot_w}x{ref_slot_h})")
                # English: First view — read up to 9 (or fewer if total < 9)
                read_slots = slots[:min(9, current)]
            else:
                # English: Validate slot sizes against reference; warn on large deviations
                if ref_slot_w and ref_slot_h:
                    for s in slots:
                        sw, sh = s['rect'][2], s['rect'][3]
                        dw = abs(sw - ref_slot_w) / ref_slot_w
                        dh = abs(sh - ref_slot_h) / ref_slot_h
                        if dw > SIZE_TOL or dh > SIZE_TOL:
                            print(f"  [WARNING] slot {s['index']+1} size ({sw}x{sh}) deviates from ref "
                                  f"({ref_slot_w}x{ref_slot_h}): dw={dw:.1%} dh={dh:.1%}")
                # English: After each scroll, 3 new armies appear at slots 7-9
                remaining  = current - armies_read
                new_count  = min(3, remaining)
                read_slots = slots[9 - new_count:9]

            for slot in read_slots:
                res = self.army_scanner.identify_content(
                    slot['image'], slot_id=armies_read + 1)
                self.army_queue.append({"slot": armies_read + 1, **res})
                armies_read += 1

            self.label.setText(f"Read {armies_read}/{current}...")

            if armies_read < current:
                if grid_info is None or global_slot1_top is None: break
                # English: if total > 9 and only ≤3 slots remain, use bottom zone for next scan
                remaining_next = current - armies_read
                if current > 9 and remaining_next <= 3:
                    use_bottom_scan = True
                    print(f"  [army_reader] Switching to bottom scan zone for next pass "
                          f"(remaining={remaining_next})")
                # English: Compute drag distance: current slot4 top minus the fixed slot1 baseline, x1.1
                if 3 in slot_by_idx:
                    slot4_top = slot_by_idx[3]['rect'][1]
                    drag_px = int((slot4_top - global_slot1_top) * 1.1)
                else:
                    drag_px = grid_info["slot_h"]
                print(f"  [army_reader] pass={pass_num}  slot4_top={slot_by_idx.get(3, {}).get('rect', [0,0])[1]}  drag_px={drag_px}")
                self.army_scanner.scroll(self.target_title, grid_info, drag_px)
                pass_num += 1
            else:
                break

        print(f"\n--- CURRENT ARMY STRUCTURE ({armies_read}/{maximum}) ---")
        for unit in self.army_queue:
            print(f"#{unit['slot']}: {unit['unit']} | {unit['status']} | Region: {unit['region']} | x{unit['count']}")
        print("\n\n")
        self.label.setText(f"LOADED {armies_read}/{maximum} UNITS")

    def save_screenshot(self):
        import uuid
        self.hide()
        time.sleep(0.25)
        # Click the game centre to trigger selection diamond, then capture
        hwnd = find_hwnd(self.target_title)
        cx = cy = None
        if hwnd:
            rect = win32gui.GetClientRect(hwnd)
            rect_w, rect_h = (rect[2] - rect[0]), (rect[3] - rect[1])
            cx, cy = rect_w // 2, rect_h // 2
            # apply calibration normalized offsets if available
            try:
                off_x = int(self.cal_nx * rect_w)
                off_y = int(self.cal_ny * rect_h)
            except Exception:
                off_x = 0
                off_y = 0
            click_x = cx + off_x
            click_y = cy + off_y
            # apply small up-left diagonal shift of 0.1 * edge length (if fw/fh known)
            try:
                fw, fh = self.field_scanner.grid_fw, self.field_scanner.grid_fh
                if fw and fh:
                    s = math.hypot(fw / 2.0, fh / 2.0)
                    shift = int(round(0.1 * s / math.sqrt(2)))
                else:
                    shift = 0
            except Exception:
                shift = 0
            click_x -= shift
            click_y -= shift
            # Debug: print and show computed click and shift
            try:
                print(f"[ClickCal] raw_center=({cx},{cy}) cal_offset=({off_x},{off_y}) shift={shift} -> click=({click_x},{click_y})")
                self.label.setText(f"Click({click_x},{click_y}) shift={shift} cal_samples={getattr(self,'cal_count',0)}")
            except Exception:
                pass
            pt = win32gui.ClientToScreen(hwnd, (click_x, click_y))
            pyautogui.click(pt[0], pt[1])
            time.sleep(0.6)

        screen = capture_game_ignore_ui(self.target_title)
        self.show()
        if screen is None:
            self.label.setText("截图失败，找不到游戏窗口")
            return

        # Do NOT crop the top — keep the full capture so the game's grid remains intact
        img = screen

        # Detect yellow diamond anchor (pass clicked centre in full-image coords)
        fw, fh = self.field_scanner.grid_fw, self.field_scanner.grid_fh
        click_x = cx
        click_y = cy

        _, _, anchor_rect = self.field_scanner.detect_yellow_frame(img, click_x, click_y)

        # If not found, try a sequence of nearby clicks: left, up, right x2, down x2, left x2, ...
        if anchor_rect is None and cx is not None and cy is not None:
            seq = [(-fw, 0), (0, -fh), (fw, 0), (fw, 0), (0, fh), (0, fh),
                   (-fw, 0), (-fw, 0), (0, -fh), (0, -fh), (fw, 0), (fw, 0)]
            cur_x, cur_y = cx, cy
            for dx, dy in seq:
                cur_x += dx
                cur_y += dy
                # click and recapture
                ptc = win32gui.ClientToScreen(hwnd, (int(cur_x), int(cur_y)))
                pyautogui.click(ptc[0], ptc[1])
                time.sleep(0.5)
                screen2 = capture_game_ignore_ui(self.target_title)
                if screen2 is None:
                    continue
                img2 = screen2
                ckx = cur_x
                cky = cur_y
                _, _, anchor_rect = self.field_scanner.detect_yellow_frame(img2, ckx, cky)
                if anchor_rect:
                    img = img2
                    break

        if anchor_rect:
            anchor_fx, anchor_fy, _, _ = anchor_rect
            # apply the same up-left diagonal shift to the saved screenshot's anchor
            try:
                fw, fh = self.field_scanner.grid_fw, self.field_scanner.grid_fh
                if fw and fh:
                    s = math.hypot(fw / 2.0, fh / 2.0)
                    shift = int(round(0.1 * s / math.sqrt(2)))
                else:
                    shift = 0
            except Exception:
                shift = 0
            anchor_fx -= shift
            anchor_fy -= shift
            try:
                print(f"[AnchorCal] detected_anchor=({anchor_fx+shift},{anchor_fy+shift}) applied_shift={shift} -> anchor=({anchor_fx},{anchor_fy})")
            except Exception:
                pass
            # --- Save an anchor-focused crop (but keep full img for the main save) ---
            try:
                cx_tile = int(anchor_fx + fw // 2)
                cy_tile = int(anchor_fy + fh // 2)
                mult = 3
                half_w = max(int(mult * fw), fw)
                half_h = max(int(mult * fh), fh)
                x1 = max(0, cx_tile - half_w)
                y1 = max(0, cy_tile - half_h)
                x2 = min(img.shape[1], cx_tile + half_w)
                y2 = min(img.shape[0], cy_tile + half_h)
                anchor_crop = img[y1:y2, x1:x2].copy()
                if anchor_crop.size > 0:
                    # save anchor crop to a separate folder for labeler convenience
                    anchor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "trainingBackup", "anchors")
                    os.makedirs(anchor_dir, exist_ok=True)
                    anchor_fname = os.path.join(anchor_dir, uuid.uuid4().hex.upper() + "_ANCHOR.jpg")
                    cv2.imencode(".jpg", anchor_crop)[1].tofile(anchor_fname)
                    print(f"[Crop] saved anchor-crop {anchor_fname} size=({anchor_crop.shape[1]}x{anchor_crop.shape[0]}) crop_xy=({x1},{y1})")
            except Exception:
                pass
        else:
            # fallback: use centre
            h, w = img.shape[:2]
            anchor_fx, anchor_fy = w // 2 - fw // 2, h // 2 - fh // 2

        h, w = img.shape[:2]
        grid_cells = self.field_scanner.build_grid(anchor_fx, anchor_fy, fw, fh, 0, 0, w, h)
        # Do not draw a synthetic grid overlay on saved images — keep grid hidden
        # while still using it for snapping during labeling.
        # If you want the grid visible for debugging, uncomment the line below.
        # FieldScanner.draw_grid(img, grid_cells, color=(0,200,200), thickness=2)

        # Crop saved image to maintain aspect ratio 966:1718 (height:width),
        # keeping only the bottom region.
        try:
            h, w = img.shape[:2]
            target_h = int(round(w * 966.0 / 1718.0))
            if target_h < h:
                img = img[h - target_h:h, :, :].copy()
        except Exception:
            pass

        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "images", "trainingBackup")
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, uuid.uuid4().hex.upper() + ".jpg")
        cv2.imencode(".jpg", img)[1].tofile(fname)
        self.label.setText(f"已存到 trainingBackup/{os.path.basename(fname)}")

    def test_field(self):
        self.label.setText("Scanning fields...")
        results = self.field_scanner.scan(status_cb=self.label.setText)
        hits_empty    = results.get("空粘土",     [])
        hits_occupied = results.get("被占领粘土", [])
        self.label.setText(f"empty={len(hits_empty)}  occupied={len(hits_occupied)}")

    def roi_test(self):
        """Show extended ROI and all slot_topright candidates colour-coded by match score."""
        self.label.setText("ROI Test...")
        screen = capture_game_ignore_ui(self.target_title)
        if screen is None:
            self.label.setText("ROI Test: no capture"); return
        self.army_scanner.debug_roi_view(screen)
        self.label.setText("ROI Test done — see console + debug window")

    def scroll_test(self):
        """Detect grid, perform exactly one scroll, then sleep 2s so result can be inspected."""
        self.label.setText("Scroll Test...")
        screen = capture_game_ignore_ui(self.target_title)
        if screen is None:
            self.label.setText("Scroll Test: no capture"); return
        slots, grid_info = self.army_scanner.get_slots(screen)
        if not slots:
            self.label.setText("Scroll Test: grid not found"); return
        slot_by_idx = {s['index']: s for s in slots}
        if 0 in slot_by_idx and 3 in slot_by_idx:
            slot1_top = slot_by_idx[0]['rect'][1]
            slot4_top = slot_by_idx[3]['rect'][1]
            drag_px = int((slot4_top - slot1_top) * 1.1)
        else:
            drag_px = grid_info["slot_h"]
        print(f"[ScrollTest] drag_px={drag_px}px — performing one scroll")
        self.army_scanner.scroll(self.target_title, grid_info, drag_px)
        time.sleep(2.0)   # pause so you can see the result
        self.label.setText(f"Scroll Test done (dragged {drag_px}px)")

    def run_vision_test(self):
        img = capture_game_ignore_ui(self.target_title)
        if img is not None:
            cv2.imshow("Vision Test", img)
            cv2.waitKey(1)

    def update_position(self):
        try:
            wins = [w for w in gw.getAllWindows() if self.target_title.lower() in w.title.lower()]
            if wins:
                t = wins[0]
                if t.left > -10000: self.setGeometry(t.left + 10, t.top + 45, self.ui_width, self.layout().sizeHint().height())
        except: pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    overlay = EmpireOverlay()
    overlay.show()
    sys.exit(app.exec_())