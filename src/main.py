import sys, time, os, cv2, win32gui, win32con, win32api
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
from in_civi_operation import InCiviOperation
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
        
        try:
            import torch
            _dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        except Exception:
            _dev = "CPU"
        self.label = QLabel(f"TAS ACTIVE ({_dev})")
        self.label.setStyleSheet("color: #00FF00; font-weight: bold; background-color: rgba(0,0,0,120); padding: 5px;")
        self.main_layout.addWidget(self.label)

        self.add_control_button("Map Scan Global", lambda: self.map_scan_global(50, 500))
        # self.add_control_button("Map Scan Full", self.map_scan_full)
        self.add_control_button("Map Scan Once", self.map_scan_once)
        self.add_control_button("Test Field Change", self.test_field_change)
        # self.add_control_button("Swap To Bottom Right", self.swap_to_bottom_right)
        # self.add_control_button("Swap To Bottom", self.swap_to_bottom)
        self.add_control_button("Test Alliance Region", self.test_alliance_region)
        self.add_control_button("Test Location", self.test_location)
        self.add_control_button("Save Screenshot (x)", self.save_screenshot)
        self.add_control_button("Quick Harvest", self.once_harv)
        self.add_control_button("Check Army", self.army_reader)
        # Population toggle button (click to enable/disable automatic population clicks)
        self.btn_population = QPushButton("Population")
        self.btn_population.setFixedHeight(28)
        self.btn_population.setStyleSheet("background-color: rgba(50, 50, 50, 200); color: white; padding: 5px; font-weight: bold;")
        self.btn_population.clicked.connect(self._toggle_population)
        self.main_layout.addWidget(self.btn_population)
        self.add_control_button("SHUT DOWN (space)", self.close_app, is_danger=True)

        self.target_title = "MuMu安卓" 
        self.ui_width = 360
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_position)
        self.timer.start(500)
        self.army_queue = []
        self.army_scanner = ArmyScanner()
        # English: Locked grid size from reference run: dist=194.74 → fw=194, fh=97
        self.field_scanner = FieldScanner(self.target_title, grid_fw=194, grid_fh=97, threshold=0.18)
        # Calibration disabled: we will not load per-image calibration JSONs.

        # Global space hotkey — forces immediate exit even when scan is blocking the main thread
        keyboard.add_hotkey('space', self.close_app)
        # Global hotkey 'x' to save screenshot quickly
        keyboard.add_hotkey('x', self.save_screenshot)
        # In-civi operation controller
        self.in_civi_op = InCiviOperation(self.target_title)

    def add_control_button(self, text, callback, is_danger=False):
        btn = QPushButton(text)
        if is_danger:
            style = "background-color: rgba(150, 0, 0, 200); color: white; padding: 5px; font-weight: bold;"
        else:
            style = "background-color: rgba(50, 50, 50, 200); color: white; padding: 5px; font-weight: bold;"
        btn.setStyleSheet(style)
        btn.clicked.connect(callback)
        self.main_layout.addWidget(btn)

    def close_app(self):
        import os as _os
        _os._exit(0)

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
        # Calibration handling removed — nothing to do here.
        return

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
            # No per-image calibration: click the centre
            click_x = cx
            click_y = cy
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
                print(f"[Click] raw_center=({cx},{cy}) shift={shift} -> click=({click_x},{click_y})")
                self.label.setText(f"Click({click_x},{click_y}) shift={shift}")
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
            # Anchor crop is detected in-memory for grid construction but we do not
            # save any anchor-focused image files to disk (per configuration).
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
        # legacy kept for compatibility — dispatch to field scanner
        self.label.setText("Scanning fields...")
        results = self.field_scanner.scan(status_cb=self.label.setText)
        hits_empty    = results.get("空粘土",     [])
        hits_occupied = results.get("被占领粘土", [])
        self.label.setText(f"empty={len(hits_empty)}  occupied={len(hits_occupied)}")

    def test_population(self):
        """Show ROIs used for prefer detection and population clicking for fine-tuning."""
        self.label.setText("Test Population: showing ROIs")
        try:
            self.in_civi_op.test_roi()
        except Exception as ex:
            self.label.setText(f"Test Population failed: {ex}")

    def _set_clickthrough(self, enable: bool):
        """Make the overlay transparent to mouse clicks during automation."""
        try:
            hwnd = int(self.winId())
            ex_style = win32api.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            if enable:
                win32api.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                                       ex_style | win32con.WS_EX_TRANSPARENT)
            else:
                win32api.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                                       ex_style & ~win32con.WS_EX_TRANSPARENT)
        except Exception:
            pass

    def map_scan_global(self, max_iterations=10, max_global_entries=300):
        """Start a fresh global scan session: reset global_list then run map_scan_full."""
        self._set_clickthrough(True)
        try:
            self.field_scanner.map_scan_global(
                target_title=self.target_title, status_cb=self.label.setText, max_iterations=max_iterations, max_global_entries=max_global_entries
            )
        except Exception as ex:
            self.label.setText(f"Map Scan Global 失败：{ex}")
        finally:
            self._set_clickthrough(False)

    def test_field_change(self):
        """Click zone ROI centre, wait 0.5 s, show calibration marker."""
        self._set_clickthrough(True)
        try:
            self.field_scanner.test_field_change(
                target_title=self.target_title, status_cb=self.label.setText
            )
        except Exception as ex:
            self.label.setText(f"Test Field Change 失败：{ex}")
        finally:
            self._set_clickthrough(False)

    def map_scan_once(self):
        """Scan for alliance tiles, retry with swap_to_bottom_right up to 2 times."""
        self._set_clickthrough(True)
        try:
            self.field_scanner.map_scan_once(
                target_title=self.target_title, status_cb=self.label.setText
            )
        except Exception as ex:
            self.label.setText(f"Map Scan Once 失败：{ex}")
        finally:
            self._set_clickthrough(False)

    def map_scan_full(self):
        """Full 18-step map sweep across 3 rows × 3 columns."""
        self._set_clickthrough(True)
        try:
            self.field_scanner.map_scan_full(
                target_title=self.target_title, status_cb=self.label.setText
            )
        except Exception as ex:
            self.label.setText(f"Map Scan Full 失败：{ex}")
        finally:
            self._set_clickthrough(False)

    def swap_to_left(self):
        """Scroll map left by 8 × fw / 1.3 pixels."""
        self._set_clickthrough(True)
        try:
            self.field_scanner.swap_to_left(self.target_title, status_cb=self.label.setText)
        except Exception as ex:
            self.label.setText(f"Swap To Left 失败：{ex}")
        finally:
            self._set_clickthrough(False)

    def swap_to_bottom_right(self):
        """Drag map from top-left toward bottom-right to scroll toward top-left."""
        self._set_clickthrough(True)
        try:
            self.field_scanner.swap_to_bottom_right(self.target_title, status_cb=self.label.setText)
        except Exception as ex:
            self.label.setText(f"Swap To Bottom Right 失败：{ex}")
        finally:
            self._set_clickthrough(False)

    def swap_to_bottom(self):
        """Drag map downward along vertical centre line."""
        self._set_clickthrough(True)
        try:
            self.field_scanner.swap_to_bottom(self.target_title, status_cb=self.label.setText)
        except Exception as ex:
            self.label.setText(f"Swap To Bottom 失败：{ex}")
        finally:
            self._set_clickthrough(False)

    def test_alliance_region(self):
        """Visualise the alliance info-panel ROI on the cropped capture."""
        try:
            self.field_scanner.test_alliance_region(self.target_title, status_cb=self.label.setText)
        except Exception as ex:
            self.label.setText(f"Test Alliance Region 失败：{ex}")

    def test_location(self):
        """Visualise and OCR zone-number + map-coordinate ROIs."""
        try:
            self.field_scanner.test_location(self.target_title, status_cb=self.label.setText)
        except Exception as ex:
            self.label.setText(f"Test Location 失败：{ex}")

    def _toggle_population(self):
        try:
            if self.in_civi_op.is_running():
                self.in_civi_op.stop()
                # reset button style
                self.btn_population.setStyleSheet("background-color: rgba(50, 50, 50, 200); color: white; padding: 5px; font-weight: bold;")
                self.label.setText("Population automation stopped")
            else:
                # start with status callback to update overlay label with detected mode
                self.in_civi_op.start(status_cb=self.label.setText)
                # green style when active
                self.btn_population.setStyleSheet("background-color: rgba(30, 120, 30, 220); color: white; padding: 5px; font-weight: bold;")
                self.label.setText("Population automation started")
        except Exception as ex:
            self.label.setText(f"Population toggle error: {ex}")

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