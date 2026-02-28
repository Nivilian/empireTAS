import sys, time, os, cv2, win32gui
import pygetwindow as gw
import pyautogui
import keyboard
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer
import vision 

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
        self.add_control_button("SHUT DOWN", self.close_app, is_danger=True)

        self.target_title = "MuMu安卓" 
        self.ui_width = 360
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_position)
        self.timer.start(500)
        self.army_queue = []

        # English: Global ESC hotkey — works even when the overlay is not focused
        keyboard.add_hotkey('esc', self.close_app)

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
            screen = vision.capture_game_ignore_ui(self.target_title)
            if screen is None: continue
            search_area = screen if not roi else screen[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
            result, _ = vision.find_image_and_location(search_area, template_name, threshold)
            if result:
                cx, cy = (result[0] + roi[0], result[1] + roi[1]) if roi else result
                hwnd = vision.find_hwnd(self.target_title)
                if hwnd:
                    screen_point = win32gui.ClientToScreen(hwnd, (int(cx), int(cy)))
                    pyautogui.click(screen_point[0], screen_point[1])
                    return True 
            time.sleep(0.2)
        return False

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
        screen = vision.capture_game_ignore_ui(self.target_title)
        if screen is None: return
        h, w = screen.shape[:2]
        left_roi = (0, int(h * 0.30), int(w * 0.10), int(h * 0.40))
        if not self.click_multiple("army/armyButton.png", threshold=0.5, timeout=3, roi=left_roi): return
        time.sleep(1.0)

        screen = vision.capture_game_ignore_ui(self.target_title)
        if screen is None: return

        base = os.path.dirname(os.path.abspath(__file__))
        type_tpls = {
            "Xiaowei":        vision.imread_safe(os.path.join(base, "images/army/types/连弩校尉.png")),
            "Nuwa":           vision.imread_safe(os.path.join(base, "images/army/types/女娲.png")),
            "Mammoth":        vision.imread_safe(os.path.join(base, "images/army/types/猛犸.png")),
            "PersianCavalry": vision.imread_safe(os.path.join(base, "images/army/types/波斯轻骑兵.png")),
            "Centaur":        vision.imread_safe(os.path.join(base, "images/army/types/半人马.png")),
            "Spearman":       vision.imread_safe(os.path.join(base, "images/army/types/标枪手.png")),
        }
        status_tpls = {
            "Moving":    vision.imread_safe(os.path.join(base, "images/army/status/armyMoving.png")),
            "Stationed": vision.imread_safe(os.path.join(base, "images/army/status/armyStationed.png")),
            "Waiting":   vision.imread_safe(os.path.join(base, "images/army/status/armyWaiting.png")),
            "InBattle":  vision.imread_safe(os.path.join(base, "images/army/status/armyInBattle.png"))
        }

        # English: Read total army count from UI header
        current, maximum = vision.read_army_count(screen)
        if current == 0:
            print("[army_reader] Could not read army count, defaulting to 9")
            current = 9
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
            screen = vision.capture_game_ignore_ui(self.target_title)
            if screen is None: break
            scan_zone = "bottom" if use_bottom_scan else "top"
            use_bottom_scan = False   # English: consume the flag — reset after each scan
            slots, grid_info = vision.get_army_grid_slots(screen, scan_zone=scan_zone)
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
                res = vision.identify_army_content(
                    slot['image'], status_tpls, type_tpls,
                    slot_id=armies_read + 1)
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
                vision.perform_army_scroll(self.target_title, grid_info, drag_px)
                pass_num += 1
            else:
                break

        print(f"\n--- FINAL ARMY LIST ({armies_read}/{current}) ---")
        for unit in self.army_queue:
            print(f"#{unit['slot']}: {unit['unit']} | {unit['status']} | Region: {unit['region']} | x{unit['count']}")
        print("\n\n")
        self.label.setText(f"LOADED {armies_read}/{current} UNITS")

    def roi_test(self):
        """Show extended ROI and all slot_topright candidates colour-coded by match score."""
        self.label.setText("ROI Test...")
        screen = vision.capture_game_ignore_ui(self.target_title)
        if screen is None:
            self.label.setText("ROI Test: no capture"); return
        vision.debug_roi_view(screen)
        self.label.setText("ROI Test done — see console + debug window")

    def scroll_test(self):
        """Detect grid, perform exactly one scroll, then sleep 2s so result can be inspected."""
        self.label.setText("Scroll Test...")
        screen = vision.capture_game_ignore_ui(self.target_title)
        if screen is None:
            self.label.setText("Scroll Test: no capture"); return
        slots, grid_info = vision.get_army_grid_slots(screen)
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
        vision.perform_army_scroll(self.target_title, grid_info, drag_px)
        time.sleep(2.0)   # pause so you can see the result
        self.label.setText(f"Scroll Test done (dragged {drag_px}px)")

    def run_vision_test(self):
        img = vision.capture_game_ignore_ui(self.target_title)
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