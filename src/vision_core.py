import win32gui, win32ui, win32con, ctypes
from PIL import Image
import numpy as np
import cv2, os, warnings

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
    """Reads images using numpy to handle Chinese file paths."""
    try:
        if not os.path.exists(path): return None
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except:
        return None


def find_hwnd(window_title):
    found_windows = []
    def callback(h, extra):
        title = win32gui.GetWindowText(h)
        if window_title.lower() in title.lower() and win32gui.IsWindowVisible(h):
            found_windows.append((h, title))
    win32gui.EnumWindows(callback, None)
    return found_windows[0][0] if found_windows else 0


def get_game_rect(screen, min_content_ratio=0.5):
    """
    Detect the actual game-content bounding box by stripping the emulator's
    black border (colour #1C2127, BGR ≈ (39,33,28)).

    A row/column is considered border if the mean of ALL three channels stays
    within the border-colour tolerance (±20 per channel).
    Returns (x, y, w, h).  Falls back to full frame on failure.
    """
    h, w = screen.shape[:2]

    # Border colour in BGR
    BORDER_BGR = np.array([39, 33, 28], dtype=np.float32)
    TOL        = 20          # per-channel tolerance

    # Per-row and per-column mean colour (shape h×3 and w×3)
    row_mean = screen.mean(axis=1)   # (h, 3)
    col_mean = screen.mean(axis=0)   # (w, 3)

    def is_border(means):
        """True where the mean colour is within TOL of the border colour."""
        return np.all(np.abs(means - BORDER_BGR) < TOL, axis=1)

    row_border = is_border(row_mean)   # (h,) bool
    col_border = is_border(col_mean)   # (w,) bool

    content_rows = np.where(~row_border)[0]
    content_cols = np.where(~col_border)[0]

    if len(content_rows) == 0 or len(content_cols) == 0:
        return 0, 0, w, h

    y1, y2 = int(content_rows[0]),  int(content_rows[-1]) + 1
    x1, x2 = int(content_cols[0]),  int(content_cols[-1]) + 1
    cw, ch  = x2 - x1, y2 - y1

    if cw < w * min_content_ratio or ch < h * min_content_ratio:
        return 0, 0, w, h

    print(f"  [GameRect] cropped  x={x1}:{x2}  y={y1}:{y2}  ({cw}x{ch})  full=({w}x{h})")
    return x1, y1, cw, ch


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
    bmpstr  = saveBitMap.GetBitmapBits(True)
    im = Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                          bmpstr, 'raw', 'BGRX', 0, 1)
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    full = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    # Auto-crop emulator border (#1C2127)
    x, y, cw, ch = get_game_rect(full)
    return full[y:y+ch, x:x+cw]


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
