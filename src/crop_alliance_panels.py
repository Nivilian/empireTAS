"""
crop_alliance_panels.py
=======================
交互式批量裁切脚本：
  1. 打开第一张图，用鼠标框选目标区域（右侧信息面板等）
  2. 将该区域应用到文件夹内所有图片
  3. 保存到 src/images/alliance/

操作说明：
  - 鼠标左键拖拽框选区域
  - 按 SPACE 或 ENTER 确认
  - 按 C 取消重选
  - 框选完成后自动批量处理所有图片
"""

import os
import sys
import cv2
import glob

# ── 路径配置 ─────────────────────────────────────────────────────────────────
_SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR   = os.path.join(_SRC_DIR, "images", "extra",
                            "Screenshot_2026-03-03-23-58-44-929_com.truek等34项文件")
OUTPUT_DIR  = os.path.join(_SRC_DIR, "images", "alliance")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 收集所有图片 ──────────────────────────────────────────────────────────────
exts   = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
images = []
for ext in exts:
    images += glob.glob(os.path.join(INPUT_DIR, ext))
images = sorted(images)

if not images:
    print("[ERROR] 未在以下路径找到图片：")
    print(f"  {INPUT_DIR}")
    sys.exit(1)

print(f"[INFO] 共找到 {len(images)} 张图片")

# ── 读取第一张图用于框选 ──────────────────────────────────────────────────────
first_img = cv2.imdecode(
    __import__('numpy').fromfile(images[0], dtype=__import__('numpy').uint8),
    cv2.IMREAD_COLOR
)
if first_img is None:
    print(f"[ERROR] 无法读取：{images[0]}")
    sys.exit(1)

h, w = first_img.shape[:2]
print(f"[INFO] 图片尺寸：{w} × {h}")
print("[INFO] 请在弹出窗口中框选目标区域，按 SPACE/ENTER 确认，按 C 重选")

# ── 交互式 ROI 选择 ───────────────────────────────────────────────────────────
# 显示前先缩放，避免窗口超出屏幕
MAX_DISP_W = 1280
scale_disp = min(1.0, MAX_DISP_W / w)
disp_img   = cv2.resize(first_img, (int(w * scale_disp), int(h * scale_disp)))

window_name = "框选目标区域 (SPACE/ENTER确认  C重选)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, disp_img.shape[1], disp_img.shape[0])

roi = cv2.selectROI(window_name, disp_img, showCrosshair=True, fromCenter=False)
cv2.destroyAllWindows()

x, y, rw, rh = roi
if rw == 0 or rh == 0:
    print("[ERROR] 未选择有效区域，退出")
    sys.exit(1)

# 反缩放回原始坐标
x  = int(round(x  / scale_disp))
y  = int(round(y  / scale_disp))
rw = int(round(rw / scale_disp))
rh = int(round(rh / scale_disp))

# 边界保护
x  = max(0, min(x,       w - 1))
y  = max(0, min(y,       h - 1))
rw = min(rw, w - x)
rh = min(rh, h - y)

print(f"[INFO] 选定区域：x={x} y={y} w={rw} h={rh}  (原始像素坐标)")

# ── 批量裁切 ─────────────────────────────────────────────────────────────────
import numpy as np

saved = 0
failed = 0

for fpath in images:
    img = cv2.imdecode(np.fromfile(fpath, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  [SKIP]  读取失败：{os.path.basename(fpath)}")
        failed += 1
        continue

    ih, iw = img.shape[:2]
    # 若某些图分辨率不同，按比例重映射坐标
    sx = x  * iw // w
    sy = y  * ih // h
    sw = rw * iw // w
    sh = rh * ih // h
    sw = min(sw, iw - sx)
    sh = min(sh, ih - sy)

    crop = img[sy:sy+sh, sx:sx+sw]
    if crop.size == 0:
        print(f"  [SKIP]  裁切结果为空：{os.path.basename(fpath)}")
        failed += 1
        continue

    # 保存，文件名保留原始名
    out_name = os.path.splitext(os.path.basename(fpath))[0] + "_crop.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    ret, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if ret:
        buf.tofile(out_path)
        print(f"  [OK]    {out_name}  ({sw}×{sh})")
        saved += 1
    else:
        print(f"  [FAIL]  编码失败：{os.path.basename(fpath)}")
        failed += 1

print(f"\n[DONE] 已保存 {saved} 张  失败 {failed} 张")
print(f"  输出目录：{OUTPUT_DIR}")
