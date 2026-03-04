# -*- coding: utf-8 -*-
"""
地块标注工具  —  独立运行
==========================
用法：  python labeler.py

遍历 images/trainingBackup/ 下所有图片，
标注后将裁图保存到 images/labeledImages/<地块_占领>/

画框（按住键 + 鼠标左键已有框内任意位置）：
    1  粘土   2  森林   3  渔船   4  铜矿   5  石头

改占领状态（按住键 + 点击已有框内任意位置）：
    Q  →  散人占领
    E  →  联盟占领
    （不标记 = 空地）

其他：
    Z  撤销   C  清空   A/←  上一张   D/→  下一张
    S  保存裁图并跳下一张
"""

import os
import sys

import cv2
import numpy as np
import math
import json

from PyQt5.QtCore    import Qt, QPoint, QRect
from PyQt5.QtGui     import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea, QShortcut,
    QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

import sys, os
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from map_scanner import FieldScanner

# ── 路径 ─────────────────────────────────────────────────────────────────────
_LABEL_DIR  = os.path.dirname(os.path.abspath(__file__))   # src/label_train/
_SRC_DIR    = os.path.dirname(_LABEL_DIR)                  # src/

# 确保 src/ 和 src/label_train/ 都在 import 搜索路径中
for _p in (_SRC_DIR, _LABEL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BACKUP_DIR  = os.path.join(_SRC_DIR, "images", "trainingBackup")
LABELED_DIR = os.path.join(_SRC_DIR, "images", "labeledImages")
# Resource field model path
MODEL_PATH  = os.path.join(_SRC_DIR, "resourcefield_model.npz")
TORCH_MODEL_PATH = os.path.join(_SRC_DIR, "resourcefield_model.pth")

# Ensure labeled images directory structure exists: terrain / occupation
_TERRAINS_EN = ["clay", "forest", "boat", "copper", "stone"]
_OCCS_EN = ["free", "individual", "alliance"]
for t in _TERRAINS_EN:
    for o in _OCCS_EN:
        os.makedirs(os.path.join(LABELED_DIR, t, o), exist_ok=True)

# ── 地块种类（键 1-5） ────────────────────────────────────────────────────────
TERRAIN_KEYS = {
    Qt.Key_1: ("粘土",  QColor(255, 210,  40)),  # 黄
    Qt.Key_2: ("森林",  QColor( 60, 200,  60)),  # 绿
    Qt.Key_3: ("渔船",  QColor( 40, 200, 200)),  # 青
    Qt.Key_4: ("铜矿",  QColor(220, 130,  40)),  # 橙
    Qt.Key_5: ("石头",  QColor(160, 160, 160)),  # 灰
}
KEY_LABEL = {Qt.Key_1: "1", Qt.Key_2: "2", Qt.Key_3: "3",
             Qt.Key_4: "4", Qt.Key_5: "5"}

# ── 占领颜色（用于标签文字） ─────────────────────────────────────────────────
OCC_TEXT_COLOR = {
    "空":  QColor(255, 255, 255),
    "散人": QColor(255, 160,   0),
    "联盟": QColor( 80, 160, 255),
}


# ─────────────────────────────────────────────────────────────────────────────
#  标注画布
# ─────────────────────────────────────────────────────────────────────────────
class LabelCanvas(QWidget):
    """
    Annotation格式（mutable list，便于Q/E原地修改占领状态）：
        [x1, y1, x2, y2, [terrain_name, occupation], QColor]
        occupation ∈ {"free", "individual", "alliance"}
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.orig_img   = None
        self.pixmap     = None
        self.scale      = 1.0
        self.offset_x   = 0
        self.offset_y   = 0

        self.annotations = []   # list of [x1,y1,x2,y2,[terrain,occ],color]

        self.drawing        = False
        self.start_pt       = None
        self.cur_pt         = None
        self.active_terrain = None   # current Qt.Key_1..5
        self.occ_mode       = None   # "散人" | "联盟" | None  (Q / E held)
        self.clear_mode     = False  # Space held -> clear clicked grid label

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(640, 400)

        # Terrain color map (supports English and Chinese names)
        self.TERRAIN_COLOR_MAP = {
            "clay": QColor(150, 100, 40), "粘土": QColor(150, 100, 40),
            "forest": QColor(60, 200, 60),  "森林": QColor(60, 200, 60),
            "boat": QColor(40, 120, 220),   "渔船": QColor(40, 120, 220),
            "copper": QColor(220, 60, 60),  "铜矿": QColor(220, 60, 60),
            "stone": QColor(200, 200, 200), "石头": QColor(200, 200, 200),
        }

    # ── 图像 ──────────────────────────────────────────────────────────────────
    def load_image(self, bgr_img):
        self.orig_img    = bgr_img.copy()
        self.annotations = []
        self.grid_cells = []  # 新增：保存所有格子的像素范围
        self._detect_grid()
        self._rebuild_pixmap()
        self.update()

    def _detect_grid(self):
        # 自动检测grid，保存所有格子的像素范围
        try:
            # Quick fix: prefer the known default tile size to avoid
            # incorrect tiny-grid detections on some resolutions.
            scanner = FieldScanner(grid_fw=FieldScanner.DEFAULT_FW, grid_fh=FieldScanner.DEFAULT_FH)
            # 只用detect_yellow_frame和build_grid，不做分类
            fw, fh, frame = scanner.detect_yellow_frame(self.orig_img)
            if frame is None:
                self.grid_cells = []
                return
            x, y, w, h = frame
            grid = scanner.build_grid(x, y, fw, fh, 0, 0, self.orig_img.shape[1], self.orig_img.shape[0])
            # compute ROI bounds: bottom area with aspect 966:1718
            img_h, img_w = self.orig_img.shape[:2]
            target_h = int(round(img_w * 966.0 / 1718.0))
            top_bound = max(0, img_h - target_h)
            cells = []
            for (tx, ty, gfw, gfh) in grid:
                allowed = (ty >= top_bound and (ty + gfh) <= img_h and tx >= 0 and (tx + gfw) <= img_w)
                cells.append((tx, ty, gfw, gfh, allowed))
            self.grid_cells = cells
        except Exception as e:
            print(f"[LabelCanvas] grid detect failed: {e}")
            self.grid_cells = []

    def _rebuild_pixmap(self):
        if self.orig_img is None:
            return
        h, w = self.orig_img.shape[:2]
        rgb  = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        cw, ch = max(self.width(), 10), max(self.height(), 10)
        scaled = pix.scaled(cw, ch, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.scale    = scaled.width() / w
        self.offset_x = (cw - scaled.width())  // 2
        self.offset_y = (ch - scaled.height()) // 2
        self.pixmap   = scaled

    def _to_orig(self, qpt):
        x = int((qpt.x() - self.offset_x) / self.scale)
        y = int((qpt.y() - self.offset_y) / self.scale)
        return x, y

    def _to_widget(self, ox, oy):
        return int(ox * self.scale) + self.offset_x, \
               int(oy * self.scale) + self.offset_y

    def _point_in_diamond(self, x, y, tx, ty, fw, fh):
        """Return True if point (x,y) in diamond defined by bbox (tx,ty,fw,fh).
        Diamond center = (tx+fw/2, ty+fh/2); half-widths = fw/2, fh/2.
        Use L1 (Manhattan) diamond test: |dx/hx| + |dy/hy| <= 1
        """
        cx = tx + fw / 2.0
        cy = ty + fh / 2.0
        hx = fw / 2.0 if fw != 0 else 1.0
        hy = fh / 2.0 if fh != 0 else 1.0
        val = abs((x - cx) / hx) + abs((y - cy) / hy)
        return val <= 1.0

    # ── 键鼠事件 ─────────────────────────────────────────────────────────────
    def keyPressEvent(self, e):
        # 记录当前地块类型和占领状态
        if e.key() in KEY_LABEL:
            key = e.key()
            terrain_map = {Qt.Key_1: "clay", Qt.Key_2: "forest", Qt.Key_3: "boat", Qt.Key_4: "copper", Qt.Key_5: "stone"}
            self.active_terrain = terrain_map.get(key, None)
        elif e.key() == Qt.Key_N:
            # N = mark negative tile
            self.active_terrain = "negative"
        elif e.key() == Qt.Key_Space:
            # Space = enable clear-on-click mode
            self.clear_mode = True
        elif e.key() == Qt.Key_Q:
            self.occ_mode = "individual"
        elif e.key() == Qt.Key_E:
            self.occ_mode = "alliance"
        elif e.key() == Qt.Key_W:
            # W = mark free (empty) when held and clicking a grid
            self.occ_mode = "free"
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() in KEY_LABEL:
            self.active_terrain = None
        elif e.key() == Qt.Key_N:
            self.active_terrain = None
        elif e.key() == Qt.Key_Space:
            self.clear_mode = False
        elif e.key() in (Qt.Key_Q, Qt.Key_E, Qt.Key_W):
            self.occ_mode = None
        else:
            super().keyReleaseEvent(e)

    def mousePressEvent(self, e):
        if not self.grid_cells:
            return super().mousePressEvent(e)
        if e.button() == Qt.LeftButton:
            x, y = self._to_orig(e.pos())
            # 找到点击点所在的格子 (use diamond boundary)
            for idx, cell in enumerate(self.grid_cells):
                tx, ty, fw, fh, allowed = cell
                if not allowed:
                    continue
                if self._point_in_diamond(x, y, tx, ty, fw, fh):
                    # 检查是否已有标注（提前计算，供后续分支使用）
                    ann_idx = None
                    for i, ann in enumerate(self.annotations):
                        gx, gy = ann[0], ann[1]
                        if gx == tx and gy == ty:
                            ann_idx = i
                            break
                    # If space-clear mode is active, remove existing annotation on this grid
                    if getattr(self, 'clear_mode', False):
                        if ann_idx is not None:
                            self.annotations.pop(ann_idx)
                            self.update()
                        return
                    # If Shift is held during click, mark as negative/free (or with occ_mode)
                    if int(e.modifiers()) & int(Qt.ShiftModifier):
                        occ = self.occ_mode if self.occ_mode else "free"
                        color = QColor(180, 180, 180)
                        if ann_idx is not None:
                            # update existing annotation to negative
                            self.annotations[ann_idx][4] = ["negative", occ]
                            if len(self.annotations[ann_idx]) >= 6:
                                self.annotations[ann_idx][5] = color
                            else:
                                self.annotations[ann_idx].append(color)
                        else:
                            self.annotations.append([tx, ty, tx+fw, ty+fh, ["negative", occ], color, True])
                        self.update()
                        return
                    # 判断当前按键
                    terrain = None
                    occ = "free"
                    if self.active_terrain:
                        # 数字键按下，直接用格子精确标注（以菱形外接矩形保存）
                        terrain = self.active_terrain
                        if ann_idx is not None:
                            # update existing: update label and color
                            color = self.TERRAIN_COLOR_MAP.get(terrain, QColor(255, 210, 40))
                            self.annotations[ann_idx][4] = [terrain, occ]
                            # ensure color slot exists
                            if len(self.annotations[ann_idx]) >= 6:
                                self.annotations[ann_idx][5] = color
                            else:
                                # append color if missing
                                self.annotations[ann_idx].append(color)
                        else:
                            color = self.TERRAIN_COLOR_MAP.get(terrain, QColor(255, 210, 40))
                            # store [x1,y1,x2,y2, [terrain,occ], color, is_grid]
                            self.annotations.append([tx, ty, tx+fw, ty+fh, [terrain, occ], color, True])
                        self.update()
                        return
                    if self.occ_mode:
                        # Q/E按下，切换占领状态
                        if ann_idx is not None:
                            self.annotations[ann_idx][4][1] = self.occ_mode
                            self.update()
                        else:
                            # create a negative annotation with the chosen occupation
                            terrain = "negative"
                            occ = self.occ_mode
                            color = QColor(180, 180, 180)
                            self.annotations.append([tx, ty, tx+fw, ty+fh, [terrain, occ], color, True])
                            self.update()
                        return
        return super().mousePressEvent(e)

    # ── 绘制 ─────────────────────────────────────────────────────────────────
    def paintEvent(self, _e):
        p = QPainter(self)
        if self.pixmap:
            # Draw the pixmap as background
            p.drawPixmap(self.offset_x, self.offset_y, self.pixmap)
            # Draw the pixmap as background. Do NOT draw a synthetic grid on top
            # (this avoids double/ghosted grid lines when the game already renders them).
            p.drawPixmap(self.offset_x, self.offset_y, self.pixmap)
        # Draw computed grid overlay (UI-only). Do NOT modify self.orig_img.
        if getattr(self, 'grid_cells', None):
            for (tx, ty, gw, gh, allowed) in self.grid_cells:
                pen = QPen(QColor(0, 200, 200, 160) if allowed else QColor(120, 120, 120, 100), 1)
                p.setPen(pen)
                top = QPoint(*self._to_widget(tx + gw // 2, ty))
                right = QPoint(*self._to_widget(tx + gw, ty + gh // 2))
                bottom = QPoint(*self._to_widget(tx + gw // 2, ty + gh))
                left = QPoint(*self._to_widget(tx, ty + gh // 2))
                p.drawLine(top, right)
                p.drawLine(right, bottom)
                p.drawLine(bottom, left)
                p.drawLine(left, top)
        # 高亮已标注格子
        for ann in self.annotations:
            # Support both legacy [x1,y1,x2,y2,label,color] and new [x1,y1,x2,y2,label,color,is_grid]
            x1, y1, x2, y2 = ann[0], ann[1], ann[2], ann[3]
            label, color = ann[4], ann[5]
            is_grid = (len(ann) >= 7 and ann[6])
            terrain, occ = label
            if is_grid:
                # draw diamond polygon using grid bbox
                fw = x2 - x1
                fh = y2 - y1
                # diamond points in original image coords (no shift)
                pts_img = [
                    (x1 + fw // 2, y1),
                    (x2, y1 + fh // 2),
                    (x1 + fw // 2, y2),
                    (x1, y1 + fh // 2),
                ]
                # convert to widget coords
                qpts = [QPoint(*self._to_widget(px, py)) for (px, py) in pts_img]
                # fill for occupation state (individual -> white 0.2, alliance -> black 0.2)
                occ_fill = None
                occ_val = occ
                if occ_val in ("individual", "散人"):
                    occ_fill = QColor(255, 255, 255, int(0.4 * 255))
                elif occ_val in ("alliance", "联盟"):
                    occ_fill = QColor(0, 0, 0, int(0.4 * 255))
                if occ_fill:
                    p.setBrush(occ_fill)
                else:
                    p.setBrush(Qt.NoBrush)
                p.setPen(QPen(color, 4, Qt.SolidLine))
                p.drawPolygon(*qpts)
                p.setBrush(Qt.NoBrush)
                # label at diamond center
                cx_img = x1 + fw // 2
                cy_img = y1 + fh // 2
                wx, wy = self._to_widget(cx_img, cy_img)
                rect = QRect(wx - int(fw * self.scale / 2), wy - 10, int(fw * self.scale), 20)
                old_font = p.font()
                f = QFont()
                f.setPointSize(10)
                p.setFont(f)
                p.setPen(QPen(QColor(0,0,0), 3))
                p.drawText(rect, Qt.AlignCenter, f"[{terrain}, {occ}]")
                p.setPen(QPen(color, 1))
                p.drawText(rect, Qt.AlignCenter, f"[{terrain}, {occ}]")
                p.setFont(old_font)
            else:
                wx1, wy1 = self._to_widget(x1, y1)
                wx2, wy2 = self._to_widget(x2, y2)
                # fill rect for occupation state
                occ_fill = None
                occ_val = occ
                if occ_val in ("individual", "散人"):
                    occ_fill = QColor(255, 255, 255, int(0.4 * 255))
                elif occ_val in ("alliance", "联盟"):
                    occ_fill = QColor(0, 0, 0, int(0.4 * 255))
                if occ_fill:
                    p.setBrush(occ_fill)
                else:
                    p.setBrush(Qt.NoBrush)
                p.setPen(QPen(color, 4, Qt.SolidLine))
                p.drawRect(QRect(QPoint(wx1, wy1), QPoint(wx2, wy2)))
                p.setBrush(Qt.NoBrush)
                rect = QRect(QPoint(wx1, wy1), QPoint(wx2, wy2))
                old_font = p.font()
                f = QFont()
                f.setPointSize(10)
                p.setFont(f)
                p.setPen(QPen(QColor(0,0,0), 3))
                p.drawText(rect, Qt.AlignCenter, f"[{terrain}, {occ}]")
                p.setPen(QPen(color, 1))
                p.drawText(rect, Qt.AlignCenter, f"[{terrain}, {occ}]")
                p.setFont(old_font)

        # 正在画的矩形（虚线预览）
        if self.drawing and self.start_pt and self.cur_pt and self.active_terrain:
            _, color = TERRAIN_KEYS[self.active_terrain]
            p.setPen(QPen(color, 1, Qt.DashLine))
            p.drawRect(QRect(self.start_pt, self.cur_pt))

        p.end()

    def qpixmap_to_cvimg(self, pixmap):
        # Convert QPixmap to numpy BGR image
        qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        w, h = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(h * w * 3)
        arr = np.array(ptr, dtype=np.uint8).reshape((h, w, 3))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def cvimg_to_qpixmap(self, img):
        # Convert numpy BGR image to QPixmap
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    # ── 操作 ─────────────────────────────────────────────────────────────────
    def undo(self):
        if self.annotations:
            self.annotations.pop()
            self.update()

    def clear(self):
        self.annotations = []
        self.update()

    def save_crops(self, src_name="img"):
        """裁图 → LABELED_DIR/<terrain>/<occ>/，返回保存数。"""
        if self.orig_img is None or not self.annotations:
            return 0
        saved = 0
        base = os.path.splitext(src_name)[0]
        for i, ann in enumerate(self.annotations):
            # support legacy and new annotation formats
            x1, y1, x2, y2 = ann[0], ann[1], ann[2], ann[3]
            label = ann[4] if len(ann) > 4 else ["unknown", "free"]
            terrain, occ = label[0], label[1]
            # 兼容旧数据，映射中文到英文
            terrain_map = {"粘土": "clay", "森林": "forest", "渔船": "boat", "铜矿": "copper", "石头": "stone"}
            occ_map = {"空": "free", "散人": "individual", "联盟": "alliance"}
            terrain = terrain_map.get(terrain, terrain)
            occ = occ_map.get(occ, occ)
            # If terrain is negative or unrecognized, store under negative/<occ>
            if isinstance(terrain, str) and terrain.lower() in ("negative", "neg", "负", "负样本"):
                cls_dir = os.path.join(LABELED_DIR, "negative", occ)
            else:
                cls_dir = os.path.join(LABELED_DIR, terrain, occ)
            os.makedirs(cls_dir, exist_ok=True)
            # only save crops that are fully inside allowed ROI if grid-based
            allowed_to_save = True
            if len(ann) >= 7 and ann[6]:
                # check whether this bbox is inside an allowed grid cell
                # compare with self.grid_cells
                allowed_to_save = False
                for (tx, ty, gfw, gfh, allowed) in getattr(self, 'grid_cells', []):
                    if tx == x1 and ty == y1 and (tx + gfw) == x2 and (ty + gfh) == y2:
                        allowed_to_save = allowed
                        break
            if not allowed_to_save:
                continue
            crop = self.orig_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            fname = os.path.join(cls_dir, f"{base}_{i}.png")
            cv2.imencode(".png", crop)[1].tofile(fname)
            saved += 1
        return saved

    def export_calibration(self, src_name="img"):
        """Export grid-based annotation centers for external calibration.
        Writes JSON to images/labeledImages/calibration/<base>.json
        """
        # Calibration export removed — do not write per-image calibration files.
        return None

    def generate_negatives(self, src_name="img"):
        """Generate negative crops for grid cells in the current image that are not annotated.
        Saves to LABELED_DIR/negative/free/ and returns number saved.
        """
        if self.orig_img is None or not getattr(self, 'grid_cells', None):
            return 0
        base = os.path.splitext(src_name)[0]
        out_dir = os.path.join(LABELED_DIR, "negative", "free")
        os.makedirs(out_dir, exist_ok=True)
        saved = 0
        # build set of annotated grid origins for quick lookup
        ann_origins = set()
        for ann in self.annotations:
            if len(ann) >= 7 and ann[6]:
                ann_origins.add((ann[0], ann[1]))
        for i, cell in enumerate(self.grid_cells):
            tx, ty, fw, fh = cell[:4]
            if (tx, ty) in ann_origins:
                continue
            x1, y1 = int(tx), int(ty)
            x2, y2 = int(tx + fw), int(ty + fh)
            crop = self.orig_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            fname = os.path.join(out_dir, f"{base}_neg_{i}.png")
            cv2.imencode('.png', crop)[1].tofile(fname)
            saved += 1
        return saved


# ─────────────────────────────────────────────────────────────────────────────
#  对比窗口（标注 vs 模型识别）
# ─────────────────────────────────────────────────────────────────────────────
class CompareWindow(QDialog):
    """左：真实标注；右：模型预测（可在未标注图上展示）。"""
    def __init__(self, orig_img, gt_annotations=None, predict_boxes=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("标注 vs 模型识别  对比")
        self.resize(1300, 650)

        if gt_annotations is None:
            gt_annotations = []
        self.gt_annotations = gt_annotations
        self.predict_boxes = predict_boxes

        gt_img = self._draw_gt(orig_img, self.gt_annotations)
        pred_img = self._draw_pred(orig_img, self.predict_boxes if self.predict_boxes is not None else self.gt_annotations)

        combined = self._hstack(gt_img, pred_img)
        lbl = QLabel()
        lbl.setPixmap(self._to_pixmap(combined))
        lbl.setAlignment(Qt.AlignCenter)

        scroll = QScrollArea()
        scroll.setWidget(lbl)
        scroll.setWidgetResizable(True)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("  📌  真实标注"))
        hdr.addStretch()
        hdr.addWidget(QLabel("🤖  模型预测  "))

        layout = QVBoxLayout()
        layout.addLayout(hdr)
        layout.addWidget(scroll)
        self.setLayout(layout)

    @staticmethod
    def _draw_gt(orig, annotations):
        img = orig.copy()
        for ann in annotations:
            x1, y1, x2, y2 = ann[0], ann[1], ann[2], ann[3]
            label = ann[4] if len(ann) > 4 else ["unknown", "free"]
            qcolor = ann[5] if len(ann) > 5 and isinstance(ann[5], QColor) else QColor(255, 255, 255)
            terrain, occ = label[0], label[1]
            bgr = (qcolor.blue(), qcolor.green(), qcolor.red())
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
            label_str = f"[{terrain}, {occ}]"
            cv2.putText(img, label_str, (x1 + 2, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(img, label_str, (x1 + 2, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return img

    @staticmethod
    def _draw_pred(orig, boxes):
        img = orig.copy()
        clf = None
        # Prefer a PyTorch model if present
        try:
            sys.path.insert(0, _SRC_DIR)
            try:
                from torch_classifier import TorchResourceFieldClassifier
                torch_clf = TorchResourceFieldClassifier()
                if os.path.exists(TORCH_MODEL_PATH) and torch_clf.load(TORCH_MODEL_PATH):
                    clf = torch_clf
            except Exception:
                pass
            # fallback to lightweight centroid classifier
            if clf is None and os.path.exists(MODEL_PATH):
                from resourcefield_classifier import ResourceFieldClassifier
                clf = ResourceFieldClassifier()
                if not clf.load(MODEL_PATH):
                    clf = None
        except Exception:
            clf = None

        # Normalize boxes and run prediction on diamond-shaped mask inside each box.
        box_list = []
        for b in boxes or []:
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                box_list.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))

        # terrain -> BGR color map for drawing
        terrain_to_bgr = {
            'clay':   (40, 100, 150),
            'forest': (60, 200, 60),
            'boat':   (220, 120, 40),
            'copper': (60, 60, 220),
            'stone':  (200, 200, 200),
        }

        for (x1, y1, x2, y2) in box_list:
            gw = x2 - x1
            gh = y2 - y1
            # prepare diamond polygon (original-image coords)
            pts = np.array([
                (x1 + gw // 2, y1),
                (x2, y1 + gh // 2),
                (x1 + gw // 2, y2),
                (x1, y1 + gh // 2),
            ], dtype=np.int32)

            pred_label = None
            pred_color = (128, 128, 128)
            pred_conf = 0.0
            if clf is not None:
                crop = orig[y1:y2, x1:x2].copy()
                if crop.size > 0:
                    ch, cw = crop.shape[:2]
                    # require a minimal diamond size
                    if cw > 8 and ch > 8:
                        # create diamond mask based on the actual crop size to avoid shape mismatch
                        mask = np.zeros((ch, cw), dtype=np.uint8)
                        poly = np.array([[
                            (cw // 2, 0), (cw - 1, ch // 2), (cw // 2, ch - 1), (0, ch // 2)
                        ]], dtype=np.int32)
                        cv2.fillPoly(mask, poly, 255)
                        # apply mask: set background to mean color to avoid confusing classifier
                        bg = crop.mean(axis=(0, 1)).astype(np.uint8)
                        bg_img = np.zeros_like(crop)
                        bg_img[:, :] = bg
                        masked = np.where(mask[:, :, None] == 255, crop, bg_img)
                        name, conf = clf.predict(masked)
                    else:
                        name, conf = (None, 0.0)
                    pred_label = name
                    pred_conf = conf
                    # classifier may return combined class names like "clay_individual"
                    # split into terrain and occupation when present
                    terrain_name = None
                    occ_name = "free"
                    if isinstance(name, str) and name:
                        parts = name.split("_")
                        terrain_name = parts[0]
                        if len(parts) > 1:
                            occ_name = parts[1]
                    else:
                        terrain_name = None
                    pred_color = terrain_to_bgr.get(terrain_name, pred_color)

            # Only draw when classifier predicts a known terrain with modest confidence
            if pred_label is not None and terrain_name in terrain_to_bgr and pred_conf >= 0.15:
                # translucent fill
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color=pred_color)
                alpha = 0.28
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                # outline
                cv2.polylines(img, [pts], isClosed=True, color=pred_color, thickness=2)
                # label near center
                cx = x1 + gw // 2
                cy = y1 + gh // 2
                # draw occupation overlay similar to ground-truth visualization
                if occ_name in ("individual", "散人"):
                    # white translucent fill
                    occ_overlay = img.copy()
                    cv2.fillPoly(occ_overlay, [pts], color=(255, 255, 255))
                    cv2.addWeighted(occ_overlay, 0.18, img, 1 - 0.18, 0, img)
                elif occ_name in ("alliance", "联盟"):
                    occ_overlay = img.copy()
                    cv2.fillPoly(occ_overlay, [pts], color=(0, 0, 0))
                    cv2.addWeighted(occ_overlay, 0.18, img, 1 - 0.18, 0, img)

                label_str = f"{terrain_name}, {occ_name} {int(pred_conf*100)}%"
                cv2.putText(img, label_str, (cx - gw//4, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
                cv2.putText(img, label_str, (cx - gw//4, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pred_color, 1)
            # else: do not draw anything for negative / low-confidence boxes
        return img

    @staticmethod
    def _hstack(a, b):
        h = max(a.shape[0], b.shape[0])
        def pad(im):
            if im.shape[0] < h:
                pad_rows = np.zeros((h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
                im = np.vstack([im, pad_rows])
            return im
        sep = np.full((h, 6, 3), 60, dtype=np.uint8)
        return np.hstack([pad(a), sep, pad(b)])

    @staticmethod
    def _to_pixmap(bgr_img):
        h, w = bgr_img.shape[:2]
        rgb  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)


class PredictWindow(QDialog):
    """单张图片展示：仅显示模型在整张图上的识别结果（不显示真实标注）。"""
    def __init__(self, orig_img, predict_boxes=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型识别结果")
        self.resize(1100, 700)
        # Use CompareWindow._draw_pred to reuse prediction drawing logic
        pred_img = CompareWindow._draw_pred(orig_img, predict_boxes)
        lbl = QLabel()
        lbl.setPixmap(self._to_pixmap(pred_img))
        lbl.setAlignment(Qt.AlignCenter)

        scroll = QScrollArea()
        scroll.setWidget(lbl)
        scroll.setWidgetResizable(True)

        layout = QVBoxLayout()
        layout.addWidget(scroll)
        self.setLayout(layout)

    @staticmethod
    def _to_pixmap(bgr_img):
        h, w = bgr_img.shape[:2]
        rgb  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)


# ─────────────────────────────────────────────────────────────────────────────
#  辅助：修复 SpinBox 数字乱码
# ─────────────────────────────────────────────────────────────────────────────
def _apply_spinbox_font(widget):
    """在中文 Windows 上，QSpinBox 内部 QLineEdit 不继承 app 字体，
    必须直接 setFont 才能让数字用 Segoe UI 字形（而非 CJK 数字字形）。"""
    from PyQt5.QtGui import QFont
    _f = QFont("Segoe UI", 9)
    for w in widget.findChildren((QSpinBox, QDoubleSpinBox)):
        w.setFont(_f)
        try:
            if w.lineEdit():
                w.lineEdit().setFont(_f)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  训练参数设置对话框
# ─────────────────────────────────────────────────────────────────────────────
class TrainSettingsDialog(QDialog):
    """弹出对话框，让用户调整所有训练超参数再开始训练。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("训练参数设置")
        self.setMinimumWidth(420)

        main_layout = QVBoxLayout(self)

        # ── 训练类型 ─────────────────────────────────────────────────────────
        grp_mode = QGroupBox("训练类型")
        form_mode = QFormLayout(grp_mode)
        self.cb_train_mode = QComboBox()
        self.cb_train_mode.addItems(["固定参数", "随机搜索 (Random Search)", "网格搜索 (Grid Search)"])
        lbl_mode_hint = QLabel("随机/网格搜索: 自动尝试多组超参，保留测试集最优模型")
        lbl_mode_hint.setStyleSheet("color:#888; font-size:10px;")
        form_mode.addRow("训练类型", self.cb_train_mode)
        form_mode.addRow(lbl_mode_hint)

        # ── 基础参数 ─────────────────────────────────────────────────────────
        grp_basic = QGroupBox("基础参数")
        form_basic = QFormLayout(grp_basic)

        self.sp_epochs = QSpinBox(); self.sp_epochs.setRange(1, 500); self.sp_epochs.setValue(30)
        self.sp_batch  = QSpinBox(); self.sp_batch.setRange(1, 256);  self.sp_batch.setValue(32)
        self.sp_lr     = QDoubleSpinBox()
        self.sp_lr.setRange(1e-6, 0.5); self.sp_lr.setSingleStep(0.0001)
        self.sp_lr.setDecimals(6); self.sp_lr.setValue(0.001)

        self.cb_model = QComboBox()
        self.cb_model.addItems(["resnet18", "mobilenet_v2", "custom"])

        self.cb_scheduler = QComboBox()
        self.cb_scheduler.addItems(["cosine", "plateau", "none"])
        self.cb_scheduler.setCurrentText("cosine")
        lbl_sched_hint = QLabel("cosine: 自动衰减LR(推荐)   plateau: 需开启每epoch评估")
        lbl_sched_hint.setStyleSheet("color:#888; font-size:10px;")

        self.chk_pretrained  = QCheckBox("使用预训练权重")
        self.chk_pretrained.setChecked(False)
        self.chk_class_w     = QCheckBox("使用类别权重平衡")
        self.chk_class_w.setChecked(True)

        form_basic.addRow("训练轮数 (epochs)",  self.sp_epochs)
        form_basic.addRow("批大小 (batch_size)", self.sp_batch)
        form_basic.addRow("学习率 (lr)",         self.sp_lr)
        form_basic.addRow("模型架构",            self.cb_model)
        form_basic.addRow("LR Scheduler",        self.cb_scheduler)
        form_basic.addRow(lbl_sched_hint)
        form_basic.addRow(self.chk_pretrained)
        form_basic.addRow(self.chk_class_w)

        # ── 参数搜索范围 ─────────────────────────────────────────────────────
        self.grp_search = QGroupBox("搜索参数范围 (逗号分隔候选值)")
        form_search = QFormLayout(self.grp_search)

        self.le_search_epochs = QLineEdit("30, 60")
        self.le_search_batch  = QLineEdit("16, 32")
        self.le_search_lr     = QLineEdit("0.001, 0.0003, 0.0001")
        self.le_search_k      = QLineEdit("3, 5")
        self.sp_num_trials    = QSpinBox()
        self.sp_num_trials.setRange(1, 200); self.sp_num_trials.setValue(6)
        self._lbl_trials      = QLabel("最大次数 (随机搜索)")
        lbl_k_hint = QLabel("每次切换 K 前会将现有 cluster_* 文件还原再重新聚类")
        lbl_k_hint.setStyleSheet("color:#888; font-size:10px;")

        form_search.addRow("Epochs 候选",         self.le_search_epochs)
        form_search.addRow("Batch 候选",          self.le_search_batch)
        form_search.addRow("LR 候选",             self.le_search_lr)
        form_search.addRow("Negative K 候选",     self.le_search_k)
        form_search.addRow(lbl_k_hint)
        form_search.addRow(self._lbl_trials, self.sp_num_trials)
        lbl_sh = QLabel("网格搜索: 尝试所有组合   随机搜索: 随机采样至多N次")
        lbl_sh.setStyleSheet("color:#888; font-size:10px;")
        form_search.addRow(lbl_sh)

        self.grp_search.setVisible(False)

        # ── 数据划分 ─────────────────────────────────────────────────────────
        grp_split = QGroupBox("数据划分")
        form_split = QFormLayout(grp_split)

        self.sp_val  = QDoubleSpinBox(); self.sp_val.setRange(0.05, 0.45);  self.sp_val.setSingleStep(0.05);  self.sp_val.setDecimals(2); self.sp_val.setValue(0.20)
        self.sp_test = QDoubleSpinBox(); self.sp_test.setRange(0.05, 0.45); self.sp_test.setSingleStep(0.05); self.sp_test.setDecimals(2); self.sp_test.setValue(0.20)
        self.chk_val_during = QCheckBox("每 epoch 评估验证集")
        self.chk_val_during.setChecked(False)

        form_split.addRow("验证集比例 (val_split)",  self.sp_val)
        form_split.addRow("测试集比例 (test_split)", self.sp_test)
        form_split.addRow(self.chk_val_during)

        # ── 高级 / 性能 ──────────────────────────────────────────────────────
        grp_adv = QGroupBox("高级 / 性能")
        form_adv = QFormLayout(grp_adv)

        self.sp_workers   = QSpinBox(); self.sp_workers.setRange(0, 16); self.sp_workers.setValue(4)
        self.chk_pin_mem  = QCheckBox("pin_memory (GPU 加速)")
        self.chk_pin_mem.setChecked(True)
        self.chk_persist  = QCheckBox("persistent_workers")
        self.chk_persist.setChecked(True)   # default True on Windows: avoids per-epoch spawn stalls

        self.sp_num_conv   = QSpinBox(); self.sp_num_conv.setRange(1, 8); self.sp_num_conv.setValue(3)
        self.sp_base_ch    = QSpinBox(); self.sp_base_ch.setRange(8, 256); self.sp_base_ch.setValue(32)
        lbl_conv_hint = QLabel("仅 custom 模型有效")
        lbl_conv_hint.setStyleSheet("color:#888; font-size:10px;")

        form_adv.addRow("DataLoader workers", self.sp_workers)
        form_adv.addRow(self.chk_pin_mem)
        form_adv.addRow(self.chk_persist)
        form_adv.addRow("Custom 卷积层数",   self.sp_num_conv)
        form_adv.addRow("Custom 基础通道数", self.sp_base_ch)
        form_adv.addRow(lbl_conv_hint)

        # 仅在选 custom 模型时启用卷积层数/通道数控件
        def _on_model_changed(text):
            is_custom = (text == "custom")
            self.sp_num_conv.setEnabled(is_custom)
            self.sp_base_ch.setEnabled(is_custom)
            lbl_conv_hint.setEnabled(is_custom)
        self.cb_model.currentTextChanged.connect(_on_model_changed)
        _on_model_changed(self.cb_model.currentText())  # 初始化状态

        # 训练模式切换：搜索模式时显示搜索范围组，置灰基础参数中的 lr/epochs/batch
        def _on_mode_changed(text):
            is_search = "搜索" in text
            self.grp_search.setVisible(is_search)
            is_random = "随机" in text
            self._lbl_trials.setVisible(is_random)
            self.sp_num_trials.setVisible(is_random)
            for w in (self.sp_epochs, self.sp_batch, self.sp_lr):
                w.setEnabled(not is_search)
            self.adjustSize()
        self.cb_train_mode.currentTextChanged.connect(_on_mode_changed)
        _on_mode_changed(self.cb_train_mode.currentText())

        # ── 按钮 ─────────────────────────────────────────────────────────────
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        main_layout.addWidget(grp_mode)
        main_layout.addWidget(grp_basic)
        main_layout.addWidget(self.grp_search)
        main_layout.addWidget(grp_split)
        main_layout.addWidget(grp_adv)
        main_layout.addWidget(buttons)
        _apply_spinbox_font(self)

    @staticmethod
    def _parse_values(text, cast):
        """Parse comma-separated candidate values, e.g. '0.001, 0.0003' -> [0.001, 0.0003]."""
        result = []
        for part in text.split(","):
            part = part.strip()
            if part:
                try:
                    result.append(cast(part))
                except ValueError:
                    pass
        return result if result else [cast("1")]

    def get_params(self) -> dict:
        mode_txt = self.cb_train_mode.currentText()
        if "随机" in mode_txt:
            train_mode = "random_search"
        elif "网格" in mode_txt:
            train_mode = "grid_search"
        else:
            train_mode = "fixed"
        return dict(
            train_mode           = train_mode,
            epochs               = self.sp_epochs.value(),
            batch_size           = self.sp_batch.value(),
            lr                   = self.sp_lr.value(),
            model_type           = self.cb_model.currentText(),
            lr_scheduler         = self.cb_scheduler.currentText(),
            pretrained           = self.chk_pretrained.isChecked(),
            use_class_weights    = self.chk_class_w.isChecked(),
            val_split            = self.sp_val.value(),
            test_split           = self.sp_test.value(),
            validate_during_training = self.chk_val_during.isChecked(),
            num_workers          = self.sp_workers.value(),
            pin_memory           = self.chk_pin_mem.isChecked(),
            persistent_workers   = self.chk_persist.isChecked(),
            num_conv             = self.sp_num_conv.value(),
            base_channels        = self.sp_base_ch.value(),
            search_epochs        = self._parse_values(self.le_search_epochs.text(), int),
            search_batch         = self._parse_values(self.le_search_batch.text(), int),
            search_lr            = self._parse_values(self.le_search_lr.text(), float),
            search_neg_k         = self._parse_values(self.le_search_k.text(), int),
            num_trials           = self.sp_num_trials.value(),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  聚类参数设置对话框
# ─────────────────────────────────────────────────────────────────────────────
class ClusterSettingsDialog(QDialog):
    """弹出对话框，让用户调整负样本聚类参数再执行聚类。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("聚类参数设置")
        self.setMinimumWidth(340)

        layout = QVBoxLayout(self)

        grp_main = QGroupBox("聚类参数")
        form = QFormLayout(grp_main)

        self.sp_k      = QSpinBox(); self.sp_k.setRange(2, 30); self.sp_k.setValue(5)
        self.le_prefix = QLineEdit("cluster")

        form.addRow("分组数 k",       self.sp_k)
        form.addRow("子目录前缀",     self.le_prefix)

        grp_feat = QGroupBox("特征提取（HSV 直方图）")
        form_feat = QFormLayout(grp_feat)

        self.sp_resize = QSpinBox(); self.sp_resize.setRange(8, 128); self.sp_resize.setValue(32)
        self.sp_h_bins = QSpinBox(); self.sp_h_bins.setRange(2,  36); self.sp_h_bins.setValue(18)
        self.sp_s_bins = QSpinBox(); self.sp_s_bins.setRange(2,  16); self.sp_s_bins.setValue(8)
        self.sp_v_bins = QSpinBox(); self.sp_v_bins.setRange(2,  16); self.sp_v_bins.setValue(8)

        form_feat.addRow("缩放尺寸 (px)",  self.sp_resize)
        form_feat.addRow("H 直方图桶数",   self.sp_h_bins)
        form_feat.addRow("S 直方图桶数",   self.sp_s_bins)
        form_feat.addRow("V 直方图桶数",   self.sp_v_bins)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(grp_main)
        layout.addWidget(grp_feat)
        layout.addWidget(buttons)
        _apply_spinbox_font(self)

    def get_params(self) -> dict:
        return dict(
            k       = self.sp_k.value(),
            prefix  = self.le_prefix.text().strip() or "cluster",
            resize  = self.sp_resize.value(),
            h_bins  = self.sp_h_bins.value(),
            s_bins  = self.sp_s_bins.value(),
            v_bins  = self.sp_v_bins.value(),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  误分类图片浏览对话框
# ─────────────────────────────────────────────────────────────────────────────
class MisclassifiedDialog(QDialog):
    """训练结束后显示测试集中预测错误的图片，辅助发现标注错误。"""
    _COLS   = 6
    _IMG_SZ = 88
    _TILE_W = 114
    _TILE_H = 162
    _CAP    = 300

    def __init__(self, misclassified, parent=None):
        super().__init__(parent)
        n = len(misclassified)
        self.setWindowTitle(f"误分类图片 ({n} 张) — 检查标注是否有误")
        self.resize(self._COLS * (self._TILE_W + 8) + 40, 640)

        container = QWidget()
        grid = QGridLayout(container)
        grid.setSpacing(5)
        grid.setContentsMargins(6, 6, 6, 6)

        for i, item in enumerate(misclassified[:self._CAP]):
            frame = QFrame()
            frame.setFixedSize(self._TILE_W, self._TILE_H)
            frame.setStyleSheet(
                "QFrame { border:1px solid #555; background:#1c1c1c; border-radius:4px; }"
            )
            vbox = QVBoxLayout(frame)
            vbox.setContentsMargins(3, 4, 3, 4)
            vbox.setSpacing(2)

            img_lbl = QLabel()
            img_lbl.setFixedSize(self._IMG_SZ, self._IMG_SZ)
            img_lbl.setAlignment(Qt.AlignCenter)
            try:
                pix = QPixmap(item['path'])
                if not pix.isNull():
                    img_lbl.setPixmap(
                        pix.scaled(self._IMG_SZ, self._IMG_SZ,
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    )
                else:
                    img_lbl.setText("?")
            except Exception:
                img_lbl.setText("!")
            vbox.addWidget(img_lbl, 0, Qt.AlignCenter)

            # T = 标注真实类别 (green)
            lbl_t = QLabel(f"T: {item['true']}")
            lbl_t.setStyleSheet("color:#44ee44; font-size:9px;")
            lbl_t.setAlignment(Qt.AlignCenter)
            lbl_t.setWordWrap(True)
            vbox.addWidget(lbl_t)

            # P = 模型预测类别 (red)
            lbl_p = QLabel(f"P: {item['pred']}  {item['conf']*100:.0f}%")
            lbl_p.setStyleSheet("color:#ff5555; font-size:9px;")
            lbl_p.setAlignment(Qt.AlignCenter)
            lbl_p.setWordWrap(True)
            vbox.addWidget(lbl_p)

            # 文件名 (灰色小字)
            fname = os.path.basename(item['path'])
            if len(fname) > 18:
                fname = fname[:15] + "..."
            lbl_name = QLabel(fname)
            lbl_name.setStyleSheet("color:#888888; font-size:8px;")
            lbl_name.setAlignment(Qt.AlignCenter)
            lbl_name.setWordWrap(True)
            vbox.addWidget(lbl_name)

            row, col = divmod(i, self._COLS)
            grid.addWidget(frame, row, col)

        scroll = QScrollArea()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border:none; }")

        trunc = f"  (已截断至 {self._CAP} 张)" if n > self._CAP else ""
        hint = QLabel(
            f"测试集共 {n} 张预测错误{trunc}\n"
            "绿色 T = 标注类别    红色 P = 模型预测类别    百分比 = 置信度\n"
            "若 T 和 P 相差大，请检查该图片标注是否正确。"
        )
        hint.setStyleSheet("color:#aaaaaa; font-size:10px; padding:6px 4px 2px 4px;")
        hint.setWordWrap(True)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)

        lay = QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addWidget(scroll, 1)
        lay.addWidget(close_btn)


# ─────────────────────────────────────────────────────────────────────────────
#  主窗口
# ─────────────────────────────────────────────────────────────────────────────
class LabelerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "地块标注工具  |  1-5=地块  Q=散人  E=联盟  |  A/D=翻页  S=保存  Z=撤销  C=清空"
        )
        self.resize(1400, 860)

        self.image_paths = []
        self.cur_idx     = -1
        self.labeled_set = set()

        # ── 左侧文件列表 ──────────────────────────────────────────────────────
        self.file_list = QListWidget()
        self.file_list.setFixedWidth(220)
        self.file_list.currentRowChanged.connect(self._on_list_select)

        # ── 画布 ──────────────────────────────────────────────────────────────
        self.canvas = LabelCanvas()

        # ── 按钮 ──────────────────────────────────────────────────────────────
        self.btn_prev    = QPushButton("← 上一张")
        self.btn_next    = QPushButton("下一张 →")
        self.btn_save    = QPushButton("💾 保存 (S)")
        self.btn_undo    = QPushButton("↩ 撤销 (Z)")
        self.btn_clear   = QPushButton("🗑 清空 (C)")
        self.btn_compare = QPushButton("🔍 对比")
        self.btn_train   = QPushButton("🚀 训练")
        self.lbl_idx     = QLabel("0 / 0")
        self.lbl_status  = QLabel("从左侧选择图片开始标注")

        # 生成负样本按钮与自动开关（手动/自动）
        self.btn_gen_neg = QPushButton("生成负样本")
        self.btn_auto_neg = QPushButton("自动生成负样本：关")
        self.btn_cluster_neg = QPushButton("🔀 聚类负样本")
        self.btn_gen_neg.setFixedHeight(28)
        self.btn_auto_neg.setFixedHeight(28)
        self.btn_cluster_neg.setFixedHeight(28)
        self.btn_gen_neg.clicked.connect(self._do_generate_negatives)
        self.btn_auto_neg.clicked.connect(self._toggle_auto_negatives)
        self.btn_cluster_neg.clicked.connect(self._do_cluster_negatives)
        self.auto_generate_negatives = False

        for btn in [self.btn_prev, self.btn_next, self.btn_save, self.btn_undo,
                self.btn_clear, self.btn_compare, self.btn_train,
                self.btn_gen_neg, self.btn_auto_neg, self.btn_cluster_neg]:
            btn.setFixedHeight(28)
        self.btn_save.setStyleSheet("background:#1a7a1a; color:white; font-weight:bold;")
        self.btn_train.setStyleSheet("background:#1a3a99; color:white; font-weight:bold;")
        self.btn_compare.setStyleSheet("background:#5a3a00; color:white; font-weight:bold;")
        self.lbl_status.setStyleSheet("color:#aaaaaa; padding:0 6px;")

        self.btn_prev.clicked.connect(self._go_prev)
        self.btn_next.clicked.connect(self._go_next)
        self.btn_save.clicked.connect(self._do_save)
        self.btn_undo.clicked.connect(self.canvas.undo)
        self.btn_clear.clicked.connect(self.canvas.clear)
        # Rename the button label to reflect single-image prediction display
        self.btn_compare.setText("🔎 显示识别结果")
        self.btn_compare.clicked.connect(self._do_compare)
        self.btn_train.clicked.connect(self._do_train)

        # ── QShortcut（不受焦点影响）────────────────────────────────────────
        QShortcut(QKeySequence("S"),     self).activated.connect(self._do_save)
        QShortcut(QKeySequence("A"),     self).activated.connect(self._go_prev)
        QShortcut(QKeySequence("D"),     self).activated.connect(self._go_next)
        QShortcut(QKeySequence("Left"),  self).activated.connect(self._go_prev)
        QShortcut(QKeySequence("Right"), self).activated.connect(self._go_next)
        QShortcut(QKeySequence("Z"),     self).activated.connect(self.canvas.undo)
        QShortcut(QKeySequence("C"),     self).activated.connect(self.canvas.clear)
        # X or Delete = remove current image from backup folder
        QShortcut(QKeySequence("X"),     self).activated.connect(self._do_delete_current)
        QShortcut(QKeySequence("Delete"),self).activated.connect(self._do_delete_current)

        # ── 图例 ─────────────────────────────────────────────────────────────
        legend = QHBoxLayout()
        for key, (name, color) in TERRAIN_KEYS.items():
            lbl = QLabel(f"[{KEY_LABEL[key]}]{name}")
            lbl.setStyleSheet(
                f"color:rgb({color.red()},{color.green()},{color.blue()});"
                f"font-weight:bold; padding:0 6px;"
            )
            legend.addWidget(lbl)
        legend.addSpacing(16)
        for occ, occ_color in [("散人", OCC_TEXT_COLOR["散人"]), ("联盟", OCC_TEXT_COLOR["联盟"])]:
            key_ch = "Q" if occ == "散人" else "E"
            lbl = QLabel(f"[{key_ch}]{occ}")
            lbl.setStyleSheet(
                f"color:rgb({occ_color.red()},{occ_color.green()},{occ_color.blue()});"
                f"font-weight:bold; padding:0 6px;"
            )
            legend.addWidget(lbl)
        legend.addStretch()

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.btn_prev)
        top_bar.addWidget(self.lbl_idx)
        top_bar.addWidget(self.btn_next)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.btn_save)
        top_bar.addWidget(self.btn_gen_neg)
        top_bar.addWidget(self.btn_auto_neg)
        top_bar.addWidget(self.btn_cluster_neg)
        top_bar.addWidget(self.btn_undo)
        top_bar.addWidget(self.btn_clear)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.btn_compare)
        top_bar.addWidget(self.btn_train)
        top_bar.addStretch()
        top_bar.addWidget(self.lbl_status)

        # ── 训练进度条 ────────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(16)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.addLayout(top_bar)
        right_layout.addWidget(self.progress_bar)
        right_layout.addLayout(legend)
        right_layout.addWidget(self.canvas, 1)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.file_list)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)
        self._load_file_list()

    # ── 文件列表 ──────────────────────────────────────────────────────────────
    def _load_file_list(self):
        self.file_list.clear()
        self.image_paths = []
        if not os.path.isdir(BACKUP_DIR):
            self.lbl_status.setText(f"目录不存在：{BACKUP_DIR}")
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        names = sorted(f for f in os.listdir(BACKUP_DIR)
                       if os.path.splitext(f)[1].lower() in exts)
        for name in names:
            self.image_paths.append(os.path.join(BACKUP_DIR, name))
            self.file_list.addItem(QListWidgetItem("   " + name))
        total = len(self.image_paths)
        self.lbl_idx.setText(f"0 / {total}")
        if total > 0:
            self.file_list.setCurrentRow(0)

    def _update_list_icon(self, idx):
        if idx < 0 or idx >= self.file_list.count():
            return
        name   = os.path.basename(self.image_paths[idx])
        prefix = "✅ " if idx in self.labeled_set else "   "
        self.file_list.item(idx).setText(prefix + name)

    # ── 导航 ─────────────────────────────────────────────────────────────────
    def _on_list_select(self, row):
        if row < 0 or row >= len(self.image_paths):
            return
        self.cur_idx = row
        path = self.image_paths[row]
        img  = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.lbl_status.setText(f"❌ 无法读取：{os.path.basename(path)}")
            return
        self.canvas.load_image(img)
        self.canvas.setFocus()
        total = len(self.image_paths)
        self.lbl_idx.setText(f"{row + 1} / {total}")
        status = "✅ 已保存" if row in self.labeled_set else "未标注"
        self.lbl_status.setText(f"{os.path.basename(path)}  [{status}]")

    def _go_prev(self):
        if self.image_paths:
            self.file_list.setCurrentRow((self.cur_idx - 1) % len(self.image_paths))

    def _go_next(self):
        if self.image_paths:
            self.file_list.setCurrentRow((self.cur_idx + 1) % len(self.image_paths))

    # ── 保存 ─────────────────────────────────────────────────────────────────
    def _do_save(self):
        if self.cur_idx < 0:
            return
        if not self.canvas.annotations:
            self.lbl_status.setText("⚠ 没有标注框，请先按住 1-5 拖拽画框")
            return
        src_name = os.path.basename(self.image_paths[self.cur_idx])
        n = self.canvas.save_crops(src_name)
        if n > 0:
            self.labeled_set.add(self.cur_idx)
            self._update_list_icon(self.cur_idx)
            calpath = self.canvas.export_calibration(src_name)
            msg = f"✅ 保存 {n} 张裁图 → labeledImages/"
            if calpath:
                msg += f"  校准点已导出：{os.path.basename(calpath)}"
            # If auto-generate-negatives is enabled, create negatives now
            if getattr(self, 'auto_generate_negatives', False):
                neg_n = self.canvas.generate_negatives(src_name)
                if neg_n:
                    msg += f"  生成负样本 {neg_n} 张"
            # 删除原始备份图片（用户要求：保存后删除）
            try:
                backup_path = self.image_paths[self.cur_idx]
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                    msg += "  已删除原始备份"
            except Exception as ex:
                msg += f"  删除备份失败: {ex}"
            self.lbl_status.setText(msg)
            # 刷新文件列表并移动到下一张
            # 清空并重新加载文件列表以反映已删除的文件
            self.labeled_set = set()
            self._load_file_list()
            # 选中原位置或下一项
            if self.image_paths:
                new_idx = min(self.cur_idx, len(self.image_paths) - 1)
                self.file_list.setCurrentRow(new_idx)
            else:
                self.cur_idx = -1

    def _do_generate_negatives(self):
        """Manual trigger: generate negative crops for current image."""
        if self.cur_idx < 0:
            return
        src_name = os.path.basename(self.image_paths[self.cur_idx])
        n = self.canvas.generate_negatives(src_name)
        if n > 0:
            self.lbl_status.setText(f"✅ 生成负样本 {n} 张 → images/labeledImages/negative/free/")
        else:
            self.lbl_status.setText("⚠ 未生成负样本（可能已全部标注或无可用网格）")

    def _do_cluster_negatives(self):
        """Auto-cluster all negatives into sub-directories via K-means on HSV histograms."""
        dlg = ClusterSettingsDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        p = dlg.get_params()
        self.lbl_status.setText(f"⏳ 正在聚类（k={p['k']}）...")
        QApplication.processEvents()
        try:
            from cluster_negatives import cluster_negatives
            report = cluster_negatives(
                data_dir=LABELED_DIR,
                k=p['k'],
                dry_run=False,
                prefix=p['prefix'],
                resize=p['resize'],
                hist_bins=(p['h_bins'], p['s_bins'], p['v_bins']),
            )
            counts = ", ".join(f"{name}:{len(files)}张" for name, files in report.items())
            self.lbl_status.setText(
                f"✅ 聚类完成({p['k']}组): {counts}  "
                f"— 可在 labeledImages/negative/ 中重命名子目录（如 city、empty）后重新训练"
            )
        except Exception as ex:
            self.lbl_status.setText(f"❌ 聚类失败: {ex}")

    def _do_delete_current(self):
        """Delete the current backup image immediately (no confirmation) and refresh list."""
        if self.cur_idx < 0 or self.cur_idx >= len(self.image_paths):
            return
        path = self.image_paths[self.cur_idx]
        name = os.path.basename(path)
        try:
            if os.path.exists(path):
                os.remove(path)
            self.lbl_status.setText(f"✅ 已删除：{name}")
        except Exception as ex:
            self.lbl_status.setText(f"❌ 删除失败：{ex}")
        # reload file list and select next
        self.labeled_set = set()
        self._load_file_list()
        if self.image_paths:
            new_idx = min(self.cur_idx, len(self.image_paths) - 1)
            self.file_list.setCurrentRow(new_idx)
        else:
            self.cur_idx = -1

    def _toggle_auto_negatives(self):
        """Toggle automatic negative generation after save."""
        self.auto_generate_negatives = not getattr(self, 'auto_generate_negatives', False)
        label = "自动生成负样本：开" if self.auto_generate_negatives else "自动生成负样本：关"
        self.btn_auto_neg.setText(label)
        self.lbl_status.setText("已启用自动生成负样本" if self.auto_generate_negatives else "已禁用自动生成负样本")

    # ── 对比 ─────────────────────────────────────────────────────────────────
    def _do_compare(self):
        if self.canvas.orig_img is None:
            self.lbl_status.setText("⚠ 请先选择图片")
            return
        img = self.canvas.orig_img
        # Always try to build a grid from the detected yellow anchor and use that
        # grid as the set of boxes for model prediction. This ensures the right
        # side shows model output rather than simply mirroring the annotations.
        try:
            # Use default locked grid size to avoid spurious small cells
            scanner = FieldScanner(grid_fw=FieldScanner.DEFAULT_FW, grid_fh=FieldScanner.DEFAULT_FH)
            fw, fh, frame = scanner.detect_yellow_frame(img)
            if frame is None:
                self.lbl_status.setText("⚠ 未检测到黄菱形锚点，无法生成网格进行模型预测")
                return
            ax, ay, _, _ = frame
            h, w = img.shape[:2]
            boxes = scanner.build_grid(ax, ay, fw, fh, 0, 0, w, h)
            predict_boxes = [(tx, ty, tx + gw, ty + gh) for (tx, ty, gw, gh) in boxes]
            dlg = PredictWindow(img, predict_boxes=predict_boxes, parent=self)
            dlg.exec_()
        except Exception as ex:
            self.lbl_status.setText(f"⚠ 对比时出错：{ex}")
            return

    # ── 训练 ─────────────────────────────────────────────────────────────────
    def _do_train(self):
        # Prefer PyTorch trainer when available; otherwise use lightweight baseline
        use_torch = False
        try:
            import torch  # type: ignore
            use_torch = True
        except Exception:
            use_torch = False
        # 统计样本
        total = 0
        if os.path.isdir(LABELED_DIR):
            for d in os.listdir(LABELED_DIR):
                full = os.path.join(LABELED_DIR, d)
                if os.path.isdir(full):
                    total += len(os.listdir(full))
        if total < 10:
            QMessageBox.warning(self, "样本不足",
                                f"当前共 {total} 张裁图，建议每类至少 20 张再训练。")
            return
        # ── 弹出参数设置对话框 ────────────────────────────────────────────────
        settings_dlg = TrainSettingsDialog(self)
        if settings_dlg.exec_() != QDialog.Accepted:
            return
        tp = settings_dlg.get_params()

        if not use_torch:
            # Baseline branch (no search support)
            self.lbl_status.setText("Training (baseline model)...")
            QApplication.processEvents()
            try:
                from resourcefield_classifier import ResourceFieldClassifier
                clf = ResourceFieldClassifier()
                acc = clf.train(data_dir=LABELED_DIR, save_path=MODEL_PATH, epochs=tp['epochs'])
                msg = f"Training complete  val_acc {acc*100:.1f}%  -> {MODEL_PATH}"
                self.lbl_status.setText(msg)
                print(f"[TRAIN] Finished. {msg}", flush=True)
            except Exception as ex:
                self.lbl_status.setText(f"❌ 训练失败：{ex}")
                QMessageBox.critical(self, "训练失败", str(ex))
            return

        # ── PyTorch 训练 ─────────────────────────────────────────────────────
        try:
            from torch_classifier import TorchResourceFieldClassifier  # noqa: F401
        except Exception:
            QMessageBox.critical(self, "错误", "检测到 PyTorch 但无法导入 torch_classifier.py")
            return

        # 删除旧模型文件，强制从头训练
        try:
            if os.path.exists(TORCH_MODEL_PATH):
                os.remove(TORCH_MODEL_PATH)
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
        except Exception:
            pass

        if tp['train_mode'] == 'fixed':
            self._run_fixed_train(tp)
        else:
            self._run_search_train(tp)

    # ── 固定参数训练 ─────────────────────────────────────────────────────────
    def _run_fixed_train(self, tp):
        from torch_classifier import TorchResourceFieldClassifier
        epochs_to_use = tp['epochs']
        self.lbl_status.setText("Training... please wait...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        QApplication.processEvents()

        def progress_cb(epoch, batch_idx, num_batches, train_loss=None, train_acc=None,
                        val_loss=None, val_acc=None):
            # Every batch: update the progress bar
            self.progress_bar.setMaximum(num_batches)
            self.progress_bar.setValue(batch_idx)
            self.progress_bar.setFormat(
                f"Epoch {epoch}/{epochs_to_use}   batch {batch_idx}/{num_batches}"
            )
            # Terminal: overwrite same line with batch progress
            try:
                bar_w = 20
                filled = int(bar_w * batch_idx / num_batches) if num_batches else 0
                bar = '█' * filled + '░' * (bar_w - filled)
                print(f"\r[TRAIN] epoch {epoch:>3}/{epochs_to_use}  [{bar}] {batch_idx}/{num_batches}",
                      end='', flush=True)
            except Exception:
                pass
            # End of epoch (val_acc slot carries test_acc): update status + log once
            if val_acc is not None:
                tr_s = f"  tr={train_acc*100:.1f}%" if train_acc is not None else ""
                val_label = "val" if tp['validate_during_training'] else "test"
                self.lbl_status.setText(
                    f"Epoch {epoch}/{epochs_to_use}{tr_s}   "
                    f"{val_label}={val_acc*100:.1f}%"
                )
                try:
                    tr_str = f"  tr={train_acc*100:.1f}%" if train_acc is not None else ""
                    print(
                        f"\r[TRAIN] epoch {epoch:>3}/{epochs_to_use}{tr_str}"
                        f"   {val_label}={val_acc*100:.1f}%",
                        flush=True
                    )
                except Exception:
                    pass
            QApplication.processEvents()

        try:
            clf = TorchResourceFieldClassifier()
            print(
                f"[TRAIN] Fixed: epochs={tp['epochs']} batch={tp['batch_size']} "
                f"lr={tp['lr']} model={tp['model_type']} scheduler={tp['lr_scheduler']}",
                flush=True
            )
            acc = clf.train(
                data_dir=LABELED_DIR,
                save_path=TORCH_MODEL_PATH,
                epochs=tp['epochs'],
                batch_size=tp['batch_size'],
                lr=tp['lr'],
                model_type=tp['model_type'],
                lr_scheduler=tp['lr_scheduler'],
                pretrained=tp['pretrained'],
                val_split=tp['val_split'],
                test_split=tp['test_split'],
                validate_during_training=tp['validate_during_training'],
                use_class_weights=tp['use_class_weights'],
                num_workers=tp['num_workers'],
                pin_memory=tp['pin_memory'],
                persistent_workers=tp['persistent_workers'],
                num_conv=tp['num_conv'],
                base_channels=tp['base_channels'],
                progress_callback=progress_cb,
            )
            msg = f"训练完成  test_acc={acc*100:.1f}%  -> {TORCH_MODEL_PATH}"
            self.progress_bar.setVisible(False)
            self.lbl_status.setText(msg)
            print(f"[TRAIN] Finished. {msg}", flush=True)
            self._show_misclassified(clf)
            self._show_predict_after_train(acc)
        except Exception as ex:
            self.progress_bar.setVisible(False)
            self.lbl_status.setText(f"❌ 训练失败：{ex}")
            QMessageBox.critical(self, "训练失败", str(ex))

    # ── 负样本还原：将 cluster_* 中的文件移回 negative/free/ ────────────────
    @staticmethod
    def _restore_negatives_to_free():
        """Move all images from cluster_* subdirs of negative/ back to negative/free/
        so that cluster_negatives() can re-cluster from a clean state."""
        import shutil as _shutil
        from pathlib import Path as _Path
        neg_root = _Path(LABELED_DIR) / "negative"
        if not neg_root.exists():
            return
        free_dir = neg_root / "free"
        free_dir.mkdir(parents=True, exist_ok=True)
        img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        SOURCE_DIRS = {"free", "individual", "alliance"}
        moved = 0
        for sub in list(neg_root.iterdir()):
            if not sub.is_dir() or sub.name in SOURCE_DIRS:
                continue
            # this is a cluster_* (or any other non-source) dir — move files back
            for f in list(sub.iterdir()):
                if f.is_file() and f.suffix.lower() in img_exts:
                    dst = free_dir / f.name
                    if dst.exists():   # avoid collision
                        dst = free_dir / f"{f.stem}_{moved}{f.suffix}"
                    _shutil.move(str(f), str(dst))
                    moved += 1
            # remove now-empty cluster dir
            try:
                sub.rmdir()
            except OSError:
                pass
        print(f"[K-SEARCH] restored {moved} negatives → negative/free/", flush=True)

    # ── 参数搜索训练 ─────────────────────────────────────────────────────────
    def _run_search_train(self, tp):
        import itertools
        import random as _random
        import shutil
        from torch_classifier import TorchResourceFieldClassifier

        search_k = tp.get('search_neg_k', [None])   # [None] means "don't change K"
        all_combos = list(itertools.product(
            tp['search_epochs'], tp['search_batch'], tp['search_lr'], search_k
        ))
        if tp['train_mode'] == 'random_search':
            _random.shuffle(all_combos)
            all_combos = all_combos[:tp['num_trials']]

        total_trials = len(all_combos)
        has_k_search = not (len(search_k) == 1 and search_k[0] is None)
        print(f"[SEARCH] Mode={tp['train_mode']}  total combos={total_trials}", flush=True)
        print(f"[SEARCH] model={tp['model_type']} scheduler={tp['lr_scheduler']}", flush=True)
        if has_k_search:
            print(f"[SEARCH] Negative K candidates={search_k}  (will re-cluster before each K change)", flush=True)

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        QApplication.processEvents()

        best_acc   = -1.0
        best_combo = None
        best_clf   = None
        results    = []
        tmp_path   = TORCH_MODEL_PATH + ".search_tmp.pth"

        _epoch_total = [0]
        _trial_num   = [0]
        _last_k      = [None]   # track last K used to avoid redundant re-clustering

        def progress_cb(epoch, batch_idx, num_batches, train_loss=None, train_acc=None,
                        val_loss=None, val_acc=None):
            # Every batch: update the progress bar
            self.progress_bar.setMaximum(num_batches)
            self.progress_bar.setValue(batch_idx)
            self.progress_bar.setFormat(
                f"[{_trial_num[0]}/{total_trials}] Epoch {epoch}/{_epoch_total[0]}   "
                f"batch {batch_idx}/{num_batches}"
            )
            # Terminal: overwrite same line with batch progress
            try:
                bar_w = 20
                filled = int(bar_w * batch_idx / num_batches) if num_batches else 0
                bar = '█' * filled + '░' * (bar_w - filled)
                print(f"\r[SEARCH] trial {_trial_num[0]}/{total_trials}  "
                      f"epoch {epoch:>3}/{_epoch_total[0]}  [{bar}] {batch_idx}/{num_batches}",
                      end='', flush=True)
            except Exception:
                pass
            # End of epoch: update status label + log once
            if val_acc is not None:
                tr_s = f"  tr={train_acc*100:.1f}%" if train_acc is not None else ""
                val_label = "val" if tp['validate_during_training'] else "test"
                self.lbl_status.setText(
                    f"[{_trial_num[0]}/{total_trials}] "
                    f"Epoch {epoch}/{_epoch_total[0]}{tr_s}   "
                    f"{val_label}={val_acc*100:.1f}%"
                )
                try:
                    tr_str = f"  tr={train_acc*100:.1f}%" if train_acc is not None else ""
                    print(
                        f"\r[SEARCH] trial {_trial_num[0]}/{total_trials}  "
                        f"epoch {epoch:>3}/{_epoch_total[0]}{tr_str}"
                        f"   {val_label}={val_acc*100:.1f}%",
                        flush=True
                    )
                except Exception:
                    pass
            QApplication.processEvents()

        try:
            for (epochs_v, batch_v, lr_v, neg_k_v) in all_combos:
                _trial_num[0] += 1
                _epoch_total[0] = epochs_v

                # re-cluster negatives if K changed
                if has_k_search and neg_k_v is not None and neg_k_v != _last_k[0]:
                    self.lbl_status.setText(f"[K={neg_k_v}] 正在重新聚类负样本...")
                    QApplication.processEvents()
                    self._restore_negatives_to_free()
                    try:
                        from cluster_negatives import cluster_negatives as _cn
                        _cn(data_dir=LABELED_DIR, k=neg_k_v)
                    except Exception as _ke:
                        print(f"[K-SEARCH] cluster failed: {_ke}", flush=True)
                    _last_k[0] = neg_k_v

                k_tag = f" K={neg_k_v}" if has_k_search else ""
                print(
                    f"[SEARCH] Trial {_trial_num[0]}/{total_trials}: "
                    f"epochs={epochs_v} batch={batch_v} lr={lr_v:.6f}{k_tag}",
                    flush=True
                )
                self.lbl_status.setText(
                    f"搜索 [{_trial_num[0]}/{total_trials}]  "
                    f"epochs={epochs_v} batch={batch_v} lr={lr_v:.6f}{k_tag}"
                )
                QApplication.processEvents()

                clf = TorchResourceFieldClassifier()
                acc = clf.train(
                    data_dir=LABELED_DIR,
                    save_path=tmp_path,
                    epochs=epochs_v,
                    batch_size=batch_v,
                    lr=lr_v,
                    model_type=tp['model_type'],
                    lr_scheduler=tp['lr_scheduler'],
                    pretrained=tp['pretrained'],
                    val_split=tp['val_split'],
                    test_split=tp['test_split'],
                    validate_during_training=tp['validate_during_training'],
                    use_class_weights=tp['use_class_weights'],
                    num_workers=tp['num_workers'],
                    pin_memory=tp['pin_memory'],
                    persistent_workers=tp['persistent_workers'],
                    num_conv=tp['num_conv'],
                    base_channels=tp['base_channels'],
                    progress_callback=progress_cb,
                )
                results.append((_trial_num[0], epochs_v, batch_v, lr_v, neg_k_v, acc))
                is_best = acc > best_acc
                print(
                    f"[SEARCH] Trial {_trial_num[0]} result: acc={acc*100:.1f}%"
                    f"{'  <- BEST' if is_best else ''}",
                    flush=True
                )
                if is_best:
                    best_acc = acc
                    best_combo = (epochs_v, batch_v, lr_v, neg_k_v)
                    best_clf = clf
                    try:
                        shutil.copy(tmp_path, TORCH_MODEL_PATH)
                    except Exception as cp_ex:
                        print(f"[SEARCH] Warning: copy failed: {cp_ex}", flush=True)

            # 清理临时文件
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

            # 打印结果表
            print("\n[SEARCH] ===== 搜索结果 =====", flush=True)
            k_col = f"  {'K':>4}" if has_k_search else ""
            print(f"{'Trial':>5}  {'Epochs':>6}  {'Batch':>5}  {'LR':>10}{k_col}  {'Acc':>7}", flush=True)
            for (t, e, b, lr, kv, a) in sorted(results, key=lambda x: -x[5]):
                marker = " <- best" if (e, b, lr, kv) == best_combo else ""
                k_val = f"  {kv:>4}" if has_k_search else ""
                print(f"{t:>5}  {e:>6}  {b:>5}  {lr:>10.6f}{k_val}  {a*100:>6.1f}%{marker}", flush=True)
            print("[SEARCH] ====================\n", flush=True)

            if best_combo:
                k_part = f" K={best_combo[3]}" if has_k_search else ""
                msg = (
                    f"搜索完成 ({total_trials}次)  最优: "
                    f"epochs={best_combo[0]} batch={best_combo[1]} lr={best_combo[2]:.6f}{k_part}  "
                    f"acc={best_acc*100:.1f}%"
                )
            else:
                msg = f"搜索完成 ({total_trials}次)  best_acc={best_acc*100:.1f}%"
            self.progress_bar.setVisible(False)
            self.lbl_status.setText(msg)
            if best_clf is not None:
                self._show_misclassified(best_clf)
            self._show_predict_after_train(best_acc)
        except Exception as ex:
            self.progress_bar.setVisible(False)
            self.lbl_status.setText(f"❌ 搜索训练失败：{ex}")
            QMessageBox.critical(self, "搜索训练失败", str(ex))

    # ── 误分类图片瀏览 ───────────────────────────────────────────────────────
    def _show_misclassified(self, clf):
        try:
            items = getattr(clf, 'last_misclassified', [])
            if not items:
                return
            dlg = MisclassifiedDialog(items, parent=self)
            dlg.exec_()
        except Exception as _ex:
            print(f"[TRAIN] _show_misclassified error: {_ex}", flush=True)

    # ── 训练完成后展示预测 ────────────────────────────────────────────────────
    def _show_predict_after_train(self, acc):
        try:
            idx = next((i for i in range(len(self.image_paths)) if i not in self.labeled_set), None)
            if idx is None and self.image_paths:
                idx = 0
            if idx is not None:
                img_path = self.image_paths[idx]
                img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    scanner = FieldScanner()
                    fw, fh, frame = scanner.detect_yellow_frame(img)
                    if frame is not None:
                        ax, ay, _, _ = frame
                        h, w = img.shape[:2]
                        boxes = scanner.build_grid(ax, ay, fw, fh, 0, 0, w, h)
                        predict_boxes = [(tx, ty, tx+gw, ty+gh) for (tx, ty, gw, gh) in boxes]
                        dlg = PredictWindow(img, predict_boxes=predict_boxes, parent=self)
                        dlg.exec_()
                    else:
                        QMessageBox.information(self, "训练完成",
                            f"训练完成：{acc*100:.1f}%\n未检测到锚点，无法展示预测。")
                else:
                    QMessageBox.information(self, "训练完成",
                        f"训练完成：{acc*100:.1f}%\n无法读取样本图。")
            else:
                QMessageBox.information(self, "训练完成",
                    f"训练完成：{acc*100:.1f}%\n未找到样本图片。")
        except Exception as ex:
            QMessageBox.information(self, "训练完成",
                f"训练完成：{acc*100:.1f}%\n展示预测时出错：{ex}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import io
    # Ensure UTF-8 output on Windows consoles
    if sys.platform == "win32":
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Set Segoe UI as primary font so digits/Latin always use correct (non-CJK) glyphs.
    # Qt automatically falls back to the system CJK font for characters not in Segoe UI,
    # so Chinese labels still render correctly without any CSS tricks.
    from PyQt5.QtGui import QFont
    _base_font = QFont("Segoe UI", 9)
    _base_font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(_base_font)
    win = LabelerWindow()
    win.show()
    sys.exit(app.exec_())
