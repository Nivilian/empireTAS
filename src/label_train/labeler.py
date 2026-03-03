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
    QApplication, QDialog, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QPushButton, QScrollArea, QShortcut,
    QSplitter, QVBoxLayout, QWidget,
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
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() in KEY_LABEL:
            self.active_terrain = None
        elif e.key() == Qt.Key_N:
            self.active_terrain = None
        elif e.key() == Qt.Key_Space:
            self.clear_mode = False
        elif e.key() in (Qt.Key_Q, Qt.Key_E):
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
                    # If space-clear mode is active, remove existing annotation on this grid
                    if getattr(self, 'clear_mode', False):
                        ann_idx = None
                        for i, ann in enumerate(self.annotations):
                            gx, gy = ann[0], ann[1]
                            if gx == tx and gy == ty:
                                ann_idx = i
                                break
                        if ann_idx is not None:
                            self.annotations.pop(ann_idx)
                            self.update()
                        return
                    # 检查是否已有标注
                    ann_idx = None
                    for i, ann in enumerate(self.annotations):
                        gx, gy = ann[0], ann[1]
                        if gx == tx and gy == ty:
                            ann_idx = i
                            break
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
        self.btn_gen_neg.setFixedHeight(28)
        self.btn_auto_neg.setFixedHeight(28)
        self.btn_gen_neg.clicked.connect(self._do_generate_negatives)
        self.btn_auto_neg.clicked.connect(self._toggle_auto_negatives)
        self.auto_generate_negatives = False

        for btn in [self.btn_prev, self.btn_next, self.btn_save, self.btn_undo,
                self.btn_clear, self.btn_compare, self.btn_train, self.btn_gen_neg, self.btn_auto_neg]:
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
        top_bar.addWidget(self.btn_undo)
        top_bar.addWidget(self.btn_clear)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.btn_compare)
        top_bar.addWidget(self.btn_train)
        top_bar.addStretch()
        top_bar.addWidget(self.lbl_status)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.addLayout(top_bar)
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
        epochs_to_use = 30
        self.lbl_status.setText("🚀 训练中，请稍候…")
        QApplication.processEvents()
        def progress_cb(epoch, batch_idx, num_batches, train_loss=None, train_acc=None, val_loss=None, val_acc=None):
            # Build succinct status strings
            tr_acc_s = f" tr={train_acc*100:.1f}%" if train_acc is not None else ""
            val_acc_s = f" val={val_acc*100:.1f}%" if val_acc is not None else ""
            status_text = f"🚀 训练中  epoch {epoch}/{epochs_to_use}  batch {batch_idx}/{num_batches}{tr_acc_s}{val_acc_s}"
            try:
                self.lbl_status.setText(status_text)
            except Exception:
                try:
                    self.lbl_status.setText(f"🚀 训练中  epoch {epoch}/{epochs_to_use}  batch {batch_idx}/{num_batches}")
                except Exception:
                    pass
            # Also print to terminal for logging/monitoring
            try:
                print(f"[TRAIN] {status_text}", flush=True)
            except Exception:
                pass
            QApplication.processEvents()
        try:
            if use_torch:
                try:
                    from torch_classifier import TorchResourceFieldClassifier
                except Exception:
                    QMessageBox.critical(self, "错误", "检测到 PyTorch 但无法导入 torch_classifier.py")
                    return
                clf = TorchResourceFieldClassifier()
                print(f"[TRAIN] Starting PyTorch training for {epochs_to_use} epochs...", flush=True)
                acc = clf.train(data_dir=LABELED_DIR, save_path=TORCH_MODEL_PATH, epochs=epochs_to_use, progress_callback=progress_cb)
                msg = f"✅ 训练完成  验证精度 {acc*100:.1f}%  → {TORCH_MODEL_PATH}"
                self.lbl_status.setText(msg)
                print(f"[TRAIN] Finished. {msg}", flush=True)
            else:
                from resourcefield_classifier import ResourceFieldClassifier
                clf = ResourceFieldClassifier()
                # baseline training is fast; show simple status before/after
                self.lbl_status.setText("🚀 训练中（基线模型）…")
                print(f"[TRAIN] Starting baseline training for {epochs_to_use} epochs...", flush=True)
                QApplication.processEvents()
                acc = clf.train(data_dir=LABELED_DIR, save_path=MODEL_PATH, epochs=epochs_to_use)
                msg = f"✅ 训练完成  验证精度 {acc*100:.1f}%  → {MODEL_PATH}"
                self.lbl_status.setText(msg)
                print(f"[TRAIN] Finished. {msg}", flush=True)
            # 训练完成后弹出对比：使用一张未标注的备份图并在其网格上展示模型预测
            try:
                # choose first unannotated image
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
                            # boxes is list of (tx,ty,fw,fh) -> convert to x1,y1,x2,y2
                            predict_boxes = [(tx, ty, tx+gw, ty+gh) for (tx, ty, gw, gh) in boxes]
                            dlg = PredictWindow(img, predict_boxes=predict_boxes, parent=self)
                            dlg.exec_()
                        else:
                            QMessageBox.information(self, "训练完成", f"训练完成：{acc*100:.1f}%\n但未能在样本图中检测到锚点以展示预测。")
                    else:
                        QMessageBox.information(self, "训练完成", f"训练完成：{acc*100:.1f}%\n无法读取样本图。")
                else:
                    QMessageBox.information(self, "训练完成", f"训练完成：{acc*100:.1f}%\n未找到样本图片用于展示。")
            except Exception as ex:
                QMessageBox.information(self, "训练完成", f"训练完成：{acc*100:.1f}%\n但展示预测时出错：{ex}")
        except Exception as ex:
            self.lbl_status.setText(f"❌ 训练失败：{ex}")
            QMessageBox.critical(self, "训练失败", str(ex))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = LabelerWindow()
    win.show()
    sys.exit(app.exec_())
