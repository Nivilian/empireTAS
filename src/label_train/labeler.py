"""
地块标注工具  —  独立运行
==========================
用法：  python labeler.py

遍历 images/trainingBackup/ 下所有图片，
标注后将裁图保存到 images/labeledImages/<地块_占领>/

画框（按住键 + 拖拽）：
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
MODEL_PATH  = os.path.join(_SRC_DIR, "clay_model.pth")

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
            scanner = FieldScanner()
            # 只用detect_yellow_frame和build_grid，不做分类
            fw, fh, frame = scanner.detect_yellow_frame(self.orig_img)
            if frame is None:
                self.grid_cells = []
                return
            x, y, w, h = frame
            grid = scanner.build_grid(x, y, fw, fh, 0, 0, self.orig_img.shape[1], self.orig_img.shape[0])
            self.grid_cells = [(tx, ty, fw, fh) for (tx, ty, fw, fh) in grid]
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

    def resizeEvent(self, _e):
        self._rebuild_pixmap()
        self.update()

    # ── 坐标转换 ──────────────────────────────────────────────────────────────
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
        elif e.key() == Qt.Key_Q:
            self.occ_mode = "individual"
        elif e.key() == Qt.Key_E:
            self.occ_mode = "alliance"
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() in KEY_LABEL:
            self.active_terrain = None
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
            for idx, (tx, ty, fw, fh) in enumerate(self.grid_cells):
                if self._point_in_diamond(x, y, tx, ty, fw, fh):
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
                            # update existing
                            self.annotations[ann_idx][4] = [terrain, occ]
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
            pen = QPen(QColor(0, 200, 200, 160), 1)
            p.setPen(pen)
            for (tx, ty, gw, gh) in self.grid_cells:
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
            cls_dir = os.path.join(LABELED_DIR, terrain, occ)
            os.makedirs(cls_dir, exist_ok=True)
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
        if self.orig_img is None or not self.annotations:
            return None
        base = os.path.splitext(src_name)[0]
        out = []
        for ann in self.annotations:
            # only grid-based annotations carry is_grid flag
            is_grid = (len(ann) >= 7 and ann[6])
            if not is_grid:
                continue
            x1, y1, x2, y2 = ann[0], ann[1], ann[2], ann[3]
            cx = int(x1 + (x2 - x1) / 2)
            cy = int(y1 + (y2 - y1) / 2)
            fw = x2 - x1
            fh = y2 - y1
            out.append({"cx": cx, "cy": cy, "fw": fw, "fh": fh})
        if not out:
            return None
        cal_dir = os.path.join(LABELED_DIR, "calibration")
        os.makedirs(cal_dir, exist_ok=True)
        fname = os.path.join(cal_dir, f"{base}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return fname

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
        for i, (tx, ty, fw, fh) in enumerate(self.grid_cells):
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
    """左：真实标注；右：模型对当前图的预测。"""
    def __init__(self, orig_img, annotations, parent=None):
        super().__init__(parent)
        self.setWindowTitle("标注 vs 模型识别  对比")
        self.resize(1300, 650)

        gt_img   = self._draw_gt(orig_img, annotations)
        pred_img = self._draw_pred(orig_img, annotations)

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
        hdr.addWidget(QLabel("🤖  模型预测（框选范围取自标注）  "))

        layout = QVBoxLayout()
        layout.addLayout(hdr)
        layout.addWidget(scroll)
        self.setLayout(layout)

    # ── 绘制真实标注图 ────────────────────────────────────────────────────────
    @staticmethod
    def _draw_gt(orig, annotations):
        img = orig.copy()
        for ann in annotations:
            # support variable-length annotations
            x1, y1, x2, y2 = ann[0], ann[1], ann[2], ann[3]
            label = ann[4] if len(ann) > 4 else ["unknown", "free"]
            qcolor = ann[5] if len(ann) > 5 and isinstance(ann[5], QColor) else QColor(255, 255, 255)
            terrain, occ = label[0], label[1]
            bgr = (qcolor.blue(), qcolor.green(), qcolor.red())
            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
            label_str = f"[{terrain}, {occ}]"
            cv2.putText(img, label_str, (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(img, label_str, (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return img

    # ── 绘制模型预测图 ────────────────────────────────────────────────────────
    @staticmethod
    def _draw_pred(orig, annotations):
        img = orig.copy()
        clf = None
        if os.path.exists(MODEL_PATH):
            try:
                sys.path.insert(0, _SRC_DIR)
                from clay_classifier import ClayClassifier
                clf = ClayClassifier()
                if not clf.load(MODEL_PATH):
                    clf = None
            except Exception:
                clf = None

        for ann in annotations:
            x1, y1, x2, y2 = ann[0], ann[1], ann[2], ann[3]
            pred_label = "无模型"
            pred_color = (128, 128, 128)
            # ground-truth folder name derived from label if available
            gt_folder = None
            if len(ann) > 4:
                lbl = ann[4]
                if isinstance(lbl, (list, tuple)) and len(lbl) >= 2:
                    terrain, occ = lbl[0], lbl[1]
                    occ_en = "free" if occ in ("空", "free") else ("individual" if occ in ("散人","individual") else "alliance")
                    gt_folder = terrain if occ_en == "free" else f"{terrain}_{occ_en}"

            if clf is not None:
                crop = orig[y1:y2, x1:x2]
                if crop.size > 0:
                    name, conf = clf.predict(crop)
                    pred_label = f"{name} {conf:.0%}"
                    if gt_folder is not None:
                        pred_color = (0, 200, 0) if name == gt_folder else (0, 60, 220)

            cv2.rectangle(img, (x1, y1), (x2, y2), pred_color, 2)
            cv2.putText(img, pred_label, (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(img, pred_label, (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        tuple(int(c) for c in pred_color), 1)
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
            self.lbl_status.setText(msg)
            self._go_next()

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
        if not self.canvas.annotations:
            self.lbl_status.setText("⚠ 请先画标注框再对比")
            return
        dlg = CompareWindow(self.canvas.orig_img, self.canvas.annotations, self)
        dlg.exec_()

    # ── 训练 ─────────────────────────────────────────────────────────────────
    def _do_train(self):
        try:
            from clay_classifier import ClayClassifier
        except ImportError:
            QMessageBox.critical(self, "错误", "找不到 clay_classifier.py")
            return
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
        self.lbl_status.setText("🚀 训练中，请稍候…")
        QApplication.processEvents()
        try:
            clf = ClayClassifier()
            acc = clf.train(data_dir=LABELED_DIR, save_path=MODEL_PATH, epochs=30)
            self.lbl_status.setText(
                f"✅ 训练完成  验证精度 {acc*100:.1f}%  → {MODEL_PATH}"
            )
            # 训练完自动弹对比窗口
            if self.canvas.orig_img is not None and self.canvas.annotations:
                dlg = CompareWindow(self.canvas.orig_img, self.canvas.annotations, self)
                dlg.exec_()
            else:
                QMessageBox.information(
                    self, "训练完成",
                    f"验证精度：{acc*100:.1f}%\n模型：{MODEL_PATH}\n\n"
                    f"选择一张有标注框的图片后点「🔍 对比」可查看效果。"
                )
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
