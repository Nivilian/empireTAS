"""
A small, dependency-free ResourceFieldClassifier used for fast experiments.

This implementation uses simple color histograms (HSV) as features and
computes per-class centroids. It provides a minimal `train`, `load`, and
`predict` API so the labeler can call it without requiring PyTorch.

Model is saved as a numpy .npz archive with arrays `classes` and `centroids`.
"""

import os
import numpy as np
import cv2
from pathlib import Path

IMG_W, IMG_H = 96, 64


def _extract_feat(bgr_img):
    """Return a 1D float32 feature vector for the crop (HSV histograms)."""
    if bgr_img is None or bgr_img.size == 0:
        return None
    img = cv2.resize(bgr_img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_bins, s_bins = 16, 8
    h_hist = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256])
    feat = np.concatenate([h_hist.flatten(), s_hist.flatten()]).astype(np.float32)
    # L2 normalize
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


class ResourceFieldClassifier:
    def __init__(self):
        self.classes = []
        self.centroids = None  # numpy array shape (C, D)

    def _gather_training_files(self, data_dir):
        """Return dict: class_name -> list of file paths.

        Supports two layouts:
         - <data_dir>/<terrain>/<occ>/* (labeler default)
         - <data_dir>/<class_name>/*
        """
        data_dir = Path(data_dir)
        mapping = {}
        if not data_dir.exists():
            return mapping
        # check two-level layout (terrain/occ)
        for terrain_dir in data_dir.iterdir():
            if not terrain_dir.is_dir():
                continue
            # if contains subdirs -> use terrain/occ layout
            subdirs = [d for d in terrain_dir.iterdir() if d.is_dir()]
            if subdirs:
                for occ_dir in subdirs:
                    occ = occ_dir.name
                    class_name = terrain_dir.name if occ in ("free", "空") else f"{terrain_dir.name}_{occ}"
                    files = [str(p) for p in occ_dir.iterdir() if p.is_file()]
                    if files:
                        mapping.setdefault(class_name, []).extend(files)
            else:
                # treat terrain_dir as class folder
                files = [str(p) for p in terrain_dir.iterdir() if p.is_file()]
                if files:
                    mapping.setdefault(terrain_dir.name, []).extend(files)
        # also check flat class folders at root
        for p in data_dir.iterdir():
            if p.is_dir():
                files = [str(x) for x in p.iterdir() if x.is_file()]
                if files and p.name not in mapping:
                    mapping.setdefault(p.name, []).extend(files)
        return mapping

    def train(self, data_dir, save_path=None, epochs=1):
        mapping = self._gather_training_files(data_dir)
        if not mapping:
            raise RuntimeError(f"No training data found in {data_dir}")
        classes = []
        feats = []
        for cls, files in mapping.items():
            cls_feats = []
            for f in files:
                try:
                    img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
                    feat = _extract_feat(img)
                    if feat is not None:
                        cls_feats.append(feat)
                except Exception:
                    continue
            if not cls_feats:
                continue
            classes.append(cls)
            feats.append(np.stack(cls_feats, axis=0))
        if not classes:
            raise RuntimeError("No valid images found for any class")
        # compute per-class centroid
        centroids = np.stack([np.mean(f, axis=0) for f in feats], axis=0)
        # normalize centroids
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        centroids = centroids / norms
        self.classes = classes
        self.centroids = centroids.astype(np.float32)
        if save_path:
            np.savez_compressed(save_path, classes=np.array(self.classes), centroids=self.centroids)
        # compute simple training-set accuracy (self-match)
        correct = total = 0
        for i, cls in enumerate(classes):
            for vec in feats[i]:
                dists = np.linalg.norm(self.centroids - vec, axis=1)
                pred = int(np.argmin(dists))
                if pred == i:
                    correct += 1
                total += 1
        acc = (correct / total) if total > 0 else 0.0
        return acc

    def load(self, path):
        try:
            data = np.load(path, allow_pickle=True)
            self.classes = [str(x) for x in data['classes'].tolist()]
            self.centroids = data['centroids'].astype(np.float32)
            return True
        except Exception:
            return False

    def predict(self, bgr_crop):
        feat = _extract_feat(bgr_crop)
        if feat is None or self.centroids is None:
            return "unknown", 0.0
        dists = np.linalg.norm(self.centroids - feat, axis=1)
        idx = int(np.argmin(dists))
        # confidence: convert distance to a pseudo-probability (smaller dist -> higher confidence)
        inv = 1.0 / (dists[idx] + 1e-6)
        sims = inv / (np.sum(1.0 / (dists + 1e-6)))
        conf = float(sims)
        return self.classes[idx], conf

