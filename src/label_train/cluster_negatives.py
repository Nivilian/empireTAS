"""
cluster_negatives.py
====================
自动将 labeledImages/negative/ 下的所有负样本图片
按视觉相似度聚类，并移入 negative/<cluster_name>/ 子目录。

聚类依据：HSV 颜色直方图（H:18 + S:8 + V:8 = 34 维特征），用 cv2.kmeans。

用法（命令行）：
    python cluster_negatives.py [--k 5] [--data_dir <path>] [--dry_run]

用法（代码调用）：
    from cluster_negatives import cluster_negatives
    report = cluster_negatives(data_dir, k=5, dry_run=False)
    # report: {cluster_name: [file_list], ...}

聚类完成后子目录名为 cluster_0 / cluster_1 / ...
用户可手动将其重命名为 city / empty / small_resource 等，
_gather_training_files 会自动识别新目录名。
"""

from __future__ import annotations

import os
import sys
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np

# ─── 可调参数 ────────────────────────────────────────────────────────────────

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "images" / "labeledImages"
_IMG_EXTENSIONS   = {".png", ".jpg", ".jpeg", ".bmp"}
_RESIZE           = (32, 32)        # 特征提取用的缩放大小
_HIST_BINS        = (18, 8, 8)      # H / S / V 直方图桶数 → 34 维特征向量
_MAX_ITER         = 100
_EPSILON          = 1e-4

# ─── 特征提取 ────────────────────────────────────────────────────────────────

def extract_feature(
    bgr: np.ndarray,
    resize: tuple[int, int] | None = None,
    hist_bins: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """返回归一化 HSV 直方图特征向量 (float32, shape=(sum(hist_bins),))."""
    r = resize    if resize    else _RESIZE
    b = hist_bins if hist_bins else _HIST_BINS
    small = cv2.resize(bgr, r, interpolation=cv2.INTER_AREA)
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h_bins, s_bins, v_bins = b
    feats = []
    for ch, bins, rang in (
        (0, h_bins, (0, 180)),
        (1, s_bins, (0, 256)),
        (2, v_bins, (0, 256)),
    ):
        hist = cv2.calcHist([hsv], [ch], None, [bins], rang).flatten()
        total = hist.sum()
        hist  = hist / total if total > 0 else hist
        feats.append(hist)
    return np.concatenate(feats).astype(np.float32)


# ─── 图片收集 ────────────────────────────────────────────────────────────────

def collect_negative_images(data_dir: Path) -> list[Path]:
    """
    收集 data_dir/negative/ 下所有图片。
    已在某个子目录（非 free/individual/alliance 但也非 cluster_* ）的文件
    视为已聚类，跳过。
    只处理直接在 negative/free、negative/individual、negative/alliance
    或 negative/ 根目录下的文件（尚未被专项命名子目录接管的图片）。
    """
    neg_root = data_dir / "negative"
    if not neg_root.exists():
        return []

    SOURCE_DIRS = {"free", "individual", "alliance"}
    images: list[Path] = []

    for item in neg_root.iterdir():
        if item.is_dir():
            # 只从 free/individual/alliance 子目录中收集（待聚类的来源）
            if item.name in SOURCE_DIRS:
                for f in item.iterdir():
                    if f.is_file() and f.suffix.lower() in _IMG_EXTENSIONS:
                        images.append(f)
        elif item.is_file() and item.suffix.lower() in _IMG_EXTENSIONS:
            images.append(item)

    return images


# ─── 主聚类函数 ──────────────────────────────────────────────────────────────

def cluster_negatives(
    data_dir: str | Path | None = None,
    k: int = 5,
    dry_run: bool = False,
    prefix: str = "cluster",
    resize: int | None = None,
    hist_bins: tuple[int, int, int] | None = None,
) -> dict[str, list[str]]:
    """
    对 negative/ 下的图片进行 K-means 聚类并移入子目录。

    Args:
        data_dir:  labeledImages 根目录，默认 images/labeledImages
        k:         聚类数量
        dry_run:   True 时只打印不移动文件
        prefix:    子目录名前缀（默认 "cluster"），结果为 cluster_0, cluster_1...
        resize:    特征提取时缩放到的正方形边长（覆盖模块默认 _RESIZE）
        hist_bins: (H桶数, S桶数, V桶数)，覆盖模块默认 _HIST_BINS

    Returns:
        {cluster_dir_name: [file_path, ...]}
    """
    # Allow callers to override module-level defaults
    effective_resize = (resize, resize) if resize else _RESIZE
    effective_bins   = hist_bins if hist_bins else _HIST_BINS
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    neg_root = data_dir / "negative"

    images = collect_negative_images(data_dir)
    if not images:
        print("[cluster_negatives] 未找到可聚类的负样本图片。")
        return {}

    n = len(images)
    k = min(k, n)
    print(f"[cluster_negatives] 共找到 {n} 张负样本图片，聚类数 k={k}")

    # ── 提取特征 ──
    features = []
    valid_images: list[Path] = []
    for p in images:
        bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        features.append(extract_feature(bgr, resize=effective_resize, hist_bins=effective_bins))
        valid_images.append(p)

    if not valid_images:
        print("[cluster_negatives] 无法读取任何图片。")
        return {}

    X = np.stack(features)   # (N, 34)

    # ── K-means（使用 cv2.kmeans，无需额外依赖）──
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, _MAX_ITER, _EPSILON)
    _, labels, centers = cv2.kmeans(
        X, k, None, criteria, attempts=10, flags=cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()   # (N,)

    # ── 整理结果 ──
    result: dict[str, list[str]] = {}
    for cluster_id in range(k):
        cluster_name = f"{prefix}_{cluster_id}"
        mask = labels == cluster_id
        cluster_files = [str(valid_images[i]) for i in range(len(valid_images)) if mask[i]]
        result[cluster_name] = cluster_files

        print(f"\n  [{cluster_name}]  {len(cluster_files)} 张")
        # 打印几个代表性文件名，方便用户判断内容
        for fp in cluster_files[:4]:
            print(f"    {Path(fp).name}")
        if len(cluster_files) > 4:
            print(f"    ... 共 {len(cluster_files)} 张")

        if dry_run:
            continue

        # 移动文件
        dest_dir = neg_root / cluster_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for fp in cluster_files:
            src_path = Path(fp)
            dst_path = dest_dir / src_path.name
            # 避免重名
            if dst_path.exists():
                stem = dst_path.stem
                suffix = dst_path.suffix
                counter = 1
                while dst_path.exists():
                    dst_path = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
            shutil.move(str(src_path), str(dst_path))

    if not dry_run:
        print(f"\n[cluster_negatives] 完成！文件已移入 {neg_root}")
        print("提示：可手动将 cluster_0, cluster_1... 重命名为 city, empty, small_resource 等，")
        print("      训练器会自动识别新目录名（如 negative_city, negative_empty）。")
    else:
        print("\n[cluster_negatives] 演习模式（dry_run=True），未移动任何文件。")

    return result


# ─── 命令行入口 ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="自动聚类 negative 负样本")
    parser.add_argument("--k",        type=int,  default=5,    help="聚类数（默认 5）")
    parser.add_argument("--data_dir", type=str,  default=None, help="labeledImages 路径")
    parser.add_argument("--prefix",   type=str,  default="cluster", help="子目录前缀")
    parser.add_argument("--dry_run",  action="store_true",     help="只打印不移动文件")
    args = parser.parse_args()

    cluster_negatives(
        data_dir=args.data_dir,
        k=args.k,
        dry_run=args.dry_run,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
