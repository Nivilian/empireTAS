"""
资源地格子 CNN 分类器
==================
类别：
    0  森林
    1  粘土
    2  石头

占领状态：
    0  空地（free）
    1  联盟（alliance）
    2  散人（individual）

训练数据目录结构（由 labeler.py 自动建立）：
    images/training/
        forest_free/          *.png
        clay_alliance/        *.png
        stone_individual/     *.png

快速使用：
    clf = ResourceFieldClassifier()
    clf.load("resourcefield_model.pth")          # 加载已训练模型
    label, conf = clf.predict(bgr_crop)  # 推理

训练：
    clf = ResourceFieldClassifier()
    acc = clf.train(data_dir, save_path="resourcefield_model.pth", epochs=30)
"""

# TODO: 参考 clay_classifier.py 实现 ResourceFieldClassifier
# 支持 label 格式为 [地块类型, 占领状态]，如 ["forest", "free"]
# 训练数据目录结构建议为 images/training/<地块>_<占领>/

import os
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split
    import torchvision.transforms as T
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

IMG_W, IMG_H = 96, 64

# ...模型和数据集定义，参考 clay_classifier.py ...

class ResourceFieldClassifier:
    pass
