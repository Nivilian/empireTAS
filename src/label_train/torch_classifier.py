import os
import json
import numpy as np
from pathlib import Path
import cv2
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as T
    import torchvision.models as models
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
except Exception:
    # If torch not installed, the file can still be imported but functionality will raise
    torch = None


IMG_SIZE = 128


class _ImageDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items  # list of (path, label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, lbl = self.items[idx]
        # read with PIL for torchvision transforms
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, lbl


class TorchResourceFieldClassifier:
    """PyTorch-based classifier with train/load/predict API similar to baseline.

    Saves model to a .pth file and classes to a companion .json.
    """
    def __init__(self, device=None):
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = None
        self.classes = []

    def _gather_training_files(self, data_dir):
        data_dir = Path(data_dir)
        mapping = {}
        if not data_dir.exists():
            return mapping
        for terrain_dir in data_dir.iterdir():
            if not terrain_dir.is_dir():
                continue
            subdirs = [d for d in terrain_dir.iterdir() if d.is_dir()]
            if subdirs:
                for occ_dir in subdirs:
                    occ = occ_dir.name
                    class_name = terrain_dir.name if occ in ("free", "空") else f"{terrain_dir.name}_{occ}"
                    files = [str(p) for p in occ_dir.iterdir() if p.is_file()]
                    if files:
                        mapping.setdefault(class_name, []).extend(files)
            else:
                files = [str(p) for p in terrain_dir.iterdir() if p.is_file()]
                if files:
                    mapping.setdefault(terrain_dir.name, []).extend(files)
        for p in data_dir.iterdir():
            if p.is_dir():
                files = [str(x) for x in p.iterdir() if x.is_file()]
                if files and p.name not in mapping:
                    mapping.setdefault(p.name, []).extend(files)
        return mapping

    def _build_model(self, num_classes):
        # use pretrained ResNet18 and replace final fc (lightweight and performant)
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    def _build_custom(self, num_classes, num_conv=3, base_channels=32):
        layers = []
        in_ch = 3
        out_ch = base_channels
        for i in range(num_conv):
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
        model = nn.Sequential(
            *layers,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes)
        )
        return model

    def train(self, data_dir, save_path=None, epochs=20, batch_size=32, lr=1e-3,
              val_split=0.2, num_workers=4, img_size=IMG_SIZE,
              model_type='resnet18', pretrained=True,
              num_conv=3, base_channels=32, device=None,
              progress_callback=None):
        mapping = self._gather_training_files(data_dir)
        if not mapping:
            raise RuntimeError(f"No training data found in {data_dir}")
        classes = []
        items = []
        for cls, files in mapping.items():
            if not files:
                continue
            idx = len(classes)
            classes.append(cls)
            for f in files:
                items.append((f, idx))
        if not classes:
            raise RuntimeError("No valid images found for any class")
        self.classes = classes

        # split
        rng = np.random.default_rng(12345)
        idxs = np.arange(len(items))
        rng.shuffle(idxs)
        cut = int(len(items) * (1 - val_split))
        train_idx, val_idx = idxs[:cut], idxs[cut:]
        train_items = [items[i] for i in train_idx]
        val_items = [items[i] for i in val_idx]

        # transforms
        IMG_SIZE_LOCAL = img_size
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_tf = T.Compose([
            T.Resize((IMG_SIZE_LOCAL, IMG_SIZE_LOCAL)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
            normalize,
        ])
        val_tf = T.Compose([
            T.Resize((IMG_SIZE_LOCAL, IMG_SIZE_LOCAL)),
            T.ToTensor(),
            normalize,
        ])

        train_ds = _ImageDataset(train_items, transform=train_tf)
        val_ds = _ImageDataset(val_items, transform=val_tf)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(1, num_workers//2))

        # build model according to requested type
        if model_type == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, len(classes))
        elif model_type == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=pretrained)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, len(classes))
        elif model_type == 'custom':
            model = self._build_custom(len(classes), num_conv=num_conv, base_channels=base_channels)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_path = None
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = total = 0
            num_batches = len(train_loader)
            for batch_idx, (imgs, labels) in enumerate(train_loader):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                # report batch progress if requested
                if progress_callback:
                    try:
                        progress_callback(epoch, batch_idx + 1, num_batches, train_loss=(running_loss / max(1, total)), train_acc=(correct / max(1, total)))
                    except Exception:
                        pass
            train_loss = running_loss / max(1, total)
            train_acc = correct / max(1, total)

            # validate
            model.eval()
            vloss = 0.0
            vcorrect = vtotal = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                    vloss += loss.item() * imgs.size(0)
                    preds = logits.argmax(dim=1)
                    vcorrect += (preds == labels).sum().item()
                    vtotal += labels.size(0)
            val_loss = vloss / max(1, vtotal)
            val_acc = vcorrect / max(1, vtotal) if vtotal > 0 else 0.0
            scheduler.step(val_loss)

            # save best
            if save_path and val_acc > best_acc:
                best_acc = val_acc
                best_path = save_path
                # save model state and classes
                torch.save({'model_state_dict': model.state_dict(), 'classes': classes}, save_path)
                # report end-of-epoch with validation results
                if progress_callback:
                    try:
                        progress_callback(epoch, num_batches, num_batches, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
                    except Exception:
                        pass

        # after training set model and classes
        self.model = model.to(self.device)
        if best_path is None and save_path:
            # save final
            torch.save({'model_state_dict': model.state_dict(), 'classes': classes}, save_path)
            best_path = save_path
        return best_acc

    def loop_train(self, data_dir, param_grid, save_path_prefix=None, epochs=10, batch_size=32,
                   lr=1e-3, val_split=0.2, num_workers=4, img_size=IMG_SIZE,
                   model_type='resnet18', pretrained=True):
        """
        param_grid: dict of param name -> list of values to try. Supported params:
            model_type, num_conv, base_channels, batch_size, lr, img_size
        This will try the Cartesian product and return best (save_path, params, acc).
        """
        from itertools import product
        # build list of keys and value lists
        keys = list(param_grid.keys())
        vals = [param_grid[k] for k in keys]
        best_acc = -1.0
        best_info = None
        for comb in product(*vals):
            params = dict(zip(keys, comb))
            # merge defaults
            mt = params.get('model_type', model_type)
            nc = params.get('num_conv', 3)
            bc = params.get('base_channels', 32)
            bs = params.get('batch_size', batch_size)
            lr_p = params.get('lr', lr)
            isz = params.get('img_size', img_size)
            # make save path
            name_parts = [f"{k}={v}" for k, v in params.items()]
            save_path = None
            if save_path_prefix:
                safe_name = "__".join(name_parts)
                save_path = f"{save_path_prefix}__{safe_name}.pth"
            acc = self.train(data_dir, save_path=save_path, epochs=epochs, batch_size=bs, lr=lr_p,
                             val_split=val_split, num_workers=num_workers, img_size=isz,
                             model_type=mt, pretrained=pretrained, num_conv=nc, base_channels=bc)
            if acc > best_acc:
                best_acc = acc
                best_info = (save_path, params, acc)
        return best_info

    def load(self, path):
        if torch is None:
            return False
        if not os.path.exists(path):
            return False
        data = torch.load(path, map_location=self.device)
        classes = data.get('classes', None)
        if classes is None:
            return False
        self.classes = list(classes)
        model = self._build_model(len(self.classes))
        model.load_state_dict(data['model_state_dict'])
        model.to(self.device)
        model.eval()
        self.model = model
        return True

    def predict(self, bgr_crop):
        if self.model is None:
            return "unknown", 0.0
        # convert bgr crop (numpy) to PIL RGB
        try:
            img = Image.fromarray(cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB))
        except Exception:
            return "unknown", 0.0
        tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        x = tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            return self.classes[idx], conf