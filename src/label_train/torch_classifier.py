import os
import json
import numpy as np
from pathlib import Path
import cv2
import threading
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
import time
# optional plotting
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


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
        self.last_misclassified = []  # filled by train(): [{path, true, pred, conf}, ...]

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
              val_split=0.2, test_split=0.2, num_workers=4, img_size=IMG_SIZE,
              model_type='resnet18', pretrained=True,
              num_conv=3, base_channels=32, device=None,
              progress_callback=None,
              # Performance / behavior knobs:
              validate_during_training=True,
              pin_memory=True,
              persistent_workers=False,
              non_blocking=True,
              use_class_weights=True,
              lr_scheduler='cosine'):   # 'cosine' | 'plateau' | 'none'
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

        # Per-class split by base image id (prevent leakage while maintaining per-class ratios)
        # Windows DataLoader fix: spawning new worker processes every epoch causes 10-30s stalls.
        # Force persistent_workers=True when num_workers>0 so workers stay alive across epochs.
        # Alternatively, fall back to num_workers=0 if persistent_workers is explicitly disabled.
        import sys as _sys
        if _sys.platform == 'win32' and num_workers > 0 and not persistent_workers:
            persistent_workers = True
            try:
                print('[TRAIN] Windows detected: auto-enabled persistent_workers=True to avoid per-epoch spawn stalls.', flush=True)
            except Exception:
                pass

        if val_split + test_split >= 1.0:
            raise ValueError("val_split + test_split must be < 1.0")
        rng = np.random.default_rng(12345)
        train_items = []
        test_items = []
        val_items = []
        # For each class, group its items by base id and split the base ids
        for cls_idx in range(len(classes)):
            # collect indices of items belonging to this class
            cls_indices = [i for i, (_p, lbl) in enumerate(items) if lbl == cls_idx]
            if not cls_indices:
                continue
            # compute base id for each item (filename prefix before underscore)
            base_map = {}
            for i in cls_indices:
                p = items[i][0]
                name = os.path.basename(p)
                stem = os.path.splitext(name)[0]
                base = stem.split("_")[0]
                base_map.setdefault(base, []).append(i)
            bases = list(base_map.keys())
            rng.shuffle(bases)
            nb = len(bases)
            n_test_b = int(round(nb * test_split))
            n_val_b = int(round(nb * val_split))
            n_train_b = nb - n_test_b - n_val_b
            if n_train_b < 0:
                n_train_b = 0
            train_bases = set(bases[:n_train_b])
            test_bases = set(bases[n_train_b:n_train_b + n_test_b])
            val_bases = set(bases[n_train_b + n_test_b:n_train_b + n_test_b + n_val_b])
            # map back to items
            for b in train_bases:
                train_items.extend([items[i] for i in base_map[b]])
            for b in test_bases:
                test_items.extend([items[i] for i in base_map[b]])
            for b in val_bases:
                val_items.extend([items[i] for i in base_map[b]])
        # fallback to global split if per-class grouping failed to produce train items
        if len(train_items) == 0:
            idxs = np.arange(len(items))
            rng.shuffle(idxs)
            cut1 = int(len(items) * (1 - (val_split + test_split)))
            cut2 = int(len(items) * (1 - val_split))
            train_items = [items[i] for i in idxs[:cut1]]
            test_items = [items[i] for i in idxs[cut1:cut2]]
            val_items = [items[i] for i in idxs[cut2:]]

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
        test_ds = _ImageDataset(test_items, transform=val_tf)
        val_ds = _ImageDataset(val_items, transform=val_tf) if val_items else None
        # DataLoader performance options: pin_memory helpful when using CUDA, persistent_workers speeds up repeated epochs
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                     num_workers=max(1, num_workers//2), pin_memory=pin_memory, persistent_workers=persistent_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                    num_workers=max(1, num_workers//2), pin_memory=pin_memory, persistent_workers=persistent_workers) if val_ds is not None else None

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
        # Build LR scheduler:
        #   'cosine'  -> CosineAnnealingLR: smoothly decays LR to near-zero, steps every epoch.
        #                Works regardless of validate_during_training. Recommended default.
        #   'plateau' -> ReduceLROnPlateau: halves LR when val_loss stagnates.
        #                Only effective when validate_during_training=True.
        #   'none'    -> Constant LR throughout training.
        _lr_sched_type = str(lr_scheduler).lower()
        if _lr_sched_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-3)
        elif _lr_sched_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        else:
            scheduler = None
        # Optionally compute class weights from training set to mitigate imbalance
        if use_class_weights:
            labels_arr = [lbl for (_, lbl) in train_items]
            counts = np.bincount(labels_arr, minlength=len(classes)).astype(np.float32)
            inv = 1.0 / (counts + 1e-6)
            weights = torch.tensor(inv / inv.sum(), dtype=torch.float32).to(self.device)
            try:
                criterion = nn.CrossEntropyLoss(weight=weights)
            except Exception:
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # Diagnostic: print device info
        try:
            print(f"[TRAIN] device={self.device}  cuda_available={torch.cuda.is_available()}", flush=True)
            # sample parameter device
            p = next(model.parameters())
            print(f"[TRAIN] model parameters on device: {p.device}", flush=True)
        except Exception:
            pass

        best_acc = 0.0
        best_path = None
        train_acc_history = []
        test_acc_history = []
        val_acc_history = []
        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = total = 0
            num_batches = len(train_loader)
            for batch_idx, (imgs, labels) in enumerate(train_loader):
                imgs = imgs.to(self.device, non_blocking=non_blocking)
                labels = labels.to(self.device, non_blocking=non_blocking)
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
            train_acc_history.append(train_acc)

            # compute test accuracy each epoch (user requested per-epoch test acc)
            test_acc = 0.0
            try:
                model.eval()
                t_vcorrect = t_vtotal = 0
                with torch.no_grad():
                    for imgs, labels in test_loader:
                        imgs = imgs.to(self.device, non_blocking=non_blocking)
                        labels = labels.to(self.device, non_blocking=non_blocking)
                        logits = model(imgs)
                        preds = logits.argmax(dim=1)
                        t_vcorrect += (preds == labels).sum().item()
                        t_vtotal += labels.size(0)
                test_acc = t_vcorrect / max(1, t_vtotal) if t_vtotal > 0 else 0.0
            except Exception:
                test_acc = 0.0
            test_acc_history.append(test_acc)
            # optionally run validation during training (slow) or skip and run only at end
            if validate_during_training and val_loader is not None:
                model.eval()
                if progress_callback:
                    try:
                        progress_callback(epoch, 0, num_batches, train_loss=train_loss, train_acc=train_acc)
                    except Exception:
                        pass
                try:
                    print(f"[TRAIN] epoch {epoch}: starting validation...", flush=True)
                except Exception:
                    pass
                t0 = time.time()
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    pass
                vloss = 0.0
                vcorrect = vtotal = 0
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs = imgs.to(self.device, non_blocking=non_blocking)
                        labels = labels.to(self.device, non_blocking=non_blocking)
                        logits = model(imgs)
                        loss = criterion(logits, labels)
                        vloss += loss.item() * imgs.size(0)
                        preds = logits.argmax(dim=1)
                        vcorrect += (preds == labels).sum().item()
                        vtotal += labels.size(0)
                val_loss = vloss / max(1, vtotal)
                val_acc = vcorrect / max(1, vtotal) if vtotal > 0 else 0.0
                try:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    pass
                t1 = time.time()
                try:
                    print(f"[TRAIN] epoch {epoch}: validation time {t1 - t0:.3f}s  val_acc={val_acc:.4f}", flush=True)
                except Exception:
                    pass
                scheduler.step(val_loss)
                val_acc_history.append(val_acc)
            else:
                val_acc_history.append(float('nan'))

            # Step non-plateau schedulers every epoch (no val_loss needed)
            if scheduler is not None and _lr_sched_type != 'plateau':
                scheduler.step()
            # save best (perform heavy disk I/O asynchronously)
            score_for_best = val_acc if (validate_during_training and val_loader is not None) else test_acc
            if save_path and score_for_best > best_acc:
                best_acc = score_for_best
                best_path = save_path
                # prepare CPU state_dict to avoid GPU<->CPU sync overhead in the saver thread
                # capture a reference to state_dict quickly (avoid heavy .cpu() here)
                state_ref = model.state_dict()

                def _save_async(path, state_ref_inner, classes_list):
                    try:
                        cpu_state_inner = {k: v.cpu() for k, v in state_ref_inner.items()}
                    except Exception:
                        cpu_state_inner = {k: v for k, v in state_ref_inner.items()}
                    try:
                        torch.save({'model_state_dict': cpu_state_inner, 'classes': classes_list}, path)
                    except Exception:
                        try:
                            torch.save({'model_state_dict': cpu_state_inner}, path)
                        except Exception:
                            pass

                t = threading.Thread(target=_save_async, args=(save_path, state_ref, classes), daemon=True)
                t.start()

            # Always report end-of-epoch (regardless of whether it's a new best)
            if progress_callback:
                try:
                    # expose test_acc in the val_acc slot for compatibility
                    progress_callback(epoch, num_batches, num_batches, train_loss=train_loss, train_acc=train_acc, val_loss=None, val_acc=test_acc)
                except Exception:
                    pass

        # If validation was postponed until after training, run it once now
        if not validate_during_training:
            try:
                print("[TRAIN] running final validation after training...", flush=True)
                model.eval()
                vloss = 0.0
                vcorrect = vtotal = 0
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs = imgs.to(self.device, non_blocking=non_blocking)
                        labels = labels.to(self.device, non_blocking=non_blocking)
                        logits = model(imgs)
                        loss = criterion(logits, labels)
                        vloss += loss.item() * imgs.size(0)
                        preds = logits.argmax(dim=1)
                        vcorrect += (preds == labels).sum().item()
                        vtotal += labels.size(0)
                val_loss = vloss / max(1, vtotal)
                val_acc = vcorrect / max(1, vtotal) if vtotal > 0 else 0.0
                print(f"[TRAIN] final validation: val_acc={val_acc:.4f}", flush=True)
            except Exception:
                pass
            # record final val acc for plotting
            # if we have been appending NaNs, replace last entry
            if val_acc_history:
                val_acc_history[-1] = val_acc
            else:
                val_acc_history = [float('nan')] * (len(train_acc_history) - 1) + [val_acc]

        # Plot train/test accuracy curves if matplotlib is available
        try:
            if plt is not None:
                epochs_x = list(range(1, len(train_acc_history) + 1))
                plt.figure(figsize=(8, 4))
                plt.plot(epochs_x, [a * 100 for a in train_acc_history], label='train_acc')
                plt.plot(epochs_x, [a * 100 for a in test_acc_history], label='test_acc')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.title('Training vs Validation Accuracy')
                plt.grid(True)
                plt.legend()
                # save plot next to model save path if provided
                if save_path:
                    plot_path = save_path + '.acc.png'
                else:
                    plot_path = os.path.join(os.getcwd(), 'training_acc.png')
                try:
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    print(f"[TRAIN] Saved accuracy plot: {plot_path}")
                    # annotate final val acc if computed
                    try:
                        if not validate_during_training and val_acc is not None:
                            plt.figure(figsize=(6,1))
                            plt.text(0.01, 0.5, f'Final val_acc: {val_acc*100:.2f}%')
                    except Exception:
                        pass
                    try:
                        plt.show(block=False)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[TRAIN] Failed to save/show accuracy plot: {e}")
                finally:
                    plt.close()
            else:
                print('[TRAIN] matplotlib not available; skipping accuracy plot')
        except Exception:
            pass

        # after training set model and classes
        self.model = model.to(self.device)
        if best_path is None and save_path:
            # final save asynchronously to avoid blocking caller; copy to CPU in thread
            state_ref_final = model.state_dict()

            def _final_save(path, state_ref_inner, classes_list):
                try:
                    cpu_state_inner = {k: v.cpu() for k, v in state_ref_inner.items()}
                except Exception:
                    cpu_state_inner = {k: v for k, v in state_ref_inner.items()}
                try:
                    torch.save({'model_state_dict': cpu_state_inner, 'classes': classes_list}, path)
                except Exception:
                    try:
                        torch.save({'model_state_dict': cpu_state_inner}, path)
                    except Exception:
                        pass

            t_final = threading.Thread(target=_final_save, args=(save_path, state_ref_final, classes), daemon=True)
            t_final.start()
            best_path = save_path

        # Collect misclassified test-set items so the UI can show them for annotation review
        self.last_misclassified = []
        try:
            model.eval()
            with torch.no_grad():
                for (fpath, true_idx) in test_items:
                    try:
                        from PIL import Image as _PILI
                        _img = _PILI.open(fpath).convert('RGB')
                        _t = val_tf(_img).unsqueeze(0).to(self.device)
                        _probs = torch.softmax(model(_t), dim=1)[0]
                        _pred = int(_probs.argmax().item())
                        _conf = float(_probs[_pred].item())
                        if _pred != true_idx:
                            self.last_misclassified.append({
                                'path': str(fpath),
                                'true': classes[true_idx],
                                'pred': classes[_pred],
                                'conf': _conf,
                            })
                    except Exception:
                        pass
            print(f"[TRAIN] misclassified on test set: {len(self.last_misclassified)}/{len(test_items)}", flush=True)
        except Exception as _e:
            print(f"[TRAIN] misclassified collection error: {_e}", flush=True)

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