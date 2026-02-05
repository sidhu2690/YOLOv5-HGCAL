"""
Simplified YOLOv5 Dataloader
Only includes components needed for training
"""

import os
import glob
import hashlib
import random
from pathlib import Path
from multiprocessing.pool import Pool
from itertools import repeat

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ======================== Parameters ========================
IMG_FORMATS = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp', 'npy')
NUM_THREADS = min(8, os.cpu_count())


# ======================== Utility Functions ========================

def get_hash(paths):
    """Returns a single hash value of a list of paths"""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    h = hashlib.md5(str(size).encode())
    h.update(''.join(paths).encode())
    return h.hexdigest()


def img2label_paths(img_paths):
    """Define label paths as a function of image paths"""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    """Convert normalized [x, y, w, h] to pixel [x1, y1, x2, y2]"""
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # x1
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # y1
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # x2
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # y2
    return y


def xyxy2xywhn(x, w=640, h=640, clip=True, eps=0.0):
    """Convert pixel [x1, y1, x2, y2] to normalized [x, y, w, h]"""
    if clip:
        x[:, 0] = x[:, 0].clip(0, w - eps)
        x[:, 1] = x[:, 1].clip(0, h - eps)
        x[:, 2] = x[:, 2].clip(0, w - eps)
        x[:, 3] = x[:, 3].clip(0, h - eps)
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints"""
    shape = im.shape[:2]  # current shape [height, width]
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return im, (r, r), (dw, dh)


def verify_image_label(args):
    """Verify one image-label pair"""
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []
    
    try:
        # Verify image
        if im_file.endswith('.npy'):
            im = np.load(im_file)
            shape = im.shape[:2]
        else:
            im = Image.open(im_file)
            im.verify()
            shape = im.size[::-1]  # PIL gives (width, height), we need (height, width)
        
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        
        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                lb = np.array(lb, dtype=np.float32)
            
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates'
                
                # Remove duplicates
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1
            lb = np.zeros((0, 5), dtype=np.float32)
        
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


# ======================== Augmentation Functions ========================

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    """HSV color-space augmentation"""
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype
        
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)


def random_perspective(im, targets=(), degrees=10, translate=0.1, scale=0.1, shear=10, border=(0, 0)):
    """Random perspective/affine transformation"""
    height = im.shape[0] + border[0] * 2
    width = im.shape[1] + border[1] * 2
    
    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2
    C[1, 2] = -im.shape[0] / 2
    
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    
    # Shear
    S = np.eye(3)
    S[0, 1] = np.tan(random.uniform(-shear, shear) * np.pi / 180)
    S[1, 0] = np.tan(random.uniform(-shear, shear) * np.pi / 180)
    
    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    
    # Combined transformation matrix
    M = T @ S @ R @ C
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
    
    # Transform labels
    n = len(targets)
    if n:
        # Warp boxes
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = xy @ M.T
        xy = xy[:, :2].reshape(n, 8)
        
        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        
        # Clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
        
        # Filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.1)
        targets = targets[i]
        targets[:, 1:5] = new[i]
    
    return im, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    """Filter box candidates by size and aspect ratio constraints"""
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def mixup(im, labels, im2, labels2):
    """MixUp augmentation: blend two images and labels"""
    r = np.random.beta(32.0, 32.0)
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


# ======================== Dataloader ========================

def create_dataloader(path, imgsz, batch_size, stride, hyp=None, augment=False,
                      cache=False, workers=8, shuffle=False, prefix=''):
    """Create dataloader for training/validation"""
    
    dataset = LoadImagesAndLabels(
        path=path,
        img_size=imgsz,
        batch_size=batch_size,
        augment=augment,
        hyp=hyp,
        stride=int(stride),
        cache_images=cache,
        prefix=prefix
    )
    
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn
    )
    
    return loader, dataset


# ======================== Dataset Class ========================

class LoadImagesAndLabels(Dataset):
    """YOLOv5 Dataset for loading images and labels"""
    
    cache_version = 0.6
    
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None,
                 stride=32, cache_images=False, prefix=''):
        
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp or {}
        self.stride = stride
        self.path = path
        self.mosaic = self.augment and self.hyp.get('mosaic', 1.0) > 0
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        
        # ==================== Find Images ====================
        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]
                else:
                    raise FileNotFoundError(f'{prefix}{p} does not exist')
            
            self.im_files = sorted(x.replace('/', os.sep) for x in f 
                                   if x.split('.')[-1].lower() in IMG_FORMATS)
            assert self.im_files, f'{prefix}No images found'
        
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}')
        
        # ==================== Load Labels ====================
        self.label_files = img2label_paths(self.im_files)
        
        # Check cache
        cache_path = (Path(p) if Path(p).is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True
            assert cache['version'] == self.cache_version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False
        
        # Read cache results
        nf, nm, ne, nc, n = cache.pop('results')
        if exists:
            print(f"{prefix}Scanning '{cache_path}'... {nf} found, {nm} missing, {ne} empty, {nc} corrupt")
        assert nf > 0 or not augment, f'{prefix}No labels found. Cannot train without labels.'
        
        # Load labels from cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.im_files = list(cache.keys())
        self.label_files = img2label_paths(cache.keys())
        
        self.n = len(self.shapes)
        self.indices = list(range(self.n))
        
        # ==================== Cache Images ====================
        self.ims = [None] * self.n
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        
        if cache_images:
            gb = 0
            pbar = tqdm(range(self.n), desc=f'{prefix}Caching images', total=self.n)
            for i in pbar:
                self.ims[i], _, _ = self.load_image(i)
                gb += self.ims[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
    
    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        """Cache dataset labels, check images and read shapes"""
        x = {}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(
                pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(prefix))),
                desc=desc,
                total=len(self.im_files)
            )
            
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x[im_file] = [lb, shape, segments]
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
        
        pbar.close()
        if msgs:
            print('\n'.join(msgs))
        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}')
        
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs
        x['version'] = self.cache_version
        
        try:
            np.save(path, x)
            path.with_suffix('.cache.npy').rename(path)
            print(f'{prefix}New cache created: {path}')
        except Exception as e:
            print(f'{prefix}WARNING: Cache directory {path.parent} is not writeable: {e}')
        
        return x
    
    def __len__(self):
        return len(self.im_files)
    
    def __getitem__(self, index):
        index = self.indices[index]
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp.get('mosaic', 1.0)
        
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None
            
            # MixUp augmentation
            if random.random() < hyp.get('mixup', 0.0):
                img, labels = mixup(img, labels, *self.load_mosaic(random.randint(0, self.n - 1)))
        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)
            shapes = (h0, w0), ((h / h0, w / w0), (0, 0))
            
            labels = self.labels[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, 0, 0)
        
        nl = len(labels)
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)
        
        if self.augment:
            # HSV color-space augmentation
            augment_hsv(img, hgain=hyp.get('hsv_h', 0.015), 
                        sgain=hyp.get('hsv_s', 0.7), vgain=hyp.get('hsv_v', 0.4))
            
            # Flip up-down
            if random.random() < hyp.get('flipud', 0.0):
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]
            
            # Flip left-right
            if random.random() < hyp.get('fliplr', 0.5):
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]
        
        # Create output labels tensor
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        
        # Convert image: HWC to CHW, BGR to RGB
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        
        return torch.from_numpy(img), labels_out, self.im_files[index], shapes
    
    def load_image(self, i):
        """Load 1 image from dataset index 'i'"""
        im = self.ims[i]
        if im is None:
            npy = self.npy_files[i]
            if npy.exists():
                im = np.load(npy)
            else:
                f = self.im_files[i]
                im = cv2.imread(f)
                assert im is not None, f'Image Not Found {f}'
            
            h0, w0 = im.shape[:2]
            r = self.img_size / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_LINEAR if self.augment else cv2.INTER_AREA
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]
        else:
            return im, self.shapes[i][:2].astype(int), self.shapes[i][:2].astype(int)
    
    def load_mosaic(self, index):
        """Load 4-image mosaic"""
        labels4 = []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)
        
        for i, idx in enumerate(indices):
            img, _, (h, w) = self.load_image(idx)
            
            # Place image in mosaic
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            
            # Labels
            labels = self.labels[idx].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)
            labels4.append(labels)
        
        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])
        
        # Random perspective transformation
        img4, labels4 = random_perspective(
            img4, labels4,
            degrees=self.hyp.get('degrees', 0.0),
            translate=self.hyp.get('translate', 0.1),
            scale=self.hyp.get('scale', 0.5),
            shear=self.hyp.get('shear', 0.0),
            border=self.mosaic_border
        )
        
        return img4, labels4
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader"""
        im, label, path, shapes = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes
