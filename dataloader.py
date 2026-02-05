import os
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ======================== Parameters ========================
IMG_FORMATS = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp', 'npy')


# ======================== Utility Functions ========================

def img2label_paths(img_paths):
    """Convert image paths to label paths"""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def xywhn2xyxy(x, w=640, h=640):
    """Convert normalized [x, y, w, h] to pixel [x1, y1, x2, y2]"""
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2)  # x1
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2)  # y1
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2)  # x2
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2)  # y2
    return y


def xyxy2xywhn(x, w=640, h=640):
    """Convert pixel [x1, y1, x2, y2] to normalized [x, y, w, h]"""
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w        # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h        # height
    return y


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize and pad image to new_shape"""
    shape = im.shape[:2]  # [height, width]
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    # Padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return im, r, (dw, dh)


# ======================== Dataloader ========================

def create_dataloader(path, imgsz, batch_size, stride=32, hyp=None, 
                      augment=False, workers=8, shuffle=False):
    """Create dataloader"""
    dataset = YOLODataset(path, imgsz, augment=augment, hyp=hyp)
    
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=YOLODataset.collate_fn
    )
    
    return loader, dataset


# ======================== Dataset ========================

class YOLODataset(Dataset):
    """Simple YOLO Dataset"""
    
    def __init__(self, path, img_size=640, augment=False, hyp=None):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp or {}
        
        # Find images
        self.im_files = self._get_image_files(path)
        self.label_files = img2label_paths(self.im_files)
        
        # Load all labels
        self.labels = [self._load_label(f) for f in self.label_files]
        
        self.n = len(self.im_files)
        print(f'Loaded {self.n} images from {path}')
    
    def _get_image_files(self, path):
        """Get list of image files"""
        p = Path(path)
        if p.is_dir():
            files = glob.glob(str(p / '**' / '*.*'), recursive=True)
        elif p.is_file():
            with open(p) as f:
                files = [x.strip() for x in f.read().strip().splitlines()]
        else:
            raise FileNotFoundError(f'{path} does not exist')
        
        return sorted(x for x in files if x.split('.')[-1].lower() in IMG_FORMATS)
    
    def _load_label(self, label_path):
        """Load labels from txt file"""
        if os.path.isfile(label_path):
            with open(label_path) as f:
                labels = [x.split() for x in f.read().strip().splitlines()]
                return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
        return np.zeros((0, 5), dtype=np.float32)
    
    def _load_image(self, index):
        """Load image from file"""
        path = self.im_files[index]
        
        if path.endswith('.npy'):
            im = np.load(path)
        else:
            im = cv2.imread(path)  # BGR
        
        assert im is not None, f'Image not found: {path}'
        return im
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        # Load image
        img = self._load_image(index)
        h0, w0 = img.shape[:2]
        
        # Resize
        img, ratio, pad = letterbox(img, self.img_size)
        h, w = img.shape[:2]
        
        # Load labels
        labels = self.labels[index].copy()
        nl = len(labels)
        
        if nl:
            # Convert normalized xywh to pixel xyxy
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w0, h0)
            
            # Scale to new image size
            labels[:, 1] = labels[:, 1] * ratio + pad[0]  # x1
            labels[:, 2] = labels[:, 2] * ratio + pad[1]  # y1
            labels[:, 3] = labels[:, 3] * ratio + pad[0]  # x2
            labels[:, 4] = labels[:, 4] * ratio + pad[1]  # y2
            
            # Convert back to normalized xywh
            labels[:, 1:] = xyxy2xywhn(labels[:, 1:], w, h)
        
        # Simple augmentation: horizontal flip
        if self.augment and random.random() < self.hyp.get('fliplr', 0.5):
            img = np.fliplr(img)
            if nl:
                labels[:, 1] = 1 - labels[:, 1]  # flip x_center
        
        # Convert image: HWC to CHW, BGR to RGB
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        
        # Create labels tensor: [class, x, y, w, h] -> [batch_idx, class, x, y, w, h]
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        
        return torch.from_numpy(img).float() / 255.0, labels_out, self.im_files[index], (h0, w0)
    
    @staticmethod
    def collate_fn(batch):
        """Collate batch of samples"""
        imgs, labels, paths, shapes = zip(*batch)
        
        # Add image index to labels
        for i, lb in enumerate(labels):
            lb[:, 0] = i
        
        return torch.stack(imgs, 0), torch.cat(labels, 0), paths, shapes
