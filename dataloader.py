import os
import glob
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


IMG_FORMATS = ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp')


def img2label_paths(img_paths):
    """Replace /images/ with /labels/ and extension with .txt"""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def letterbox(im, new_shape=640, color=(114, 114, 114)):
    """Resize image preserving aspect ratio, pad remainder with grey"""
    h0, w0 = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))

    dw, dh = (new_shape[1] - new_w) / 2, (new_shape[0] - new_h) / 2

    if (w0, h0) != (new_w, new_h):
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def create_dataloader(path, imgsz, batch_size, workers=8, shuffle=False):
    dataset = YOLODataset(path, imgsz)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=YOLODataset.collate_fn,
    )
    return loader, dataset


class YOLODataset(Dataset):
    """Minimal YOLO dataset – labels stay in normalised xywh throughout."""

    def __init__(self, path, img_size=640):
        self.img_size = img_size

        p = Path(path)
        if p.is_dir():
            files = glob.glob(str(p / '**' / '*.*'), recursive=True)
        elif p.is_file():
            with open(p) as f:
                files = [x.strip() for x in f.read().strip().splitlines()]
        else:
            raise FileNotFoundError(f'{path} does not exist')

        self.im_files = sorted(x for x in files
                               if x.split('.')[-1].lower() in IMG_FORMATS)
        self.label_files = img2label_paths(self.im_files)

        # pre-load every label file once
        self.labels = []
        for lf in self.label_files:
            if os.path.isfile(lf):
                with open(lf) as f:
                    rows = [x.split() for x in f.read().strip().splitlines()]
                    lb = (np.array(rows, dtype=np.float32)
                          if rows else np.zeros((0, 5), dtype=np.float32))
            else:
                lb = np.zeros((0, 5), dtype=np.float32)
            self.labels.append(lb)

        print(f'Loaded {len(self.im_files)} images from {path}')

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        # --- image ---
        img = cv2.imread(self.im_files[index])
        assert img is not None, f'Image not found: {self.im_files[index]}'
        h0, w0 = img.shape[:2]

        img, r, (pad_w, pad_h) = letterbox(img, self.img_size)
        h, w = img.shape[:2]

        # --- labels  [cls, xc, yc, bw, bh]  all normalised ---
        labels = self.labels[index].copy()
        nl = len(labels)
        if nl:
            # map normalised-to-original  →  normalised-to-letterboxed
            labels[:, 1] = (labels[:, 1] * w0 * r + pad_w) / w   # xc
            labels[:, 2] = (labels[:, 2] * h0 * r + pad_h) / h   # yc
            labels[:, 3] = labels[:, 3] * w0 * r / w              # bw
            labels[:, 4] = labels[:, 4] * h0 * r / h              # bh

        # --- HWC-BGR  →  CHW-RGB  →  float [0,1] ---
        img = img.transpose(2, 0, 1)[::-1]
        img = np.ascontiguousarray(img)

        # [placeholder-batch-idx, cls, xc, yc, bw, bh]
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        return (torch.from_numpy(img).float() / 255.0,
                labels_out,
                self.im_files[index],
                (h0, w0))

    @staticmethod
    def collate_fn(batch):
        imgs, labels, paths, shapes = zip(*batch)
        for i, lb in enumerate(labels):
            lb[:, 0] = i                       # write batch index
        return torch.stack(imgs, 0), torch.cat(labels, 0), paths, shapes
