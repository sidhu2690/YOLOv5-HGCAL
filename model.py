import torch
import torch.nn as nn
import math


def autopad(k, p=None):
    """Calculate padding for 'same' convolution"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_divisible(x, divisor=8):
    """Make x divisible by divisor"""
    return int(math.ceil(x / divisor) * divisor)


# ======================== Building Blocks ========================

class Conv(nn.Module):
    """Standard convolution: Conv2d + BatchNorm + SiLU"""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck with optional shortcut"""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


# ======================== Detect Head ========================

class Detect(nn.Module):
    """YOLOv5 Detect head"""
    stride = torch.tensor([8., 16., 32.])

    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.empty(0)] * self.nl
        self.anchor_grid = [torch.empty(0)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                xy = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]
                wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat((xy, wh, y[..., 4:]), -1)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


# ======================== YOLOv5 Model ========================

class YOLOv5(nn.Module):
    """YOLOv5 Object Detection Model"""

    def __init__(self, nc=1, in_ch=3, depth_multiple=0.33, width_multiple=0.50, anchors=None):
        super().__init__()

        if anchors is None:
            anchors = [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326]
            ]

        # Scaling functions
        w = lambda x: make_divisible(x * width_multiple, 8)
        d = lambda x: max(round(x * depth_multiple), 1)

        # ==================== Backbone ====================
        self.stem = Conv(in_ch, w(64), 6, 2, 2)
        self.stage1 = nn.Sequential(Conv(w(64), w(128), 3, 2), C3(w(128), w(128), d(3)))
        self.stage2 = nn.Sequential(Conv(w(128), w(256), 3, 2), C3(w(256), w(256), d(6)))
        self.stage3 = nn.Sequential(Conv(w(256), w(512), 3, 2), C3(w(512), w(512), d(9)))
        self.stage4 = nn.Sequential(Conv(w(512), w(1024), 3, 2), C3(w(1024), w(1024), d(3)), SPPF(w(1024), w(1024), 5))

        # ==================== Head (FPN + PAN) ====================
        # Top-down (FPN)
        self.up_conv1 = Conv(w(1024), w(512), 1, 1)
        self.up_c3_1 = C3(w(1024), w(512), d(3), shortcut=False)

        self.up_conv2 = Conv(w(512), w(256), 1, 1)
        self.up_c3_2 = C3(w(512), w(256), d(3), shortcut=False)

        # Bottom-up (PAN)
        self.down_conv1 = Conv(w(256), w(256), 3, 2)
        self.down_c3_1 = C3(w(512), w(512), d(3), shortcut=False)

        self.down_conv2 = Conv(w(512), w(512), 3, 2)
        self.down_c3_2 = C3(w(1024), w(1024), d(3), shortcut=False)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # ==================== Detect ====================
        self.detect = Detect(nc, anchors, [w(256), w(512), w(1024)])

    def forward(self, x):
        # Backbone
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)   # P3/8
        p4 = self.stage3(p3)  # P4/16
        p5 = self.stage4(p4)  # P5/32

        # FPN (top-down)
        x10 = self.up_conv1(p5)
        x = self.up_c3_1(torch.cat([self.upsample(x10), p4], 1))

        x14 = self.up_conv2(x)
        n3 = self.up_c3_2(torch.cat([self.upsample(x14), p3], 1))  # P3 output

        # PAN (bottom-up)
        n4 = self.down_c3_1(torch.cat([self.down_conv1(n3), x14], 1))  # P4 output
        n5 = self.down_c3_2(torch.cat([self.down_conv2(n4), x10], 1))  # P5 output


        return self.detect([n3, n4, n5])
