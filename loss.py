import math
import torch
import torch.nn as nn


def bbox_ciou(box1, box2, eps=1e-7):
    """
    CIoU between box1(N,4) and box2(N,4).  Both in (xc, yc, w, h).
    Returns shape (N,1).
    """
    # centre → corners
    b1_x1 = box1[:, 0:1] - box1[:, 2:3] / 2
    b1_y1 = box1[:, 1:2] - box1[:, 3:4] / 2
    b1_x2 = box1[:, 0:1] + box1[:, 2:3] / 2
    b1_y2 = box1[:, 1:2] + box1[:, 3:4] / 2

    b2_x1 = box2[:, 0:1] - box2[:, 2:3] / 2
    b2_y1 = box2[:, 1:2] - box2[:, 3:4] / 2
    b2_x2 = box2[:, 0:1] + box2[:, 2:3] / 2
    b2_y2 = box2[:, 1:2] + box2[:, 3:4] / 2

    w1, h1 = box1[:, 2:3], box1[:, 3:4]
    w2, h2 = box2[:, 2:3], box2[:, 3:4]

    # intersection & union
    inter = ((torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) *
             (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0))
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # enclosing-box diagonal²
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps

    # centre-distance²
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4

    # aspect-ratio penalty
    v = (4 / math.pi ** 2) * (
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)

    return iou - rho2 / c2 - alpha * v


class ComputeLoss:
    """YOLOv5 loss – CIoU box + BCE obj + BCE cls."""

    def __init__(self, model):
        device = next(model.parameters()).device
        h = model.hyp

        self.BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['cls_pw']], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h['obj_pw']], device=device))

        m = model.detect
        self.balance = [4.0, 1.0, 0.4]          # P3-P5
        self.hyp = h
        self.na = m.na
        self.nc = m.nc
        self.nl = m.nl
        self.anchors = m.anchors
        self.device = device

    # ------------------------------------------------------------------ #
    def __call__(self, p, targets):
        lcls = torch.zeros(1, device=self.device)
        lbox = torch.zeros(1, device=self.device)
        lobj = torch.zeros(1, device=self.device)

        tcls, tbox, indices, anch = self.build_targets(p, targets)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype,
                               device=self.device)
            n = b.shape[0]

            if n:
                ps = pi[b, a, gj, gi]                  # (n, 5+nc)
                pxy, pwh, _, pcls = ps.split((2, 2, 1, self.nc), 1)

                # decode
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anch[i]
                pbox = torch.cat((pxy, pwh), 1)

                # box loss
                iou = bbox_ciou(pbox, tbox[i]).squeeze()
                lbox += (1.0 - iou).mean()

                # objectness target = detached iou
                tobj[b, a, gj, gi] = iou.detach().clamp(0)

                # classification
                if self.nc > 1:
                    t = torch.zeros_like(pcls)
                    t[range(n), tcls[i]] = 1.0
                    lcls += self.BCEcls(pcls, t)

            lobj += self.BCEobj(pi[..., 4], tobj) * self.balance[i]

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = p[0].shape[0]

        return ((lbox + lobj + lcls) * bs,
                torch.cat((lbox, lobj, lcls)).detach())

    # ------------------------------------------------------------------ #
    def build_targets(self, p, targets):
        """Assign each gt to anchor(s) + grid cell(s) for every layer."""
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []

        gain = torch.ones(7, device=self.device)
        # anchor indices column: shape (na, nt, 1)
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        g = 0.5
        off = (torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
                            device=self.device).float() * g)

        for i in range(self.nl):
            anchors_i = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]

            t = targets * gain                          # to grid space
            if nt:
                # ratio filter
                r = t[..., 4:6] / anchors_i[:, None]
                keep = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']
                t = t[keep]

                # neighbour offsets
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                mask = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[mask]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[mask]
            else:
                t = targets[0]
                offsets = 0

            bc, gxy, gwh, a = t.chunk(4, 1)            # split columns
            a, (b, c) = a.long().view(-1), bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            indices.append((
                b, a,
                gj.clamp_(0, int(gain[3]) - 1),
                gi.clamp_(0, int(gain[2]) - 1),
            ))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors_i[a])
            tcls.append(c)

        return tcls, tbox, indices, anch
