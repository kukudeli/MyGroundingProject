import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlatformPrototypeRebalanceLoss(nn.Module):
    """Training-only platform-conditioned prototype regularizer."""

    def __init__(
        self,
        in_dim=288,
        proto_dim=128,
        num_platforms=3,
        num_classes=6,
        momentum=0.9,
        temperature=0.07,
        gap_threshold=0.05,
        pce_weight=0.1,
        per_weight=0.01,
        warmup_epoch=5,
        use_pce=True,
        use_per=True,
    ):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, proto_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proto_dim, proto_dim),
        )
        self.num_platforms = num_platforms
        self.num_classes = num_classes
        self.momentum = momentum
        self.temperature = temperature
        self.gap_threshold = gap_threshold
        self.pce_weight = pce_weight
        self.per_weight = per_weight
        self.warmup_epoch = warmup_epoch
        self.use_pce = use_pce
        self.use_per = use_per

        self.register_buffer("prototypes", torch.zeros(num_platforms, num_classes, proto_dim))
        self.register_buffer("prototype_initialized", torch.zeros(num_platforms, num_classes, dtype=torch.bool))

    def _zero(self, features):
        return features.sum() * 0.0

    @torch.no_grad()
    def _update_prototypes(self, z, platform_labels, class_labels):
        for platform in platform_labels.unique():
            platform_idx = int(platform.item())
            if platform_idx < 0 or platform_idx >= self.num_platforms:
                continue
            platform_mask = platform_labels == platform
            for cls in class_labels[platform_mask].unique():
                cls_idx = int(cls.item())
                if cls_idx < 0 or cls_idx >= self.num_classes:
                    continue
                mask = platform_mask & (class_labels == cls)
                proto = F.normalize(z[mask].mean(dim=0), p=2, dim=0)
                if self.prototype_initialized[platform_idx, cls_idx]:
                    old_proto = self.prototypes[platform_idx, cls_idx]
                    proto = F.normalize(self.momentum * old_proto + (1.0 - self.momentum) * proto, p=2, dim=0)
                self.prototypes[platform_idx, cls_idx] = proto
                self.prototype_initialized[platform_idx, cls_idx] = True

    def forward(self, features, platform_labels, class_labels, epoch=None):
        device = features.device
        if features.numel() == 0:
            zero = self._zero(features)
            return zero, self._stats(zero, zero, zero, -1, -1, False)

        platform_labels = platform_labels.to(device=device, dtype=torch.long)
        class_labels = class_labels.to(device=device, dtype=torch.long)
        valid = (
            (platform_labels >= 0)
            & (platform_labels < self.num_platforms)
            & (class_labels >= 0)
            & (class_labels < self.num_classes)
        )
        if not valid.any():
            zero = self._zero(features)
            return zero, self._stats(zero, zero, zero, -1, -1, False)

        features = features[valid]
        platform_labels = platform_labels[valid]
        class_labels = class_labels[valid]

        z = F.normalize(self.projector(features), p=2, dim=-1)
        self._update_prototypes(z.detach(), platform_labels, class_labels)

        proto_bank = F.normalize(self.prototypes.to(device), p=2, dim=-1)
        sample_proto = proto_bank[platform_labels]
        logits = torch.bmm(sample_proto, z.unsqueeze(-1)).squeeze(-1) / self.temperature

        initialized = self.prototype_initialized.to(device)[platform_labels]
        logits = logits.masked_fill(~initialized, -1e4)
        target_initialized = initialized.gather(1, class_labels.unsqueeze(1)).squeeze(1)
        if target_initialized.any() and self.use_pce:
            loss_pce_raw = F.cross_entropy(logits[target_initialized], class_labels[target_initialized])
        else:
            loss_pce_raw = self._zero(features)
        loss_pce = loss_pce_raw * self.pce_weight

        probs = F.softmax(logits, dim=-1)
        true_probs = probs.gather(1, class_labels.unsqueeze(1)).squeeze(1)
        platform_scores = []
        present_platforms = []
        for platform_idx in platform_labels.unique():
            mask = platform_labels == platform_idx
            if mask.any():
                present_platforms.append(int(platform_idx.item()))
                platform_scores.append(true_probs[mask].mean())

        loss_per_raw = self._zero(features)
        platform_gap = self._zero(features)
        weak_platform = -1
        strong_platform = -1
        proto_active = False
        if len(platform_scores) > 1:
            scores = torch.stack(platform_scores)
            strong_idx = int(torch.argmax(scores).item())
            weak_idx = int(torch.argmin(scores).item())
            platform_gap = scores[strong_idx] - scores[weak_idx]
            strong_platform = present_platforms[strong_idx]
            weak_platform = present_platforms[weak_idx]
            current_epoch = -1 if epoch is None else int(epoch)
            proto_active = bool(
                self.use_per
                and current_epoch > self.warmup_epoch
                and platform_gap.detach().item() > self.gap_threshold
            )
            if proto_active:
                strong_mask = platform_labels == strong_platform
                entropy = -(probs[strong_mask] * torch.log(probs[strong_mask].clamp_min(1e-8))).sum(dim=-1)
                loss_per_raw = -entropy.mean() / math.log(self.num_classes)

        loss_per = loss_per_raw * self.per_weight
        loss_proto = loss_pce + loss_per
        return loss_proto, self._stats(loss_pce, loss_per, platform_gap, weak_platform, strong_platform, proto_active)

    @staticmethod
    def _stats(loss_pce, loss_per, platform_gap, weak_platform, strong_platform, proto_active):
        return {
            "loss_pce": loss_pce.detach(),
            "loss_per": loss_per.detach(),
            "platform_gap": platform_gap.detach(),
            "proto_active": proto_active,
            "weak_platform": weak_platform,
            "strong_platform": strong_platform,
        }
