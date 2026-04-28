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
        self.register_buffer("global_prototypes", torch.zeros(num_classes, proto_dim))
        self.register_buffer("global_prototype_initialized", torch.zeros(num_classes, dtype=torch.bool))
        fallback = F.normalize(torch.randn(num_classes, proto_dim), p=2, dim=-1)
        self.register_buffer("fallback_prototypes", fallback)

    def _zero(self, features):
        return features.sum() * 0.0

    @torch.no_grad()
    def _update_prototypes(self, z, platform_labels, class_labels):
        for cls in class_labels.unique():
            cls_idx = int(cls.item())
            if cls_idx < 0 or cls_idx >= self.num_classes:
                continue
            mask = class_labels == cls
            proto = F.normalize(z[mask].mean(dim=0), p=2, dim=0)
            if self.global_prototype_initialized[cls_idx]:
                old_proto = self.global_prototypes[cls_idx]
            else:
                old_proto = self.fallback_prototypes[cls_idx]
            proto = F.normalize(self.momentum * old_proto + (1.0 - self.momentum) * proto, p=2, dim=0)
            self.global_prototypes[cls_idx] = proto
            self.global_prototype_initialized[cls_idx] = True

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
                else:
                    old_proto = self.fallback_prototypes[cls_idx]
                proto = F.normalize(self.momentum * old_proto + (1.0 - self.momentum) * proto, p=2, dim=0)
                self.prototypes[platform_idx, cls_idx] = proto
                self.prototype_initialized[platform_idx, cls_idx] = True

    def _build_sample_prototypes(self, platform_labels):
        device = platform_labels.device
        platform_proto = F.normalize(self.prototypes.to(device), p=2, dim=-1)
        global_proto = F.normalize(self.global_prototypes.to(device), p=2, dim=-1)
        fallback_proto = F.normalize(self.fallback_prototypes.to(device), p=2, dim=-1)

        platform_initialized = self.prototype_initialized.to(device)[platform_labels]
        global_initialized = self.global_prototype_initialized.to(device)

        sample_proto = platform_proto[platform_labels]
        global_proto = global_proto.unsqueeze(0).expand(platform_labels.shape[0], -1, -1)
        fallback_proto = fallback_proto.unsqueeze(0).expand(platform_labels.shape[0], -1, -1)

        sample_proto = torch.where(global_initialized.view(1, self.num_classes, 1), global_proto, fallback_proto)
        sample_proto = torch.where(platform_initialized.unsqueeze(-1), platform_proto[platform_labels], sample_proto)
        sample_proto = F.normalize(sample_proto, p=2, dim=-1)

        fallback_mask = ~platform_initialized & ~global_initialized.view(1, self.num_classes)
        return sample_proto, platform_initialized, global_initialized, fallback_mask

    def forward(self, features, platform_labels, class_labels, epoch=None):
        device = features.device
        if features.numel() == 0:
            zero = self._zero(features)
            return zero, self._stats(zero, zero, zero, -1, -1, False, False, False, 0, 0, 0, 0, 0)

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
            return zero, self._stats(zero, zero, zero, -1, -1, False, False, False, 0, 0, 0, 0, 0)

        features = features[valid]
        platform_labels = platform_labels[valid]
        class_labels = class_labels[valid]
        num_valid_samples = int(features.shape[0])
        num_active_platforms = int(platform_labels.unique().numel())

        z = F.normalize(self.projector(features), p=2, dim=-1)
        self._update_prototypes(z.detach(), platform_labels, class_labels)

        sample_proto, platform_initialized, global_initialized, fallback_mask = self._build_sample_prototypes(platform_labels)
        logits = torch.bmm(sample_proto, z.unsqueeze(-1)).squeeze(-1) / self.temperature
        logits = logits.clamp(min=-50.0, max=50.0)

        pce_active = bool(self.use_pce and num_valid_samples > 0)
        if pce_active:
            loss_pce_raw = F.cross_entropy(logits, class_labels)
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
        per_active = False
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
                per_active = True
                strong_mask = platform_labels == strong_platform
                entropy = -(probs[strong_mask] * torch.log(probs[strong_mask].clamp_min(1e-8))).sum(dim=-1)
                loss_per_raw = -entropy.mean() / math.log(self.num_classes)

        loss_per = loss_per_raw * self.per_weight
        loss_proto = loss_pce + loss_per
        valid_platform_proto_count = int(self.prototype_initialized.sum().item())
        valid_global_proto_count = int(self.global_prototype_initialized.sum().item())
        fallback_proto_count = int(fallback_mask.sum().item())
        return loss_proto, self._stats(
            loss_pce,
            loss_per,
            platform_gap,
            weak_platform,
            strong_platform,
            proto_active,
            pce_active,
            per_active,
            num_valid_samples,
            num_active_platforms,
            valid_platform_proto_count,
            valid_global_proto_count,
            fallback_proto_count,
        )

    @staticmethod
    def _stats(
        loss_pce,
        loss_per,
        platform_gap,
        weak_platform,
        strong_platform,
        proto_active,
        pce_active,
        per_active,
        num_valid_samples,
        num_active_platforms,
        valid_platform_proto_count,
        valid_global_proto_count,
        fallback_proto_count,
    ):
        device = loss_pce.device
        return {
            "loss_pce": loss_pce.detach(),
            "loss_per": loss_per.detach(),
            "platform_gap": platform_gap.detach(),
            "proto_active": torch.tensor(float(proto_active), device=device),
            "pce_active": torch.tensor(float(pce_active), device=device),
            "per_active": torch.tensor(float(per_active), device=device),
            "num_valid_samples": torch.tensor(float(num_valid_samples), device=device),
            "num_active_platforms": torch.tensor(float(num_active_platforms), device=device),
            "valid_platform_proto_count": torch.tensor(float(valid_platform_proto_count), device=device),
            "valid_global_proto_count": torch.tensor(float(valid_global_proto_count), device=device),
            "fallback_proto_count": torch.tensor(float(fallback_proto_count), device=device),
            "weak_platform": torch.tensor(float(weak_platform), device=device),
            "strong_platform": torch.tensor(float(strong_platform), device=device),
        }
