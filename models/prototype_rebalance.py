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
        score_momentum=0.9,
        min_platform_samples=1,
        min_platform_seen=5,
        weak_pce_boost=1.0,
        max_pce_boost=2.0,
        status_mode="box_difficulty",
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
        self.score_momentum = score_momentum
        self.min_platform_samples = min_platform_samples
        self.min_platform_seen = min_platform_seen
        self.weak_pce_boost = weak_pce_boost
        self.max_pce_boost = max_pce_boost
        self.status_mode = status_mode

        self.register_buffer("prototypes", torch.zeros(num_platforms, num_classes, proto_dim))
        self.register_buffer("prototype_initialized", torch.zeros(num_platforms, num_classes, dtype=torch.bool))
        self.register_buffer("global_prototypes", torch.zeros(num_classes, proto_dim))
        self.register_buffer("global_prototype_initialized", torch.zeros(num_classes, dtype=torch.bool))
        fallback = F.normalize(torch.randn(num_classes, proto_dim), p=2, dim=-1)
        self.register_buffer("fallback_prototypes", fallback)
        self.register_buffer("platform_score_ema", torch.zeros(num_platforms))
        self.register_buffer("platform_score_initialized", torch.zeros(num_platforms, dtype=torch.bool))
        self.register_buffer("platform_seen_count", torch.zeros(num_platforms))

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

    @torch.no_grad()
    def _update_platform_status(self, status_values, platform_labels):
        device = status_values.device
        batch_scores = torch.zeros(self.num_platforms, device=device)
        batch_valid = torch.zeros(self.num_platforms, dtype=torch.bool, device=device)
        status_values = status_values.detach()
        for platform in platform_labels.unique():
            platform_idx = int(platform.item())
            if platform_idx < 0 or platform_idx >= self.num_platforms:
                continue
            mask = platform_labels == platform_idx
            num_samples = int(mask.sum().item())
            if num_samples < self.min_platform_samples:
                continue
            batch_score = status_values[mask].mean()
            batch_scores[platform_idx] = batch_score
            batch_valid[platform_idx] = True
            if self.platform_score_initialized[platform_idx]:
                old_score = self.platform_score_ema[platform_idx]
                self.platform_score_ema[platform_idx] = self.score_momentum * old_score + (1.0 - self.score_momentum) * batch_score
            else:
                self.platform_score_ema[platform_idx] = batch_score
                self.platform_score_initialized[platform_idx] = True
            self.platform_seen_count[platform_idx] += float(num_samples)

        ready_mask = self.platform_score_initialized & (self.platform_seen_count >= float(self.min_platform_seen))
        status_ready = bool(ready_mask.sum().item() >= 2)
        return batch_scores, batch_valid, status_ready

    def _select_strong_weak_platforms(self):
        ready_mask = self.platform_score_initialized & (self.platform_seen_count >= float(self.min_platform_seen))
        if int(ready_mask.sum().item()) < 2:
            return -1, -1, self.platform_score_ema.sum() * 0.0, False

        ready_indices = torch.nonzero(ready_mask, as_tuple=False).squeeze(1)
        ready_scores = self.platform_score_ema[ready_indices]
        if self.status_mode == "proto_confidence":
            strong_pos = torch.argmax(ready_scores)
            weak_pos = torch.argmin(ready_scores)
            platform_gap = ready_scores[strong_pos] - ready_scores[weak_pos]
        elif self.status_mode == "box_difficulty":
            weak_pos = torch.argmax(ready_scores)
            strong_pos = torch.argmin(ready_scores)
            weak_score = ready_scores[weak_pos]
            strong_score = ready_scores[strong_pos]
            platform_gap = (weak_score - strong_score) / (0.5 * (weak_score + strong_score) + 1e-6)
        else:
            raise ValueError(f"Unknown status_mode: {self.status_mode}")
        strong_platform = int(ready_indices[strong_pos].item())
        weak_platform = int(ready_indices[weak_pos].item())
        return strong_platform, weak_platform, platform_gap, True

    def forward(self, features, platform_labels, class_labels, status_values=None, epoch=None):
        device = features.device
        if features.numel() == 0:
            zero = self._zero(features)
            batch_scores = torch.zeros(self.num_platforms, device=device)
            batch_valid = torch.zeros(self.num_platforms, dtype=torch.bool, device=device)
            return zero, self._stats(zero, zero, zero, -1, -1, False, False, False, False, False, 1.0, 0, 0, 0, 0, 0, batch_scores, batch_valid)

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
            batch_scores = torch.zeros(self.num_platforms, device=device)
            batch_valid = torch.zeros(self.num_platforms, dtype=torch.bool, device=device)
            return zero, self._stats(zero, zero, zero, -1, -1, False, False, False, False, False, 1.0, 0, 0, 0, 0, 0, batch_scores, batch_valid)

        features = features[valid]
        platform_labels = platform_labels[valid]
        class_labels = class_labels[valid]
        if status_values is not None:
            status_values = status_values.to(device=device, dtype=features.dtype)[valid]
        num_valid_samples = int(features.shape[0])
        num_active_platforms = int(platform_labels.unique().numel())

        z = F.normalize(self.projector(features), p=2, dim=-1)
        sample_proto, platform_initialized, global_initialized, fallback_mask = self._build_sample_prototypes(platform_labels)
        logits = torch.bmm(sample_proto, z.unsqueeze(-1)).squeeze(-1) / self.temperature
        logits = logits.clamp(min=-50.0, max=50.0)

        probs = F.softmax(logits, dim=-1)
        if self.status_mode == "proto_confidence":
            platform_status_values = probs.gather(1, class_labels.unsqueeze(1)).squeeze(1).detach()
        elif self.status_mode == "box_difficulty":
            if status_values is None:
                platform_status_values = features.new_zeros((features.shape[0],))
            else:
                platform_status_values = status_values.detach()
        else:
            raise ValueError(f"Unknown status_mode: {self.status_mode}")
        batch_scores, batch_valid, _ = self._update_platform_status(platform_status_values, platform_labels)
        strong_platform, weak_platform, platform_gap, status_ready = self._select_strong_weak_platforms()

        gap_over_threshold = bool(platform_gap.detach().item() > self.gap_threshold)
        pce_active = bool(self.use_pce and num_valid_samples > 0)
        pce_rebalance_active = bool(pce_active and weak_platform >= 0 and status_ready and gap_over_threshold)
        weak_pce_weight = self._zero(features) + 1.0
        if pce_active:
            loss_pce_each = F.cross_entropy(logits, class_labels, reduction="none")
            sample_weights = torch.ones_like(loss_pce_each)
            if pce_rebalance_active:
                rebalance_strength = (platform_gap.detach() / (self.gap_threshold + 1e-6)).clamp(min=0.0, max=self.max_pce_boost)
                weak_pce_weight = 1.0 + self.weak_pce_boost * rebalance_strength
                weak_mask = platform_labels == weak_platform
                sample_weights = torch.where(weak_mask, weak_pce_weight.to(sample_weights.dtype), sample_weights)
            loss_pce_raw = (loss_pce_each * sample_weights).mean()
        else:
            loss_pce_raw = self._zero(features)
        loss_pce = loss_pce_raw * self.pce_weight

        loss_per_raw = self._zero(features)
        current_epoch = -1 if epoch is None else int(epoch)
        per_active = bool(
            self.use_per
            and current_epoch > self.warmup_epoch
            and status_ready
            and gap_over_threshold
            and strong_platform >= 0
        )
        proto_active = bool(pce_rebalance_active or per_active)
        if per_active:
            strong_mask = platform_labels == strong_platform
            if strong_mask.any():
                entropy = -(probs[strong_mask] * torch.log(probs[strong_mask].clamp_min(1e-8))).sum(dim=-1)
                loss_per_raw = -entropy.mean() / math.log(self.num_classes)

        loss_per = loss_per_raw * self.per_weight
        loss_proto = loss_pce + loss_per
        valid_platform_proto_count = int(self.prototype_initialized.sum().item())
        valid_global_proto_count = int(self.global_prototype_initialized.sum().item())
        fallback_proto_count = int(fallback_mask.sum().item())
        self._update_prototypes(z.detach(), platform_labels, class_labels)
        return loss_proto, self._stats(
            loss_pce,
            loss_per,
            platform_gap,
            weak_platform,
            strong_platform,
            proto_active,
            pce_active,
            per_active,
            status_ready,
            pce_rebalance_active,
            weak_pce_weight,
            num_valid_samples,
            num_active_platforms,
            valid_platform_proto_count,
            valid_global_proto_count,
            fallback_proto_count,
            batch_scores,
            batch_valid,
        )

    def _stats(
        self,
        loss_pce,
        loss_per,
        platform_gap,
        weak_platform,
        strong_platform,
        proto_active,
        pce_active,
        per_active,
        status_ready,
        pce_rebalance_active,
        weak_pce_weight,
        num_valid_samples,
        num_active_platforms,
        valid_platform_proto_count,
        valid_global_proto_count,
        fallback_proto_count,
        batch_scores,
        batch_valid,
    ):
        device = loss_pce.device
        if not torch.is_tensor(weak_pce_weight):
            weak_pce_weight = torch.tensor(float(weak_pce_weight), device=device)
        stats = {
            "loss_pce": loss_pce.detach(),
            "loss_per": loss_per.detach(),
            "platform_gap": platform_gap.detach(),
            "proto_active": torch.tensor(float(proto_active), device=device),
            "pce_active": torch.tensor(float(pce_active), device=device),
            "per_active": torch.tensor(float(per_active), device=device),
            "status_ready": torch.tensor(float(status_ready), device=device),
            "pce_rebalance_active": torch.tensor(float(pce_rebalance_active), device=device),
            "weak_pce_weight": weak_pce_weight.detach().to(device),
            "num_valid_samples": torch.tensor(float(num_valid_samples), device=device),
            "num_active_platforms": torch.tensor(float(num_active_platforms), device=device),
            "valid_platform_proto_count": torch.tensor(float(valid_platform_proto_count), device=device),
            "valid_global_proto_count": torch.tensor(float(valid_global_proto_count), device=device),
            "fallback_proto_count": torch.tensor(float(fallback_proto_count), device=device),
            "weak_platform": torch.tensor(float(weak_platform), device=device),
            "strong_platform": torch.tensor(float(strong_platform), device=device),
        }
        for platform_idx in range(len(self.platform_score_ema)):
            stats[f"platform_score_ema_{platform_idx}"] = self.platform_score_ema[platform_idx].detach().to(device)
            stats[f"platform_seen_count_{platform_idx}"] = self.platform_seen_count[platform_idx].detach().to(device)
            stats[f"platform_batch_score_{platform_idx}"] = batch_scores[platform_idx].detach().to(device)
            stats[f"platform_batch_valid_{platform_idx}"] = batch_valid[platform_idx].float().detach().to(device)
        return stats
