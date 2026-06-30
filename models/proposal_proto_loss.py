"""Proposal-level prototype ranking loss for language-guided grounding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_aligned_iou3d(boxes1, boxes2):
    """Pairwise IoU for aligned [cx, cy, cz, sx, sy, sz] boxes."""
    size1 = boxes1[..., 3:].clamp_min(1e-6)
    size2 = boxes2[..., 3:].clamp_min(1e-6)
    min1 = boxes1[..., :3] - 0.5 * size1
    max1 = boxes1[..., :3] + 0.5 * size1
    min2 = boxes2[..., :3] - 0.5 * size2
    max2 = boxes2[..., :3] + 0.5 * size2

    inter_min = torch.maximum(min1[:, None, :], min2[None, :, :])
    inter_max = torch.minimum(max1[:, None, :], max2[None, :, :])
    inter_size = (inter_max - inter_min).clamp_min(0.0)
    inter_volume = inter_size.prod(dim=-1)
    volume1 = size1.prod(dim=-1)[:, None]
    volume2 = size2.prod(dim=-1)[None, :]
    return inter_volume / (volume1 + volume2 - inter_volume).clamp_min(1e-6)


class ProposalPrototypeRankingLoss(nn.Module):
    """Rank IoU-positive proposals above language-hard negative proposals."""

    def __init__(
        self,
        weight=0.0,
        tau=0.07,
        pos_iou_thr=0.5,
        neg_iou_thr=0.25,
        hard_negative_topk=5,
        warmup_epoch=0,
    ):
        super().__init__()
        if weight < 0:
            raise ValueError("prop_proto_weight must be non-negative")
        if tau <= 0:
            raise ValueError("prop_proto_tau must be positive")
        if not 0 <= neg_iou_thr < pos_iou_thr <= 1:
            raise ValueError(
                "prop proposal IoU thresholds must satisfy "
                "0 <= prop_neg_iou_thr < prop_pos_iou_thr <= 1"
            )
        if hard_negative_topk <= 0:
            raise ValueError("prop_hn_topk must be positive")
        if warmup_epoch < 0:
            raise ValueError("prop_proto_warmup_epoch must be non-negative")

        self.weight = float(weight)
        self.tau = float(tau)
        self.pos_iou_thr = float(pos_iou_thr)
        self.neg_iou_thr = float(neg_iou_thr)
        self.hard_negative_topk = int(hard_negative_topk)
        self.warmup_epoch = int(warmup_epoch)

    def forward(
        self,
        query_features,
        pred_boxes,
        targets,
        language_scores=None,
        epoch=None,
    ):
        zero = query_features.sum() * 0.0
        stats = {
            "loss_prop_proto_raw": zero.detach(),
            "prop_proto_active": zero.detach(),
            "prop_proto_pos_count": zero.detach(),
            "prop_proto_neg_count": zero.detach(),
            "prop_proto_fallback_pos_count": zero.detach(),
        }
        current_epoch = int(epoch) if epoch is not None else 0
        if current_epoch < self.warmup_epoch:
            return zero, stats

        sample_losses = []
        positive_count = 0
        negative_count = 0
        fallback_positive_count = 0
        normalized_features = F.normalize(query_features, p=2, dim=-1)

        for batch_idx, target in enumerate(targets):
            target_boxes = target["boxes"]
            if target_boxes.numel() == 0:
                continue

            proposal_ious = _pairwise_aligned_iou3d(
                pred_boxes[batch_idx],
                target_boxes.to(device=pred_boxes.device, dtype=pred_boxes.dtype),
            ).max(dim=1).values
            positive_indices = torch.nonzero(
                proposal_ious >= self.pos_iou_thr, as_tuple=False
            ).flatten()
            negative_indices = torch.nonzero(
                proposal_ious <= self.neg_iou_thr, as_tuple=False
            ).flatten()
            if positive_indices.numel() == 0:
                positive_indices = proposal_ious.argmax().reshape(1)
                fallback_positive_count += 1
            negative_indices = negative_indices[
                ~torch.isin(negative_indices, positive_indices)
            ]
            if negative_indices.numel() == 0:
                continue

            positive_features = normalized_features[batch_idx, positive_indices]
            positive_weights = proposal_ious[positive_indices].clamp_min(1e-6)
            prototype = (
                positive_features * positive_weights[:, None]
            ).sum(dim=0) / positive_weights.sum().clamp_min(1e-6)
            prototype = F.normalize(prototype, p=2, dim=0).detach()

            if language_scores is None:
                hard_scores = torch.matmul(
                    normalized_features[batch_idx, negative_indices],
                    prototype,
                )
            else:
                hard_scores = language_scores[batch_idx, negative_indices]
            topk = min(self.hard_negative_topk, negative_indices.numel())
            hard_negative_indices = negative_indices[
                torch.topk(hard_scores, k=topk, largest=True).indices
            ]

            positive_similarity = torch.matmul(positive_features, prototype).mean()
            negative_similarity = torch.matmul(
                normalized_features[batch_idx, hard_negative_indices],
                prototype,
            )
            sample_losses.append(
                F.softplus(
                    (negative_similarity - positive_similarity) / self.tau
                ).mean()
            )
            positive_count += int(positive_indices.numel())
            negative_count += int(hard_negative_indices.numel())

        if not sample_losses:
            return zero, stats

        raw_loss = torch.stack(sample_losses).mean()
        weighted_loss = raw_loss * self.weight
        stats.update(
            {
                "loss_prop_proto_raw": raw_loss.detach(),
                "prop_proto_active": raw_loss.new_tensor(1.0),
                "prop_proto_pos_count": raw_loss.new_tensor(float(positive_count)),
                "prop_proto_neg_count": raw_loss.new_tensor(float(negative_count)),
                "prop_proto_fallback_pos_count": raw_loss.new_tensor(
                    float(fallback_positive_count)
                ),
            }
        )
        return weighted_loss, stats
