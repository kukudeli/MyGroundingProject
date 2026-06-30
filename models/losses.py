# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .prototype_rebalance import PlatformPrototypeRebalanceLoss
from .proposal_proto_loss import ProposalPrototypeRankingLoss


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def box_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    w = torch.clamp(w, min=1e-6)
    h = torch.clamp(h, min=1e-6)
    d = torch.clamp(d, min=1e-6)
    assert (w < 0).sum() == 0
    assert (h < 0).sum() == 0
    assert (d < 0).sum() == 0
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


def rotated_gt_to_enclosing_aligned_box_torch(gt_bboxes):
    """Convert 7D yaw-rotated GT boxes to minimal global aligned 6D AABBs."""
    if gt_bboxes.shape[-1] < 7:
        raise ValueError(
            f"gt_bboxes must have at least 7 values [cx,cy,cz,sx,sy,sz,yaw], got {tuple(gt_bboxes.shape)}"
        )

    center = gt_bboxes[..., :3]
    size = gt_bboxes[..., 3:6].clamp_min(1e-6)
    yaw = gt_bboxes[..., 6]
    half_size = 0.5 * size

    corner_signs = gt_bboxes.new_tensor([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
    ])
    view_shape = (1,) * (gt_bboxes.dim() - 1) + (8, 3)
    local_corners = half_size.unsqueeze(-2) * corner_signs.view(view_shape)

    cos_yaw = torch.cos(yaw).unsqueeze(-1)
    sin_yaw = torch.sin(yaw).unsqueeze(-1)
    x_local = local_corners[..., 0]
    y_local = local_corners[..., 1]
    z_local = local_corners[..., 2]
    x_rot = cos_yaw * x_local - sin_yaw * y_local
    y_rot = sin_yaw * x_local + cos_yaw * y_local
    corners = torch.stack([x_rot, y_rot, z_local], dim=-1) + center.unsqueeze(-2)

    min_corner = corners.amin(dim=-2)
    max_corner = corners.amax(dim=-2)
    aligned_center = 0.5 * (min_corner + max_corner)
    aligned_size = (max_corner - min_corner).clamp_min(1e-6)
    return torch.cat([aligned_center, aligned_size], dim=-1)


def _volume_par(box):
    return (
        (box[:, 3] - box[:, 0])
        * (box[:, 4] - box[:, 1])
        * (box[:, 5] - box[:, 2])
    )


def _intersect_par(box_a, box_b):
    xA = torch.max(box_a[:, 0][:, None], box_b[:, 0][None, :])
    yA = torch.max(box_a[:, 1][:, None], box_b[:, 1][None, :])
    zA = torch.max(box_a[:, 2][:, None], box_b[:, 2][None, :])
    xB = torch.min(box_a[:, 3][:, None], box_b[:, 3][None, :])
    yB = torch.min(box_a[:, 4][:, None], box_b[:, 4][None, :])
    zB = torch.min(box_a[:, 5][:, None], box_b[:, 5][None, :])
    return (
        torch.clamp(xB - xA, 0)
        * torch.clamp(yB - yA, 0)
        * torch.clamp(zB - zA, 0)
    )


def _iou3d_par(box_a, box_b):
    intersection = _intersect_par(box_a, box_b)
    vol_a = _volume_par(box_a)
    vol_b = _volume_par(box_b)
    union = vol_a[:, None] + vol_b[None, :] - intersection
    return intersection / union, union


def _aligned_iou3d(boxes1, boxes2):
    if boxes1.numel() == 0:
        return boxes1.new_zeros((0,))
    lt = torch.max(boxes1[:, :3], boxes2[:, :3])
    rb = torch.min(boxes1[:, 3:], boxes2[:, 3:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, 0] * wh[:, 1] * wh[:, 2]
    vol1 = _volume_par(boxes1)
    vol2 = _volume_par(boxes2)
    union = (vol1 + vol2 - intersection).clamp_min(1e-6)
    return intersection / union


def generalized_box_iou3d(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    iou, union = _iou3d_par(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)  # [N,M,3]
    volume = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]

    return iou - (volume - union) / volume


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.

    This class is taken from Group-Free code.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """
        Args:
            gamma: Weighting parameter for hard and easy examples.
            alpha: Weighting parameter for positive and negative examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input, target):
        """
        PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
        max(x, 0) - x * z + log(1 + exp(-abs(x))) in

        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #proposals, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = (
            torch.clamp(input, min=0) - input * target
            + torch.log1p(torch.exp(-torch.abs(input)))
        )
        return loss

    def forward(self, input, target, weights):
        """
        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #proposals) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #proposals, #classes) float tensor
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss
        loss = loss.squeeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


def compute_points_obj_cls_loss_hard_topk(end_points, topk):
    box_label_mask = end_points['box_label_mask']
    seed_inds = end_points['seed_inds'].long()  # B, K
    seed_xyz = end_points['seed_xyz']  # B, K, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']  # B, 1, K
    gt_center = end_points['center_label'][:, :, :3]  # B, G, 3
    gt_size = end_points['size_gts'][:, :, :3]  # B, G, 3
    B = gt_center.shape[0]  # batch size
    K = seed_xyz.shape[1]  # number if points from p++ output
    G = gt_center.shape[1]  # number of gt boxes (with padding)

    # Assign each point to a GT object
    point_instance_label = end_points['point_instance_label']  # B, num_points
    obj_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, K
    obj_assignment[obj_assignment < 0] = G - 1  # bg points to last gt
    obj_assignment_one_hot = torch.zeros((B, K, G)).to(seed_xyz.device)
    obj_assignment_one_hot.scatter_(2, obj_assignment.unsqueeze(-1), 1)

    # Normalized distances of points and gt centroids
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, G, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  # (B, K, G, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxG
    euclidean_dist1 = (
        euclidean_dist1 * obj_assignment_one_hot
        + 100 * (1 - obj_assignment_one_hot)
    )  # BxKxG
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  # BxGxK

    # Find the points that lie closest to each gt centroid
    topk_inds = (
        torch.topk(euclidean_dist1, topk, largest=False)[1]
        * box_label_mask[:, :, None]
        + (box_label_mask[:, :, None] - 1)
    )  # BxGxtopk
    topk_inds = topk_inds.long()  # BxGxtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, Gxtopk
    batch_inds = torch.arange(B)[:, None].repeat(1, G*topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([
        batch_inds,
        topk_inds
    ], -1).view(-1, 2).contiguous()

    # Topk points closest to each centroid are marked as true objects
    objectness_label = torch.zeros((B, K + 1)).long().to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)
    objectness_label[objectness_label_mask < 0] = 0

    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(
        seeds_obj_cls_logits.view(B, K, 1),
        objectness_label.unsqueeze(-1),
        weights=cls_weights
    )
    objectness_loss = cls_loss_src.sum() / B

    return objectness_loss


class HungarianMatcher(nn.Module):
    """
    Assign targets to predictions.

    This class is taken from MDETR and is modified for our purposes.

    For efficiency reasons, the targets don't include the no_object.
    Because of this, in general, there are more predictions than targets.
    In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2,
                 soft_token=False):
        """
        Initialize matcher.

        Args:
            cost_class: relative weight of the classification error
            cost_bbox: relative weight of the L1 bounding box regression error
            cost_giou: relative weight of the giou loss of the bounding box
            soft_token: whether to use soft-token prediction
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        self.soft_token = soft_token

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Perform the matching.

        Args:
            outputs: This is a dict that contains at least these entries:
                "pred_logits" (tensor): [batch_size, num_queries, num_classes]
                "pred_boxes" (tensor): [batch_size, num_queries, 6], cxcyczwhd
            targets: list (len(targets) = batch_size) of dict:
                "labels" (tensor): [num_target_boxes]
                    (where num_target_boxes is the no. of ground-truth objects)
                "boxes" (tensor): [num_target_boxes, 6], cxcyczwhd
                "positive_map" (tensor): [num_target_boxes, 256]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j):
                - index_i is the indices of the selected predictions
                - index_j is the indices of the corresponding selected targets
            For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # Notation: {B: batch_size, Q: num_queries, C: num_classes}
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*Q, C]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*Q, 6]

        # Also concat the target labels and boxes
        positive_map = torch.cat([t["positive_map"] for t in targets])
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.soft_token:
            # pad if necessary
            if out_prob.shape[-1] != positive_map.shape[-1]:
                positive_map = positive_map[..., :out_prob.shape[-1]]
            cost_class = -torch.matmul(out_prob, positive_map.transpose(0, 1))
        else:
            # Compute the classification cost.
            # Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching,
            # it can be ommitted. DETR
            # out_prob = out_prob * out_objectness.view(-1, 1)
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou3d(
            box_cxcyczwhd_to_xyzxyz(out_bbox),
            box_cxcyczwhd_to_xyzxyz(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        ).view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),  # matched pred boxes
                torch.as_tensor(j, dtype=torch.int64)  # corresponding gt boxes
            )
            for i, j in indices
        ]


class SetCriterion(nn.Module):
    """
    Computes the loss in two steps:
        1) compute hungarian assignment between ground truth and outputs
        2) supervise each pair of matched ground-truth / prediction
    """

    def __init__(
        self,
        matcher,
        losses={},
        eos_coef=0.1,
        temperature=0.07,
        use_platform_proto=False,
        proto_in_dim=288,
        proto_dim=128,
        num_platforms=3,
        num_proto_classes=6,
        proto_momentum=0.9,
        proto_temperature=0.07,
        proto_gap_threshold=0.05,
        proto_pce_weight=0.1,
        proto_per_weight=0.01,
        proto_warmup_epoch=5,
        proto_use_pce=True,
        proto_use_per=True,
        proto_feature_mode="matched_query",
        proto_status_mode="box_difficulty",
        proto_score_momentum=0.9,
        proto_min_platform_samples=1,
        proto_min_platform_seen=5,
        proto_weak_pce_boost=1.0,
        proto_max_pce_boost=2.0,
        use_enclosing_aligned_gt_as_box_target=False,
        use_prop_proto=False,
        prop_proto_weight=0.0,
        prop_proto_tau=0.07,
        prop_pos_iou_thr=0.5,
        prop_neg_iou_thr=0.25,
        prop_hn_topk=5,
        prop_proto_warmup_epoch=0,
    ):
        """
        Parameters:
            matcher: module that matches targets and proposals
            losses: list of all the losses to be applied
            eos_coef: weight of the no-object category
            temperature: used to sharpen the contrastive logits
        """
        super().__init__()
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.temperature = temperature
        self.use_platform_proto = use_platform_proto
        self.proto_feature_mode = proto_feature_mode
        self.proto_status_mode = proto_status_mode
        self.num_platforms = num_platforms
        self.proto_score_momentum = proto_score_momentum
        self.proto_min_platform_samples = proto_min_platform_samples
        self.proto_min_platform_seen = proto_min_platform_seen
        self.use_enclosing_aligned_gt_as_box_target = use_enclosing_aligned_gt_as_box_target
        self.use_prop_proto = use_prop_proto
        self.prop_proto_weight = prop_proto_weight
        self.prop_proto_tau = prop_proto_tau
        self.prop_pos_iou_thr = prop_pos_iou_thr
        self.prop_neg_iou_thr = prop_neg_iou_thr
        self.prop_hn_topk = prop_hn_topk
        self.prop_proto_warmup_epoch = prop_proto_warmup_epoch
        if self.use_platform_proto:
            self.platform_proto_loss = PlatformPrototypeRebalanceLoss(
                in_dim=proto_in_dim,
                proto_dim=proto_dim,
                num_platforms=num_platforms,
                num_classes=num_proto_classes,
                momentum=proto_momentum,
                temperature=proto_temperature,
                gap_threshold=proto_gap_threshold,
                pce_weight=proto_pce_weight,
                per_weight=proto_per_weight,
                warmup_epoch=proto_warmup_epoch,
                use_pce=proto_use_pce,
                use_per=proto_use_per,
                status_mode=proto_status_mode,
                score_momentum=proto_score_momentum,
                min_platform_samples=proto_min_platform_samples,
                min_platform_seen=proto_min_platform_seen,
                weak_pce_boost=proto_weak_pce_boost,
                max_pce_boost=proto_max_pce_boost,
            )
        if self.use_prop_proto:
            self.proposal_proto_loss = ProposalPrototypeRankingLoss(
                weight=prop_proto_weight,
                tau=prop_proto_tau,
                pos_iou_thr=prop_pos_iou_thr,
                neg_iou_thr=prop_neg_iou_thr,
                hard_negative_topk=prop_hn_topk,
                warmup_epoch=prop_proto_warmup_epoch,
            )

    def loss_labels_st(self, outputs, targets, indices, num_boxes):
        """Soft token prediction (with objectness)."""
        logits = outputs["pred_logits"].log_softmax(-1)  # (B, Q, 256)
        positive_map = torch.cat([t["positive_map"] for t in targets])

        # Trick to get target indices across batches
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        # Labels, by default lines map to the last element, no_object
        tgt_pos = positive_map[tgt_idx]
        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_pos

        # Compute entropy
        entropy = torch.log(target_sim + 1e-6) * target_sim
        loss_ce = (entropy - logits * target_sim).sum(-1)

        # Weight less 'no_object'
        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef,
            device=target_sim.device
        )
        eos_coef[src_idx] = 1
        loss_ce = loss_ce * eos_coef
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute bbox losses."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)

        loss_bbox = (
            F.l1_loss(
                src_boxes[..., :3], target_boxes[..., :3],
                reduction='none'
            )
            + 0.2 * F.l1_loss(
                src_boxes[..., 3:], target_boxes[..., 3:],
                reduction='none'
            )
        )
        losses = {}

        loss_giou = 1 - torch.diag(generalized_box_iou3d(
            box_cxcyczwhd_to_xyzxyz(src_boxes),
            box_cxcyczwhd_to_xyzxyz(target_boxes)))
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_contrastive_align(self, outputs, targets, indices, num_boxes):
        """Compute contrastive losses between projected queries and tokens."""
        tokenized = outputs["tokenized"]

        # Contrastive logits
        norm_text_emb = outputs["proj_tokens"]  # B, num_tokens, dim
        norm_img_emb = outputs["proj_queries"]  # B, num_queries, dim
        logits = (
            torch.matmul(norm_img_emb, norm_text_emb.transpose(-1, -2))
            / self.temperature
        )  # B, num_queries, num_tokens

        # construct a map such that positive_map[k, i, j] = True
        # iff query i is associated to token j in batch item k
        positive_map = torch.zeros(logits.shape, device=logits.device)
        # handle 'not mentioned'
        inds = tokenized['attention_mask'].sum(1) - 1
        positive_map[torch.arange(len(inds)), :, inds] = 0.5
        positive_map[torch.arange(len(inds)), :, inds - 1] = 0.5
        # handle true mentions
        pmap = torch.cat([
            t['positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)
        idx = self._get_src_permutation_idx(indices)
        positive_map[idx] = pmap[..., :logits.shape[-1]]
        positive_map = positive_map > 0

        # Mask for matches <> 'not mentioned'
        mask = torch.full(
            logits.shape[:2],
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        mask[idx] = 1.0
        # Token mask for matches <> 'not mentioned'
        tmask = torch.full(
            (len(logits), logits.shape[-1]),
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        tmask[torch.arange(len(inds)), inds] = 1.0

        # Positive logits are those who correspond to a match
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits

        # Loss 1: which tokens should each query match?
        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)
        nb_pos = positive_map.sum(2) + 1e-6
        entropy = -torch.log(nb_pos+1e-6) / nb_pos  # entropy of 1/nb_pos
        box_to_token_loss_ = (
            (entropy + pos_term / nb_pos + neg_term)
        ).masked_fill(~boxes_with_pos, 0)
        box_to_token_loss = (box_to_token_loss_ * mask).sum()

        # Loss 2: which queries should each token match?
        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)
        nb_pos = positive_map.sum(1) + 1e-6
        entropy = -torch.log(nb_pos+1e-6) / nb_pos
        token_to_box_loss = (
            (entropy + pos_term / nb_pos + neg_term)
        ).masked_fill(~tokens_with_pos, 0)
        token_to_box_loss = (token_to_box_loss * tmask).sum()

        tot_loss = (box_to_token_loss + token_to_box_loss) / 2
        return {"loss_contrastive_align": tot_loss / num_boxes}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
        ])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels_st,
            'boxes': self.loss_boxes,
            'contrastive_align': self.loss_contrastive_align
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        Perform the loss computation.

        Parameters:
             outputs: dict of tensors
             targets: list of dicts, such that len(targets) == batch_size.
        """
        # Retrieve the matching between outputs and targets
        indices = self.matcher(outputs, targets)

        num_boxes = sum(len(inds[1]) for inds in indices)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss,
                outputs,
                targets,
                indices,
                num_boxes,
            ))

        return losses, indices


def extract_matched_query_features_and_box_difficulty(
    proto_query_features,
    last_pred_boxes,
    indices,
    targets,
    platform_labels,
):
    """Flatten final-layer matches into prototype features and box difficulty."""
    device = proto_query_features.device
    matched_features = []
    matched_platform_labels = []
    matched_class_labels = []
    matched_box_difficulty = []
    matched_ious = []

    if last_pred_boxes is None:
        indices = []

    for batch_idx, (src_idx, tgt_idx) in enumerate(indices):
        if len(src_idx) == 0 or len(tgt_idx) == 0:
            continue

        src_idx = src_idx.to(device=device, dtype=torch.long)
        tgt_idx = tgt_idx.to(device=targets[batch_idx]["labels"].device, dtype=torch.long)

        features = proto_query_features[batch_idx, src_idx]
        pred_boxes = last_pred_boxes[batch_idx, src_idx]
        gt_boxes = targets[batch_idx]["boxes"][tgt_idx].to(device=device)
        labels = targets[batch_idx]["labels"][tgt_idx].to(device=device, dtype=torch.long)
        platforms = platform_labels[batch_idx].to(device=device, dtype=torch.long).expand(labels.shape[0])
        with torch.no_grad():
            bbox_l1 = (
                F.l1_loss(
                    pred_boxes[..., :3],
                    gt_boxes[..., :3],
                    reduction="none",
                ).sum(dim=-1)
                + 0.2 * F.l1_loss(
                    pred_boxes[..., 3:],
                    gt_boxes[..., 3:],
                    reduction="none",
                ).sum(dim=-1)
            )
            pred_xyzxyz = box_cxcyczwhd_to_xyzxyz(pred_boxes)
            gt_xyzxyz = box_cxcyczwhd_to_xyzxyz(gt_boxes)
            matched_iou = _aligned_iou3d(pred_xyzxyz, gt_xyzxyz).detach()
            giou_each = 1 - torch.diag(generalized_box_iou3d(pred_xyzxyz, gt_xyzxyz))
            difficulty = (bbox_l1 + giou_each).detach()

        matched_features.append(features)
        matched_platform_labels.append(platforms)
        matched_class_labels.append(labels)
        matched_box_difficulty.append(difficulty)
        matched_ious.append(matched_iou)

    if len(matched_features) == 0:
        feature_dim = proto_query_features.shape[-1]
        return (
            proto_query_features.new_zeros((0, feature_dim)),
            platform_labels.new_zeros((0,), dtype=torch.long).to(device),
            platform_labels.new_zeros((0,), dtype=torch.long).to(device),
            proto_query_features.new_zeros((0,)),
            proto_query_features.new_zeros((0,)),
        )

    return (
        torch.cat(matched_features, dim=0),
        torch.cat(matched_platform_labels, dim=0),
        torch.cat(matched_class_labels, dim=0),
        torch.cat(matched_box_difficulty, dim=0),
        torch.cat(matched_ious, dim=0),
    )


def compute_hungarian_loss(end_points, num_decoder_layers, set_criterion,
                           query_points_obj_topk=5):
    """Compute Hungarian matching loss containing CE, bbox and giou."""
    prefixes = ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    prefixes = ['proposal_'] + prefixes

    # Ground-truth
    gt_center = end_points['center_label'][:, :, 0:3]  # B, G, 3
    gt_size = end_points['size_gts']  # (B,G,3)
    gt_labels = end_points['sem_cls_label']  # (B, G)
    zero = gt_center.sum() * 0.0
    original_gt_bbox = torch.cat([gt_center, gt_size], dim=-1)  # cxcyczwhd
    gt_bbox = original_gt_bbox
    if set_criterion.use_enclosing_aligned_gt_as_box_target:
        if 'gt_bboxes' not in end_points:
            raise ValueError(
                "enclosing aligned GT target requires end_points['gt_bboxes'] with 7D rotated GT boxes"
            )
        gt_bboxes_rotated = end_points['gt_bboxes'].to(device=gt_center.device, dtype=gt_center.dtype)
        if gt_bboxes_rotated.shape[-1] < 7:
            raise ValueError(
                f"end_points['gt_bboxes'] must have at least 7 dims, got {tuple(gt_bboxes_rotated.shape)}"
            )
        gt_bbox = rotated_gt_to_enclosing_aligned_box_torch(gt_bboxes_rotated)
    positive_map = end_points['positive_map']
    box_label_mask = end_points['box_label_mask']
    valid_gt_mask = box_label_mask.bool()
    if valid_gt_mask.any():
        original_valid_size = original_gt_bbox[..., 3:][valid_gt_mask].clamp_min(1e-6)
        target_valid_size = gt_bbox[..., 3:][valid_gt_mask].clamp_min(1e-6)
        original_valid_volume = original_valid_size.prod(dim=-1)
        target_valid_volume = target_valid_size.prod(dim=-1)
        end_points['enclosing_box_target_active'] = torch.tensor(
            float(set_criterion.use_enclosing_aligned_gt_as_box_target), device=gt_center.device
        )
        end_points['enclosing_box_target_size_mean'] = target_valid_size.detach().mean()
        end_points['enclosing_box_target_size_min'] = target_valid_size.detach().min()
        end_points['enclosing_box_target_volume_mean'] = target_valid_volume.detach().mean()
        end_points['original_gt_size_mean'] = original_valid_size.detach().mean()
        end_points['original_gt_volume_mean'] = original_valid_volume.detach().mean()
        end_points['enclosing_to_original_volume_ratio_mean'] = (
            target_valid_volume / original_valid_volume.clamp_min(1e-6)
        ).detach().mean()
    else:
        end_points['enclosing_box_target_active'] = torch.tensor(
            float(set_criterion.use_enclosing_aligned_gt_as_box_target), device=gt_center.device
        )
        end_points['enclosing_box_target_size_mean'] = zero.detach()
        end_points['enclosing_box_target_size_min'] = zero.detach()
        end_points['enclosing_box_target_volume_mean'] = zero.detach()
        end_points['original_gt_size_mean'] = zero.detach()
        end_points['original_gt_volume_mean'] = zero.detach()
        end_points['enclosing_to_original_volume_ratio_mean'] = zero.detach()
    target = []
    for b in range(gt_labels.shape[0]):
        valid_mask = box_label_mask[b].bool()
        target_item = {
            "labels": gt_labels[b, valid_mask],
            "boxes": gt_bbox[b, valid_mask],
            "positive_map": positive_map[b, valid_mask],
        }
        target.append(target_item)

    loss_ce, loss_bbox, loss_giou, loss_contrastive_align = 0, 0, 0, 0
    last_layer_indices = None
    last_layer_pred_boxes = None
    for prefix in prefixes:
        output = {}
        if 'proj_tokens' in end_points:
            output['proj_tokens'] = end_points['proj_tokens']
            output['proj_queries'] = end_points[f'{prefix}proj_queries']
            output['tokenized'] = end_points['tokenized']

        # Get predicted boxes and labels
        pred_center = end_points[f'{prefix}center']  # B, K, 3
        pred_size = end_points[f'{prefix}pred_size']  # (B,K,3) (l,w,h)
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        pred_logits = end_points[f'{prefix}sem_cls_scores']  # (B, Q, n_class)
        output['pred_logits'] = pred_logits
        output["pred_boxes"] = pred_bbox

        # Compute all the requested losses
        losses, indices = set_criterion(output, target)
        if prefix == "last_":
            last_layer_indices = indices
            last_layer_pred_boxes = pred_bbox
        for loss_key in losses.keys():
            end_points[f'{prefix}_{loss_key}'] = losses[loss_key]
        loss_ce += losses.get('loss_ce', 0)
        loss_bbox += losses['loss_bbox']
        loss_giou += losses.get('loss_giou', 0)
        if 'proj_tokens' in end_points:
            loss_contrastive_align += losses['loss_contrastive_align']

    if 'seeds_obj_cls_logits' in end_points.keys():
        query_points_generation_loss = compute_points_obj_cls_loss_hard_topk(
            end_points, query_points_obj_topk
        )
    else:
        query_points_generation_loss = 0.0

    # loss
    loss = (
        8 * query_points_generation_loss
        + 1.0 / (num_decoder_layers + 1) * (
            loss_ce
            + 5 * loss_bbox
            + loss_giou
            + loss_contrastive_align
        )
    )
    end_points['loss_ce'] = loss_ce
    end_points['loss_bbox'] = loss_bbox
    end_points['loss_giou'] = loss_giou
    end_points['query_points_generation_loss'] = query_points_generation_loss
    end_points['loss_constrastive_align'] = loss_contrastive_align
    if set_criterion.use_prop_proto and set_criterion.training:
        language_scores = None
        if "last_proj_queries" in end_points and "proj_tokens" in end_points:
            token_scores = torch.matmul(
                end_points["last_proj_queries"],
                end_points["proj_tokens"].transpose(-1, -2),
            )
            language_scores_per_sample = []
            for batch_idx, target_item in enumerate(target):
                positive_tokens = target_item["positive_map"].sum(dim=0)
                positive_tokens = positive_tokens[: token_scores.shape[-1]]
                positive_tokens = positive_tokens / positive_tokens.sum().clamp_min(1e-6)
                language_scores_per_sample.append(
                    torch.matmul(token_scores[batch_idx], positive_tokens)
                )
            language_scores = torch.stack(language_scores_per_sample)
        loss_prop_proto, prop_proto_stats = set_criterion.proposal_proto_loss(
            end_points["proto_query_features"],
            last_layer_pred_boxes,
            target,
            language_scores=language_scores,
            epoch=end_points.get("epoch", None),
        )
        loss = loss + loss_prop_proto
        end_points["loss_prop_proto"] = loss_prop_proto
        end_points.update(prop_proto_stats)
    else:
        end_points["loss_prop_proto"] = zero
        end_points["loss_prop_proto_raw"] = zero
        end_points["prop_proto_active"] = zero
        end_points["prop_proto_pos_count"] = zero
        end_points["prop_proto_neg_count"] = zero
        end_points["prop_proto_fallback_pos_count"] = zero
    matched_query_cache = None
    need_matched_query_cache = (
        set_criterion.training
        and "platform_label" in end_points
        and "proto_query_features" in end_points
        and set_criterion.use_platform_proto
        and set_criterion.proto_feature_mode == "matched_query"
    )
    if need_matched_query_cache:
        matched_query_cache = extract_matched_query_features_and_box_difficulty(
            end_points["proto_query_features"],
            last_layer_pred_boxes,
            last_layer_indices if last_layer_indices is not None else [],
            target,
            end_points["platform_label"].long(),
        )
    if set_criterion.use_platform_proto and set_criterion.training:
        platform_labels = end_points["platform_label"].long()
        status_values = None
        if set_criterion.proto_feature_mode == "mean_query":
            if set_criterion.proto_status_mode == "box_difficulty":
                raise ValueError(
                    "box_difficulty status mode currently requires matched_query feature mode."
                )
            proto_features = end_points.get("proto_features", end_points["proto_query_features"].mean(dim=1))
            class_labels = end_points["sem_cls_label"][:, 0].long()
            valid_mask = end_points["box_label_mask"][:, 0].bool()
            proto_features = proto_features[valid_mask]
            platform_labels = platform_labels[valid_mask]
            class_labels = class_labels[valid_mask]
        elif set_criterion.proto_feature_mode == "matched_query":
            if matched_query_cache is None:
                matched_query_cache = extract_matched_query_features_and_box_difficulty(
                    end_points["proto_query_features"],
                    last_layer_pred_boxes,
                    last_layer_indices if last_layer_indices is not None else [],
                    target,
                    platform_labels,
                )
            proto_features, platform_labels, class_labels, matched_box_difficulty, _ = matched_query_cache
            if set_criterion.proto_status_mode == "box_difficulty":
                status_values = matched_box_difficulty
            elif set_criterion.proto_status_mode == "proto_confidence":
                status_values = None
            else:
                raise ValueError(f"Unknown proto_status_mode: {set_criterion.proto_status_mode}")
        else:
            raise ValueError(f"Unknown proto_feature_mode: {set_criterion.proto_feature_mode}")
        loss_proto, proto_stats = set_criterion.platform_proto_loss(
            proto_features,
            platform_labels,
            class_labels,
            status_values=status_values,
            epoch=end_points.get("epoch", None),
        )
        loss = loss + loss_proto
        end_points["loss_proto"] = loss_proto
        for key, value in proto_stats.items():
            end_points[key] = value
    else:
        zero = loss * 0.0
        end_points["loss_proto"] = zero
        end_points["loss_pce"] = zero
        end_points["loss_per"] = zero
        end_points["platform_gap"] = zero
        end_points["proto_active"] = zero
        end_points["pce_active"] = zero
        end_points["per_active"] = zero
        end_points["status_ready"] = zero
        end_points["pce_rebalance_active"] = zero
        end_points["weak_pce_weight"] = zero
        end_points["num_valid_samples"] = zero
        end_points["num_active_platforms"] = zero
        end_points["valid_platform_proto_count"] = zero
        end_points["valid_global_proto_count"] = zero
        end_points["fallback_proto_count"] = zero
        end_points["weak_platform"] = zero
        end_points["strong_platform"] = zero
    end_points['loss'] = loss
    return loss, end_points
