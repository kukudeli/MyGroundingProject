import torch

from models.losses import _iou3d_par, box_cxcyczwhd_to_xyzxyz
from utils.eval_det import iou3d_rotated_vs_aligned
import utils.misc as misc
from collections import defaultdict
import ipdb

st = ipdb.set_trace


class GroundingEvaluator:
    """
    Evaluate language grounding.

    Args:
        only_root (bool): detect only the root noun
        thresholds (list): IoU thresholds to check
        topks (list): k to evaluate top--k accuracy
        prefixes (list): names of layers to evaluate
    """

    def __init__(self, only_root=False, thresholds=[0.25, 0.5], topks=[1, 5, 10], prefixes=[]):
        """Initialize accumulators."""
        self.only_root = only_root
        self.thresholds = thresholds
        self.topks = topks
        self.prefixes = prefixes

        self.reset()

    def reset(self):
        """Reset accumulators to empty."""
        # self.dets = {
        #     (prefix, t, k, mode): 0 for prefix in self.prefixes for t in self.thresholds for k in self.topks for mode in ["bbs", "bbf"]
        # }  # Number of hit GT boxes, e.g. accuracy at IoU 0.5 for top-1
        # self.gts = dict(self.dets)  # Total number of GT boxes

        self.dets = defaultdict(int)
        self.gts = defaultdict(int)

        self.dets.update({"vd": 0, "vid": 0})
        self.dets.update({"hard": 0, "easy": 0})
        self.dets.update({"multi": 0, "unique": 0})
        self.gts.update({"vd": 1e-14, "vid": 1e-14})
        self.gts.update({"hard": 1e-14, "easy": 1e-14})
        self.gts.update({"multi": 1e-14, "unique": 1e-14})

        # Additional total_acc statistics
        self.dets.update({("total_acc", t, "bbf"): 0 for t in self.thresholds})
        self.gts.update({("total_acc", t, "bbf"): 1e-14 for t in self.thresholds})  # Prevent division by zero

        self.prediction_records = []

    def print_stats(self):
        """Print accumulated accuracies."""
        return_str = "\n"
        mode_str = {"bbs": "Box given span (soft-token)", "bbf": "Box given span (contrastive)"}
        for prefix in ["last_", "proposal_"]:  
            for mode in ["bbs", "bbf"]:
                for t in self.thresholds:
                    line = f"{prefix} {mode_str[mode]} Acc{t:.2f}: " + ", ".join(
                        [f"Top-{k}: {self.dets[(prefix, t, k, mode)] / max(self.gts[(prefix, t, k, mode)], 1):.3f}" for k in self.topks]
                    )
                    # print(line)
                    return_str += line + "\n"

        return_str += "\n==Analysis==\n"

        for t in self.thresholds:
            acc = self.dets[("total_acc", t, "bbf")] / self.gts[("total_acc", t, "bbf")]
            return_str += f"Acc@{t} = {acc:.4f}  "

        return_str += "\n\n"

        return return_str

    def synchronize_between_processes(self):
        all_dets = misc.all_gather(self.dets)
        all_gts = misc.all_gather(self.gts)

        if misc.is_main_process():
            merged_predictions = {}
            for key in all_dets[0].keys():
                merged_predictions[key] = 0
                for p in all_dets:
                    # Ensure all values are on CPU
                    if isinstance(p[key], torch.Tensor):
                        p[key] = p[key].cpu()
                    merged_predictions[key] += p[key]
            self.dets = merged_predictions

            merged_predictions = {}
            for key in all_gts[0].keys():
                merged_predictions[key] = 0
                for p in all_gts:
                    # Ensure all values are on CPU
                    if isinstance(p[key], torch.Tensor):
                        p[key] = p[key].cpu()
                    merged_predictions[key] += p[key]
            self.gts = merged_predictions

    def evaluate(self, batch_data, end_points, prefix):
        """
        Evaluate all accuracies.

        Args:
            batch_data (dict): contains original data (utterances, meta_path, etc.)
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        self.evaluate_bbox_by_span(batch_data, end_points, prefix)
        self.evaluate_bbox_by_contrast(batch_data, end_points, prefix)

    def evaluate_bbox_by_span(self, batch_data, end_points, prefix):
        """
        Evaluate bounding box IoU for top gt span detections.

        Args:
            batch_data (dict): contains original data (utterances, meta_path, etc.)
            end_points (dict): contains predictions and gt
            prefix (str): layer name
        """
        # Parse gt - NOW USING ROTATED GT FOR FAIR COMPARISON
        positive_map = torch.clone(end_points["positive_map"])
        positive_map[positive_map > 0] = 1
        gt_bboxes_rotated = batch_data["gt_bboxes"]  # (B, 132, 7) or (B, 132, 9) with rotation
        
        # Parse predictions
        sem_scores = end_points[f"{prefix}sem_cls_scores"].softmax(-1)  # B, num_query=256, len_token=256

        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_ = torch.zeros(sem_scores.shape[0], sem_scores.shape[1], positive_map.shape[-1]).to(sem_scores.device)
            sem_scores_[:, :, : sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_

        # Parse predictions
        pred_center = end_points[f"{prefix}center"]  # B, Q, 3
        pred_size = end_points[f"{prefix}pred_size"]  # (B,Q,3) (l,w,h)
        assert (pred_size < 0).sum() == 0
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)  # B, Q=256, 6, each query corresponds to a box

        # Highest scoring box -> iou
        for bid in range(len(positive_map)):
            # Keep scores for annotated objects only
            num_obj = int(end_points["box_label_mask"][bid].sum())  # 1
            pmap = positive_map[bid, :num_obj]
            scores = (sem_scores[bid].unsqueeze(0) * pmap.unsqueeze(1)).sum(-1)  # (1, Q, 256)  # (obj, 1, 256)  # (obj, Q) # Score of each query for target token

            # 10 predictions per gt box
            top = scores.argsort(1, True)[:, :10]  # (obj, 10) # Sort each GT (only 1 here) and get top 10 queries
            pbox = pred_bbox[bid, top.reshape(-1)]  #  # Query indices, sorted by score from high to low

            # IoU - NOW USING ROTATED GT FOR FAIR COMPARISON
            gt_boxes = gt_bboxes_rotated[bid][:num_obj]  # (1, 7) or (1, 9) - with rotation
            ious, _ = iou3d_rotated_vs_aligned(
                gt_boxes,  # (1, 7/9) - rotated GT bbox
                pbox       # (10, 6) - axis-aligned pred bboxes
            )  # returns (1, 10) - IoU between 1 gt and 10 predictions

            # Measure IoU>threshold, ious are (obj, 10)
            topks = self.topks  # [1, 5, 10]
            for t in self.thresholds:  # 0.25, 0.5
                thresholded = ious > t
                for k in topks:
                    found = thresholded[:, :k].any(1)  # Top-1: Check if any of first 1 has IoU > 0.5 # ious[:, :1] = [0.55] > 0.5
                    # NOTE bbs is "Box given span (soft-token)"
                    self.dets[(prefix, t, k, "bbs")] += found.sum().item()  # Number of hit GT boxes
                    self.gts[(prefix, t, k, "bbs")] += len(thresholded)  # Total number of GT boxes

    def evaluate_bbox_by_contrast(self, batch_data, end_points, prefix):
        """
        Evaluate bounding box IoU using contrastive learning (via similarity between query and token features)

        Core idea:
        1. DETR model predicts 256 candidate boxes (set prediction)
        2. Compute contrastive matching score between each candidate and language tokens
        3. Select top-k candidates with highest scores
        4. Compute IoU between these candidates and GT to evaluate accuracy

        Args:
            batch_data (dict): contains original data (utterances, meta_path, etc.)
            end_points (dict): contains model predictions and ground truth
            prefix (str): layer name, e.g., "last_" or "proposal_"
        """
        # ============ 1. Parse Ground Truth ============
        # ipdb.set_trace() # TODO if using batch_data["gt_bboxes"]
        # positive_map, gt_bboxes = self._parse_gt(end_points)
        # Get original GT bboxes from batch_data (with rotation)
        positive_map = torch.clone(end_points["positive_map"])
        positive_map[positive_map > 0] = 1
        gt_bboxes_rotated = batch_data["gt_bboxes"]  # (B, 132, 7) or (B, 132, 9)
        # Waymo: [x,y,z,l,w,h,yaw]  Quad/Drone: [x,y,z,l,w,h,yaw,pitch,roll]

        # positive_map: (B=8, 132, 256) - token positions per GT object (one-hot)
        # gt_bboxes_rotated: (B=8, 132, 7/9) - GT bbox with rotation; first num_obj are valid
        
        # ============ 2. Parse model predictions ============
        pred_center = end_points[f"{prefix}center"]  # (B=8, Q=256, 3) - predicted centers
        pred_size = end_points[f"{prefix}pred_size"]  # (B=8, Q=256, 3) - predicted sizes (l,w,h)
        assert (pred_size < 0).sum() == 0  # ensure sizes are positive
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)  # (B=8, Q=256, 6)
        # DETR: each sample predicts 256 candidate boxes (queries)

        # ============ 3. Compute contrastive scores ============
        proj_tokens = end_points["proj_tokens"]  # (B=8, tokens, 64)
        # Text features: tokens projected to contrastive space (64-d)
        
        proj_queries = end_points[f"{prefix}proj_queries"]  # (B=8, Q=256, 64)
        # Query features: 256 query features projected to contrastive space (64-d)
        
        sem_scores = torch.matmul(
            proj_queries, proj_tokens.transpose(-1, -2)
        )  # (B=8, Q=256, tokens) - similarity of each query with each token (dot product)
        
        sem_scores_ = (sem_scores / 0.07).softmax(-1)  # (B=8, Q=256, tokens)
        # Temperature 0.07: common parameter in contrastive learning
        # softmax: normalize to probability distribution
        
        # Pad to fixed dimension 256
        sem_scores = torch.zeros(sem_scores_.size(0), sem_scores_.size(1), 256)  # (B=8, Q=256, 256)
        sem_scores = sem_scores.to(sem_scores_.device)
        sem_scores[:, : sem_scores_.size(1), : sem_scores_.size(2)] = sem_scores_

        # ============ 4. Evaluate per sample ============
        for bid in range(len(positive_map)):  # iterate over each sample in batch
            # 4.1 Get valid number of GT objects
            num_obj = int(end_points["box_label_mask"][bid].sum())  # how many GTs in current sample
            assert num_obj == 1, f"num_obj: {num_obj}. only support obj number is 1."
            # Currently only single-object settings supported (quad/drone/waymo single target)
            
            # 4.2 Compute the matching score between each query and the target description
            pmap = positive_map[bid, :num_obj]  # (1, 256) - target token positions
            # e.g., for "the red car", pmap marks token positions for "red" and "car"
            
            scores = (sem_scores[bid].unsqueeze(0) * pmap.unsqueeze(1)).sum(-1)  # (1, 256)
            # sem_scores[bid]: (256, 256) - scores of 256 queries against all tokens for the current sample
            # .unsqueeze(0): (1, 256, 256)
            # pmap.unsqueeze(1): (1, 1, 256) - mask of target tokens
            # multiply then sum(-1): keep only scores on target tokens to get per-query total
            # result: (1, 256) - 256 queries' matching scores to the target description

            # 4.3 Select top-10 highest scoring candidates
            top = scores.argsort(1, True)[:, :10]  # (1, 10)
            # argsort(1, True): sort by score descending, return indices
            # [:, :10]: take top 10 indices
            
            pbox = pred_bbox[bid, top.reshape(-1)]  # (10, 6) - [cx,cy,cz,w,h,d] axis-aligned
            # Pick these 10 best-matching predicted boxes from 256 candidates

            # ipdb.set_trace() # TODO print gt_bboxes_rotated[bid][:num_obj]
            # 4.4 Compute IoU
            # ious, _ = _iou3d_par(
            #     box_cxcyczwhd_to_xyzxyz(gt_bboxes[bid][:num_obj]),  # (1, 6) - gt bbox
            #     box_cxcyczwhd_to_xyzxyz(pbox)  # (10, 6) - top-10 predicted boxes
            # )  # returns (1, 10) - IoU between 1 GT and 10 predicted boxes
            
            # 4.4 Compute IoU (rotated GT vs axis-aligned pred)
            gt_boxes = gt_bboxes_rotated[bid][:num_obj]  # (1, 7) or (1, 9) - with rotation
            ious, _ = iou3d_rotated_vs_aligned(
                gt_boxes,  # (1, 7/9) - rotated GT bbox
                pbox       # (10, 6) - axis-aligned pred bboxes
            )  # returns (1, 10) - IoU between 1 GT and 10 predictions
            # Since num_obj==1 (single target), ious shape is already correct

            # 4.5 Record predictions (for later analysis)
            meta_path = batch_data["meta_path"][bid]
            dataset = meta_path.split("/")[-4]
            sequence = meta_path.split("/")[-3]
            frame = meta_path.split("/")[-2]

            record = {
                "id": f"{dataset}/{sequence}/{frame}",
                "utterance": batch_data["utterances"][bid],
                "gt_box": batch_data["gt_bboxes"][bid][:num_obj].cpu().numpy().tolist(), # (1, 7/9) - keep full rotation
                "pred_box": pbox[0].cpu().numpy().tolist(),  # top-1 predicted box
                "ious": ious[:, 0].cpu().numpy().tolist(),  # IoU of top-1
            }
            self.prediction_records.append(record)

            # Accumulate mean IoU (for mIoU)
            self.dets["iou"] += ious[:, 0].cpu().numpy().sum()
            self.dets["num_iou"] += num_obj

            # ============ 5. Compute accuracy metrics ============
            # Iterate different IoU thresholds (0.25, 0.5)
            for t in self.thresholds:
                thresholded = ious > t  # (1, 10) - which predictions exceed threshold
                # e.g., when t=0.25, thresholded = [False, True, True, False, ...]

                # Iterate different top-k (1, 5, 10)
                for k in self.topks:
                    # Check if any of top-k predictions has IoU > threshold
                    found = thresholded[:, :k].any(1)  # (1,) - bool tensor
                    # .any(1): for each GT, check if any among top-k matches
                    # e.g., k=1: only check top-1; k=5: check top-5
                    
                    all_found = found.all().item()  # bool (0 or 1)
                    # .all(): require all GTs to be matched (equals found[0] for single GT)
                    # .item(): convert to Python bool/int

                    # Update statistics
                    # NOTE: bbf = "Box given span (contrastive)"
                    self.dets[(prefix, t, k, "bbf")] += all_found  # success samples +1 or +0
                    self.gts[(prefix, t, k, "bbf")] += 1  # total samples +1

                    # For total_acc calculation only on the last layer and top-1 (for Acc@0.25 printing)
                    if prefix == "last_" and k == 1:
                        self.dets[("total_acc", t, "bbf")] += all_found
                        self.gts[("total_acc", t, "bbf")] += 1

    def _parse_gt(self, end_points):
        positive_map = torch.clone(end_points["positive_map"])  # (B, K, 256)
        positive_map[positive_map > 0] = 1
        gt_center = end_points["center_label"][:, :, 0:3]  # (B, K, 3)
        gt_size = end_points["size_gts"]  # (B, K2,3)
        gt_bboxes = torch.cat([gt_center, gt_size], dim=-1)  # cxcyczwhd
        if self.only_root:  # MARK ony first object if true
            positive_map = positive_map[:, :1]  # (B, 1, 256)
            gt_bboxes = gt_bboxes[:, :1]  # (B, 1, 6)
        return positive_map, gt_bboxes
