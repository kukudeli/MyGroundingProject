import csv
import os

import torch
import torch.distributed as dist


class PlatformImbalanceProbe:
    """Collect per-platform train-loss diagnostics without affecting training."""

    DEFAULT_PLATFORM_NAMES = ("waymo", "drone", "quad")

    def __init__(self, logger, tb_writer, log_dir, platform_names=None, freq=100):
        self.logger = logger
        self.tb_writer = tb_writer
        self.log_dir = log_dir
        self.platform_names = list(platform_names or self.DEFAULT_PLATFORM_NAMES)
        self.freq = freq
        self.csv_path = os.path.join(log_dir, "platform_probe_train_loss.csv")
        self.metrics_csv_path = os.path.join(log_dir, "platform_probe_train_metrics.csv")
        self._warned_missing_label = False
        self._warned_missing_box_loss = False
        self._csv_ready = False
        self._metrics_csv_ready = False

    @staticmethod
    def _rank():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @property
    def _is_main_process(self):
        return self._rank() == 0

    def _platform_name(self, platform_id):
        if 0 <= platform_id < len(self.platform_names):
            return self.platform_names[platform_id]
        return str(platform_id)

    def _ensure_csv(self):
        if not self._is_main_process or self._csv_ready:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        need_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        if need_header:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "batch_idx", "global_step", "platform_id", "platform_name", "loss", "count"])
        self._csv_ready = True

    def _ensure_metrics_csv(self):
        if not self._is_main_process or self._metrics_csv_ready:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        need_header = not os.path.exists(self.metrics_csv_path) or os.path.getsize(self.metrics_csv_path) == 0
        if need_header:
            with open(self.metrics_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "batch_idx", "global_step", "platform_id", "platform_name", "total_loss", "box_loss", "count"])
        self._metrics_csv_ready = True

    @staticmethod
    def _slice_batch(batch_data, mask):
        sliced = {}
        batch_size = int(mask.shape[0])
        list_indices = mask.detach().cpu().nonzero(as_tuple=False).view(-1).tolist()
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
                sliced[key] = value[mask.to(value.device)]
            elif isinstance(value, list) and len(value) == batch_size:
                sliced[key] = [value[i] for i in list_indices]
            else:
                sliced[key] = value
        return sliced

    def _extract_box_loss(self, end_points):
        if "loss_bbox" in end_points or "loss_giou" in end_points:
            parts = []
            for key in ("loss_bbox", "loss_giou"):
                if key in end_points:
                    value = end_points[key]
                    parts.append(value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(float(value)))
            if parts:
                device = parts[0].device
                return torch.stack([part.to(device).float().reshape(()) for part in parts]).sum()

        parts = []
        for key, value in end_points.items():
            key_lower = key.lower()
            is_box_key = key_lower.endswith("loss_bbox") or key_lower.endswith("loss_giou")
            if not is_box_key:
                continue
            if any(skip in key_lower for skip in ("contrast", "align", "cls", "class", "objectness", "token")):
                continue
            parts.append(value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(float(value)))
        if parts:
            device = parts[0].device
            return torch.stack([part.to(device).float().reshape(()) for part in parts]).sum()

        if self._is_main_process and not self._warned_missing_box_loss and self.logger is not None:
            self.logger.warning("PlatformProbe: no box loss key found in end_points; box_loss will be empty.")
        self._warned_missing_box_loss = True
        return None

    def _write_result(self, epoch, batch_idx, global_step, platform_id, total_loss_value, box_loss_value, count):
        if not self._is_main_process:
            return
        platform_name = self._platform_name(platform_id)
        self._ensure_csv()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, batch_idx, global_step, platform_id, platform_name, total_loss_value, count])

        self._ensure_metrics_csv()
        with open(self.metrics_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            box_cell = "" if box_loss_value is None else box_loss_value
            writer.writerow([epoch, batch_idx, global_step, platform_id, platform_name, total_loss_value, box_cell, count])

        if self.tb_writer is not None:
            self.tb_writer.add_scalar(f"PlatformProbe/train_total_loss/{platform_name}", total_loss_value, global_step)
            if box_loss_value is not None:
                self.tb_writer.add_scalar(f"PlatformProbe/train_box_loss/{platform_name}", box_loss_value, global_step)
            self.tb_writer.add_scalar(f"PlatformProbe/train_count/{platform_name}", count, global_step)

        if self.logger is not None:
            box_msg = "nan" if box_loss_value is None else f"{box_loss_value:.4f}"
            self.logger.info(
                f"PlatformProbe: epoch {epoch} batch {batch_idx} "
                f"platform {platform_name}({platform_id}) total_loss {total_loss_value:.4f} "
                f"box_loss {box_msg} count {count}"
            )

    @staticmethod
    def _clone_module_state(module):
        if module is None:
            return None
        return {key: value.detach().clone() for key, value in module.state_dict().items()}

    @staticmethod
    def _restore_module_state(module, state):
        if module is not None and state is not None:
            module.load_state_dict(state, strict=True)

    def log_train_platform_loss(
        self,
        epoch,
        batch_idx,
        global_step,
        batch_data,
        model,
        criterion,
        set_criterion,
        compute_loss_fn,
        get_inputs_fn,
        args,
    ):
        if self.freq <= 0 or batch_idx % self.freq != 0:
            return
        if "platform_label" not in batch_data:
            if self._is_main_process and not self._warned_missing_label and self.logger is not None:
                self.logger.warning("PlatformProbe skipped: batch_data has no platform_label.")
            self._warned_missing_label = True
            return

        platform_labels = batch_data["platform_label"]
        if not isinstance(platform_labels, torch.Tensor):
            platform_labels = torch.as_tensor(platform_labels)
        platform_labels = platform_labels.long()

        was_training = model.training
        criterion_was_training = set_criterion.training if set_criterion is not None else False
        criterion_state = self._clone_module_state(set_criterion)

        try:
            model.eval()
            if set_criterion is not None and criterion_was_training:
                set_criterion.train()
            with torch.no_grad():
                for platform in platform_labels.detach().cpu().unique():
                    platform_id = int(platform.item())
                    mask = platform_labels == platform_id
                    count = int(mask.sum().item())
                    if count == 0:
                        continue

                    self._restore_module_state(set_criterion, criterion_state)
                    sub_batch = self._slice_batch(batch_data, mask)
                    inputs = get_inputs_fn(sub_batch)
                    end_points = model(inputs)
                    for key, value in sub_batch.items():
                        if key not in end_points:
                            end_points[key] = value
                    end_points["epoch"] = epoch
                    loss, sub_end_points = compute_loss_fn(end_points, criterion, set_criterion, args)
                    box_loss = self._extract_box_loss(sub_end_points)
                    box_loss_value = None if box_loss is None else float(box_loss.detach().item())
                    self._write_result(epoch, batch_idx, global_step, platform_id, float(loss.detach().item()), box_loss_value, count)
        finally:
            self._restore_module_state(set_criterion, criterion_state)
            if was_training:
                model.train()
            else:
                model.eval()
            if set_criterion is not None:
                if criterion_was_training:
                    set_criterion.train()
                else:
                    set_criterion.eval()


class PlatformValidationRecorder:
    """Accumulate per-platform validation Acc@25, Acc@50 and mIoU."""

    DEFAULT_PLATFORM_NAMES = ("waymo", "drone", "quad")

    def __init__(self, logger, log_dir, platform_names=None):
        self.logger = logger
        self.log_dir = log_dir
        self.platform_names = list(platform_names or self.DEFAULT_PLATFORM_NAMES)
        self.csv_path = os.path.join(log_dir, "platform_probe_val_metrics.csv")
        self._warned_missing_label = False
        self._warned_missing_metrics = False
        self.reset()

    @staticmethod
    def _rank():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    @property
    def _is_main_process(self):
        return self._rank() == 0

    def reset(self):
        self.stats = {}

    def _platform_name(self, platform_id):
        if 0 <= platform_id < len(self.platform_names):
            return self.platform_names[platform_id]
        return str(platform_id)

    def _get_platform_stats(self, platform_id):
        if platform_id not in self.stats:
            self.stats[platform_id] = {"count": 0.0, "sum_iou": 0.0, "correct_25": 0.0, "correct_50": 0.0}
        return self.stats[platform_id]

    def _warn_once(self, attr, message):
        if self._is_main_process and not getattr(self, attr) and self.logger is not None:
            self.logger.warning(message)
        setattr(self, attr, True)

    def update(self, epoch, batch_data, end_points):
        if "platform_label" not in batch_data:
            self._warn_once("_warned_missing_label", "PlatformValidationRecorder skipped: batch_data has no platform_label.")
            return

        required = ("iou_per_sample", "acc25_per_sample", "acc50_per_sample")
        missing = [key for key in required if key not in end_points]
        if missing:
            self._warn_once("_warned_missing_metrics", f"PlatformValidationRecorder skipped: missing end_points keys {missing}.")
            return

        platform_labels = batch_data["platform_label"]
        if not isinstance(platform_labels, torch.Tensor):
            platform_labels = torch.as_tensor(platform_labels)
        platform_labels = platform_labels.detach().cpu().long()
        ious = end_points["iou_per_sample"].detach().cpu().float()
        acc25 = end_points["acc25_per_sample"].detach().cpu().float()
        acc50 = end_points["acc50_per_sample"].detach().cpu().float()

        for platform in platform_labels.unique():
            platform_id = int(platform.item())
            mask = platform_labels == platform_id
            count = int(mask.sum().item())
            if count == 0:
                continue
            platform_stats = self._get_platform_stats(platform_id)
            platform_stats["count"] += float(count)
            platform_stats["sum_iou"] += float(ious[mask].sum().item())
            platform_stats["correct_25"] += float(acc25[mask].sum().item())
            platform_stats["correct_50"] += float(acc50[mask].sum().item())

    def _sync_stats(self):
        if not (dist.is_available() and dist.is_initialized()):
            return self.stats

        max_platforms = len(self.platform_names)
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        tensor = torch.zeros(max_platforms, 4, device=device)
        for platform_id, values in self.stats.items():
            if platform_id >= max_platforms:
                continue
            tensor[platform_id, 0] = values["count"]
            tensor[platform_id, 1] = values["sum_iou"]
            tensor[platform_id, 2] = values["correct_25"]
            tensor[platform_id, 3] = values["correct_50"]
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        synced = {}
        tensor = tensor.cpu()
        for platform_id in range(max_platforms):
            count = float(tensor[platform_id, 0].item())
            if count <= 0:
                continue
            synced[platform_id] = {
                "count": count,
                "sum_iou": float(tensor[platform_id, 1].item()),
                "correct_25": float(tensor[platform_id, 2].item()),
                "correct_50": float(tensor[platform_id, 3].item()),
            }
        return synced

    def finalize(self, epoch):
        synced_stats = self._sync_stats()
        if not self._is_main_process:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        need_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(["epoch", "platform_id", "platform_name", "acc25", "acc50", "miou", "count"])
            for platform_id in sorted(synced_stats):
                values = synced_stats[platform_id]
                count = values["count"]
                if count <= 0:
                    continue
                # Acc values are percentages to match common Acc@25 / Acc@50 logging conventions.
                acc25 = values["correct_25"] / count * 100.0
                acc50 = values["correct_50"] / count * 100.0
                miou = values["sum_iou"] / count
                writer.writerow([epoch, platform_id, self._platform_name(platform_id), acc25, acc50, miou, int(count)])
                if self.logger is not None:
                    self.logger.info(
                        f"PlatformVal: epoch {epoch} platform {self._platform_name(platform_id)}({platform_id}) "
                        f"Acc@25 {acc25:.2f} Acc@50 {acc50:.2f} mIoU {miou:.4f} count {int(count)}"
                    )
