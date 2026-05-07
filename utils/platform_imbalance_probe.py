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
        self._warned_missing_label = False
        self._csv_ready = False

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

    def _write_result(self, epoch, batch_idx, global_step, platform_id, loss_value, count):
        if not self._is_main_process:
            return
        platform_name = self._platform_name(platform_id)
        self._ensure_csv()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, batch_idx, global_step, platform_id, platform_name, loss_value, count])

        if self.tb_writer is not None:
            self.tb_writer.add_scalar(f"PlatformProbe/train_loss/{platform_name}", loss_value, global_step)
            self.tb_writer.add_scalar(f"PlatformProbe/train_count/{platform_name}", count, global_step)

        if self.logger is not None:
            self.logger.info(
                f"PlatformProbe: epoch {epoch} batch {batch_idx} "
                f"platform {platform_name}({platform_id}) loss {loss_value:.4f} count {count}"
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
                    loss, _ = compute_loss_fn(end_points, criterion, set_criterion, args)
                    self._write_result(epoch, batch_idx, global_step, platform_id, float(loss.detach().item()), count)
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
