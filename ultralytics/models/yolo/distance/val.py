# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import DistMetrics, box_iou


class DistValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "dist"
        self.metrics = DistMetrics()

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().preprocess(batch)
        batch["distances"] = batch["distances"].float()
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self.confusion_matrix.task = "dist"

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        pbatch = super()._prepare_batch(si, batch)
        idx = batch["batch_idx"] == si
        pbatch["distances"] = batch["distances"][idx]
        return pbatch

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        n_pred = preds["cls"].shape[0]

        # Default outputs
        tp = np.zeros((n_pred, self.niou), dtype=bool)
        pred_dist = np.full((n_pred,), np.nan, dtype=float)
        target_dist = np.full((n_pred,), np.nan, dtype=float)
        target_cls_pred = np.full((n_pred,), -1, dtype=int)

        # Nothing to match
        if batch["cls"].shape[0] == 0 or n_pred == 0:
            return {
                "tp": tp,
                "pred_dist": pred_dist,
                "target_dist": target_dist,
                "target_cls_pred": target_cls_pred
            }

        # Detection matching
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        tp_t = self.match_predictions(preds["cls"], batch["cls"], iou)
        tp = tp_t.cpu().numpy()

        # Distance predicitions
        pred_d = preds.get("distances")
        if pred_d is not None:
            pred_d = pred_d.view(-1).cpu().numpy().astype(float)
        else:
            pred_d = np.zeros((n_pred,), dtype=float)

        # whether to use Euclidean distance or z-axis/forward distance
        use_euclidean = self.args.get("use_euclidean", False)
        # Ground truth distances
        tgt_d = batch.get("distances")
        if tgt_d is not None and tgt_d.shape[0]:
            if use_euclidean:
                tgt_d = torch.linalg.vector_norm(tgt_d, dim=1)
            else:
                tgt_d = tgt_d[:, -1]
            tgt_d = tgt_d.cpu().numpy().astype(float)
        else:
            tgt_d = np.zeros((batch["cls"].shape[0],), dtype=float)

        # assign distances only for TP@0.5, index 0 corresponds to IoU = 0.5 in Ultralytics
        tp_05 = tp[:, 0]
        pred_idx = np.where(tp_05)[0]

        if pred_idx.size:
            # recover GT indices using the same IoU + class constraint
            gt_cls = batch["cls"].cpu().numpy()
            pred_cls = preds["cls"].cpu().numpy()
            iou_np = iou.cpu().numpy()

            for pi in pred_idx:
                gi = np.argmax(
                    (pred_cls[pi] == gt_cls) * iou_np[:, pi]
                )
                pred_dist[pi] = pred_d[pi]
                target_dist[pi] = tgt_d[gi]
                target_cls_pred[pi] = gt_cls[gi]

        # scale back prediction distance to original
        max_dist = self.args.get("max_dist", 100.0)
        pred_dist *= max_dist

        return {
            "tp": tp,
            "pred_dist": pred_dist,
            "target_dist": target_dist,
            "target_cls_pred": target_cls_pred
        }

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        preds = super().postprocess(preds)
        for pred in preds:
            pred["distances"] = pred.pop("extra").view(-1, 1)  # remove extra if exists
        return preds
    
    def get_desc(self) -> str:
        return ("%22s" + "%11s" * 8) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Dist(MAE",
            "MRE)",
        )
    
    def print_results(self) -> None:
        super().print_results()
        if not self.training:
            self.metrics.print_dist_metrics()
    
    def plot_predictions(self, batch: dict[str, Any], preds: list[torch.Tensor], ni: int) -> None:
        for p in preds:
            p["bboxes"][:, :4] = ops.xywh2xyxy(p["bboxes"][:, :4])  # convert to xyxy format for plotting
        super().plot_predictions(batch, preds, ni)  # plot bboxes
    