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
        self.metrics.max_dist = self.args.get("max_dist", 100.0)

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
        # No preds or no targets -> return defaults
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            n_pred = preds["cls"].shape[0]
            return {
                "tp": np.zeros((n_pred, self.niou), dtype=bool),
                "pred_dist": np.zeros((n_pred,), dtype=float),
                "target_dist": np.zeros((n_pred,), dtype=float),
                "target_cls_pred": np.full((n_pred,), -1, dtype=int)
            }

        # IoU / similarity between GT and predictions (N_gt, N_pred).
        # Ensure IoU is a torch tensor on the same device as predictions for match_predictions.
        pred_bboxes = preds["bboxes"]
        gt_bboxes = batch["bboxes"].to(pred_bboxes.device)
        pred_cls = preds["cls"].to(pred_bboxes.device)
        gt_cls = batch["cls"].to(pred_bboxes.device)

        iou_t = box_iou(gt_bboxes, pred_bboxes).to(pred_bboxes.device)
        # Build tp (same as detection) for mAP stats using a torch tensor on correct device
        tp = self.match_predictions(pred_cls, gt_cls, iou_t).cpu().numpy()

        # Prepare distance arrays (z component only)
        # preds['distances'] expected shape (N_pred, D) or (N_pred,1)
        pred_d = preds.get("distances")
        if pred_d is None:
            pred_z = np.zeros((preds["cls"].shape[0],), dtype=float)
        else:
            pred_z = pred_d[:, -1].cpu().numpy().astype(float).reshape(-1)

        # targets distances per GT (N_gt, D)
        tgt_d = batch.get("distances")
        if tgt_d is None or tgt_d.shape[0] == 0:
            tgt_z = np.zeros((batch["cls"].shape[0],), dtype=float)
        else:
            tgt_z = tgt_d[:, -1].cpu().numpy().astype(float).reshape(-1)

        # Determine matching pairs at IoU >= 0.5 and correct class
        # Use same matching logic as BaseValidator.match_predictions for threshold 0.5
        iou_np = iou_t.cpu().numpy()
        thr = 0.5
        matches = np.nonzero(iou_np >= thr)
        if matches[0].size:
            # matches is (label_idx, pred_idx) pairs; resolve duplicates similar to match_predictions
            pairs = np.array(matches).T
            # sort by iou desc
            order = iou_np[pairs[:, 0], pairs[:, 1]].argsort()[::-1]
            pairs = pairs[order]
            # unique by pred index and label index to ensure one-to-one
            _, idx_unique_pred = np.unique(pairs[:, 1], return_index=True)
            pairs = pairs[np.sort(idx_unique_pred)]
            _, idx_unique_label = np.unique(pairs[:, 0], return_index=True)
            pairs = pairs[np.sort(idx_unique_label)]
            label_idx = pairs[:, 0]
            pred_idx = pairs[:, 1]
        else:
            label_idx = np.array([], dtype=int)
            pred_idx = np.array([], dtype=int)

        # Build arrays of length N_pred with zeros, fill matched positions
        n_pred = preds["cls"].shape[0]
        pred_dist_arr = np.zeros((n_pred,), dtype=float)
        target_dist_arr = np.zeros((n_pred,), dtype=float)
        target_cls_pred = np.full((n_pred,), -1, dtype=int)
        if pred_idx.size:
            pred_dist_arr[pred_idx] = pred_z[pred_idx]
            target_dist_arr[pred_idx] = tgt_z[label_idx]
            # store the matched GT class for each matched prediction (per-pred target class)
            # gt_cls is a torch tensor on pred device; move to cpu numpy
            gt_cls_np = gt_cls.cpu().numpy().astype(int)
            target_cls_pred[pred_idx] = gt_cls_np[label_idx]

        # scale back to real-life distance
        max_dist = self.args.get("max_dist", 100.0)
        pred_dist_arr *= max_dist

        return {
            "tp": tp,
            "pred_dist": pred_dist_arr,
            "target_dist": target_dist_arr,
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
        self.metrics.print_dist_metrics()
    
    def plot_predictions(self, batch: dict[str, Any], preds: list[torch.Tensor], ni: int) -> None:
        for p in preds:
            p["bboxes"][:, :4] = ops.xywh2xyxy(p["bboxes"][:, :4])  # convert to xyxy format for plotting
        super().plot_predictions(batch, preds, ni)  # plot bboxes
    