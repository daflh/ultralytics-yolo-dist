# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import DistMetrics, batch_probiou
from ultralytics.utils.nms import TorchNMS


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
        dists = batch["distances"][batch["batch_idx"] == si]
        dists = dists.clone()
        pbatch["distances"] = dists
        return pbatch

    # TODO: needs review
    # def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    #     if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
    #         return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
    #     iou = batch_probiou(batch["bboxes"], preds["bboxes"])
    #     return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        preds = super().postprocess(preds)
        for pred in preds:
            pred["distances"] = pred.pop("extra").view(-1, 1)  # remove extra if exists
        return preds
    
    # def get_desc(self) -> str:
    #     return ("%22s" + "%11s" * 10) % (
    #         "Class",
    #         "Images",
    #         "Instances",
    #         "Box(P",
    #         "R",
    #         "mAP50",
    #         "mAP50-95)",
    #         "Dist(MAE",
    #         "MRE)",
    #     )

    def plot_predictions(self, batch: dict[str, Any], preds: list[torch.Tensor], ni: int) -> None:
        for p in preds:
            # TODO: fix this duplicated `xywh2xyxy`
            p["bboxes"][:, :4] = ops.xywh2xyxy(p["bboxes"][:, :4])  # convert to xyxy format for plotting
        super().plot_predictions(batch, preds, ni)  # plot bboxes
    