# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import DistMetrics, batch_probiou


class DistValidator(DetectionValidator):
    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "dist"
        self.metrics = DistMetrics()

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        self.confusion_matrix.task = "dist"

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        iou = batch_probiou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def postprocess(self, preds: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        preds = super().postprocess(preds)
        for pred in preds:
            pred["bboxes"] = torch.cat([pred["bboxes"], pred.pop("extra")], dim=-1)  # concatenate angle
        return preds

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def plot_predictions(self, batch: dict[str, Any], preds: list[torch.Tensor], ni: int) -> None:
        for p in preds:
            # TODO: fix this duplicated `xywh2xyxy`
            p["bboxes"][:, :4] = ops.xywh2xyxy(p["bboxes"][:, :4])  # convert to xyxy format for plotting
        super().plot_predictions(batch, preds, ni)  # plot bboxes

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            **predn,
            "bboxes": ops.scale_boxes(
                pbatch["imgsz"], predn["bboxes"].clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True
            ),
        }
