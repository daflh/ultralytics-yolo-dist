# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any
from copy import copy
from pathlib import Path

from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.nn.tasks import DistModel
from ultralytics.utils import DEFAULT_CFG, RANK


class DistTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: list[Any] | None = None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "dist"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(
        self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True
    ) -> DistModel:
        model = DistModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        self.loss_names = "box_loss", "dist_loss", "cls_loss", "dfl_loss"
        return DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
