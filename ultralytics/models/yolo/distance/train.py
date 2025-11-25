# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any
from copy import copy
from pathlib import Path
import numpy as np
import torch

from ultralytics.models import yolo
from ultralytics.nn.tasks import DistModel
from ultralytics.utils import DEFAULT_CFG, RANK


class DistTrainer(yolo.detect.DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: list[Any] | None = None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "dist"
        super().__init__(cfg, overrides, _callbacks)

    def calc_geometric_coeffs(self):
        max_dist = self.args.get("max_dist", 100.0)
        imgsz = self.args.get("imgsz", 640)

        dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        labels = dataset.get_labels()
        bh_dz_pairs = []

        for label in labels:
            h, w = label["shape"]
            gain = min(imgsz / h, imgsz / w) # bbox resize gain

            for cls, bbox, dist in zip(label["cls"], label["bboxes"], label["distances"]):
                cls = cls.item()

                bh = bbox[3].item()
                bh_px = bh * h # unnormalize
                bh_px = bh_px * gain # resize to model input size (normally 640x640)

                d_z = dist[2].item()
                d_z = d_z / max_dist # normalize distance
                # TODO: euclidean distance
                # d_euc = (dx**2 + dy**2 + dz**2)**0.5

                bh_dz_pairs.append([ cls, bh_px, d_z ])
        
        bh_dz_pairs = np.array(bh_dz_pairs)

        bh = bh_dz_pairs[:, 1]
        bh = np.maximum(bh, 1e-6) # avoid division by zero
        inv_bh = 1.0 / bh
        dz = bh_dz_pairs[:, 2]

        # Geometric distance params
        geoa = torch.ones(self.data["nc"])
        geob = torch.zeros(self.data["nc"])

        for ci in self.data["names"].keys():
            if not np.any(bh_dz_pairs[:, 0] == ci):
                continue

            a, b = np.polyfit(inv_bh[bh_dz_pairs[:, 0] == ci], dz[bh_dz_pairs[:, 0] == ci], 1)
            # print(f"Class {ci} inverse model:  d_z ≈ {a:.6f} * (1/bh) + {b:.6f}")

            geoa[ci] = a
            geob[ci] = b

        # Set the calculated geometric coefficients in the model head and freeze them
        for k, v in self.model.named_parameters():
            # this two layers will be frozen during training, see BaseTrainer._setup_train()
            if ".geoa" in k:
                v.data[:] = geoa
            if ".geob" in k:
                v.data[:] = geob

    def setup_model(self):
        ckpt = yolo.detect.DetectionTrainer.setup_model(self)
        self.calc_geometric_coeffs()
        return ckpt

    def get_model(
        self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True
    ) -> DistModel:
        model = DistModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        self.loss_names = "box_loss", "dist_loss", "cls_loss", "dfl_loss"
        return yolo.distance.DistValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
