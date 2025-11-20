# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


class DistPredictor(DetectionPredictor):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "dist"

    def construct_result(self, pred, img, orig_img, img_path):
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        # bring back to real-life distance
        pred[:, 6] *= self.model.model.args["max_dist"] if hasattr(self.model.model.args, "max_dist") else 100.0
        return Results(
            orig_img,
            path=img_path,
            names=self.model.names,
            boxes=pred[:, :6],
            distances=pred[:, 6] if pred.shape[1] > 6 else None,
        )
