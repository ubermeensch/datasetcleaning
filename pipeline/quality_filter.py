import cv2
import numpy as np
from typing import Dict, Any


class QualityFilter:
    def __init__(self, config: dict):
        self.min_blur_variance = config.get("min_blur_variance", 0.0)  # 0 = disabled
        self.min_width         = config.get("min_width", 20)
        self.min_height        = config.get("min_height", 40)

    def check(self, image: np.ndarray) -> Dict[str, Any]:
        h, w = image.shape[:2]

        # Reject images too small to contain meaningful content
        if w < self.min_width or h < self.min_height:
            return {"passed": False, "reason": "image_too_small", "size": (w, h)}

        # Laplacian variance blur check — skipped if threshold is 0
        if self.min_blur_variance > 0:
            gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if variance < self.min_blur_variance:
                return {"passed": False, "reason": "image_too_blurry",
                        "blur_variance": variance}

        return {"passed": True}
