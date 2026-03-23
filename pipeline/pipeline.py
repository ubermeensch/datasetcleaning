import yaml
from typing import Dict, Any

from .body_completeness import BodyCompletenessFilter
from .ad_detection      import AdvertisementFilter
from .age_estimation    import AgeEstimationFilter
from .quality_filter    import QualityFilter


class CurationPipeline:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        pc = cfg.get("pipeline", {})

        print("[Pipeline] Loading models…")
        self.quality_filter = QualityFilter(pc.get("quality_filter", {}))
        self.body_filter    = BodyCompletenessFilter(pc.get("body_completeness", {}))
        self.ad_filter      = AdvertisementFilter(pc.get("advertisement_detection", {}))
        self.age_filter     = AgeEstimationFilter(pc.get("age_estimation", {}))
        print("[Pipeline] All models loaded. Ready.")

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Single-image sequential pipeline. For testing/debugging only.
        Production runs use run_pipeline.py which calls each filter directly in batch."""
        image = cv2.imread(image_path)
        if image is None:
            return {"accepted": False, "rejection_reasons": ["invalid_image"],
                    "stages": {}, "path": image_path}

        stages: Dict[str, Any] = {}

        # Stage 0 — quality
        r0 = self.quality_filter.check(image)
        stages["quality"] = r0
        if not r0["passed"]:
            return {"accepted": False, "rejection_reasons": ["quality"],
                    "rejection_detail": r0.get("reason"),
                    "stages": stages, "path": image_path}

        # Stage 1 — body completeness
        r1 = self.body_filter.check(image)
        stages["body_completeness"] = {k: v for k, v in r1.items()
                                        if k != "_raw_keypoints"}
        if not r1["passed"]:
            return {"accepted": False, "rejection_reasons": ["body"],
                    "rejection_detail": r1.get("reason"),
                    "stages": stages, "path": image_path}

        # Stage 1.5 — real human gate (skip if back-facing, trust YOLO instead)
        face_type = r1.get("face_type", "unknown")
        stages["real_human_gate"] = {"face_type": face_type}
        if face_type in ("frontal", "side", "partial"):
            has_face = self.age_filter.has_real_face(image)
            stages["real_human_gate"]["has_real_face"] = has_face
            if not has_face:
                return {"accepted": False,
                        "rejection_reasons": ["not_real_human"],
                        "rejection_detail": "no_real_face_detected",
                        "stages": stages, "path": image_path}

        # Stage 2 — advertisement detection
        r2 = self.ad_filter.check(image)
        stages["advertisement"] = r2
        if not r2["passed"]:
            return {"accepted": False, "rejection_reasons": ["ad"],
                    "rejection_detail": r2.get("reason"),
                    "stages": stages, "path": image_path}

        # Stage 3 — age estimation
        r3 = self.age_filter.check(image, pose_keypoints=r1.get("_raw_keypoints"))
        stages["age_estimation"] = r3
        if not r3["passed"]:
            return {"accepted": False, "rejection_reasons": ["age"],
                    "rejection_detail": r3.get("reason"),
                    "stages": stages, "path": image_path}

        return {"accepted": True, "rejection_reasons": [],
                "stages": stages, "path": image_path}
