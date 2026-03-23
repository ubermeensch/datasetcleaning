import cv2
import numpy as np
from insightface.app import FaceAnalysis
from typing import Dict, Any, Optional


class AgeEstimationFilter:
    def __init__(self, config: dict):
        self.min_age       = config.get("min_acceptable_age", 13)
        self.buffer        = config.get("age_uncertainty_buffer", 3)
        self.effective_min = self.min_age - self.buffer

        providers  = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.app   = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0, det_size=(320, 320))

    def has_real_face(self, image: np.ndarray,
                      yolo_conf: float = 0.0,
                      n_visible_keypoints: int = 0) -> bool:
        """
        Verify a real human face is present.
        Two-pass: high threshold first, then lower threshold for occluded faces.
        YOLO bypass only when both confidence and keypoint count are high —
        mannequins rarely produce many confident pose keypoints.
        """
        rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)

        if len(faces) > 0:
            best_score = max(float(f.det_score) for f in faces)
            if best_score >= 0.75:
                return True

        # Retry at lower threshold for sunglasses, masks, partial occlusion
        try:
            self.app.det_model.det_thresh = 0.20
            faces = self.app.get(rgb)
            self.app.det_model.det_thresh = 0.50
            if len(faces) > 0:
                best_score = max(float(f.det_score) for f in faces)
                if best_score >= 0.55:
                    return True
        except Exception:
            pass

        # YOLO bypass as last resort — requires both high confidence + many keypoints
        if yolo_conf > 0.75 and n_visible_keypoints >= 6:
            return True

        return False

    def _head_body_ratio_check(self,
                                keypoints: Optional[np.ndarray]) -> Optional[bool]:
        """
        Estimate age group from head-to-body ratio using ear and ankle keypoints.
        Returns True (adult), False (child), or None (insufficient keypoints).
        """
        if keypoints is None:
            return None
        l_ear, r_ear = keypoints[3], keypoints[4]
        if l_ear[2] < 0.3 or r_ear[2] < 0.3:
            return None
        head_width = abs(float(r_ear[0]) - float(l_ear[0]))
        if head_width < 1.0:
            return None
        nose_y  = float(keypoints[0, 1])
        ankle_y = max(float(keypoints[15, 1]), float(keypoints[16, 1]))
        ratio   = (ankle_y - nose_y) / head_width
        if ratio < 4.5:
            return False   # child proportions
        if ratio > 5.5:
            return True    # adult proportions
        return None        # ambiguous

    def check(self, image: np.ndarray,
              pose_keypoints: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Estimate age via InsightFace. Falls back to head-body ratio
        if no face is detected (e.g. back-facing, heavy occlusion).
        """
        rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)

        if not faces:
            # No face detected — use pose proportions as fallback
            adult_likely = self._head_body_ratio_check(pose_keypoints)
            if adult_likely is False:
                return {"passed": False, "reason": "child_body_proportions",
                        "age_estimated": None}
            return {"passed": True, "age_estimated": None,
                    "note": "no_face_detected_passed_on_proportion"}

        # Use largest detected face for age estimation
        face          = max(faces, key=lambda f:
                            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        estimated_age = int(face.age)
        is_child      = estimated_age < self.effective_min

        return {
            "passed":        not is_child,
            "reason":        "child_detected" if is_child else None,
            "age_estimated": estimated_age,
        }
