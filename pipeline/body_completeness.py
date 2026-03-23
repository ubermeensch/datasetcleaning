import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Any


COCO_KEYPOINTS = {
    0: "nose",       1: "left_eye",    2: "right_eye",
    3: "left_ear",   4: "right_ear",   5: "left_shoulder",
    6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip",
    12: "right_hip", 13: "left_knee",  14: "right_knee",
    15: "left_ankle", 16: "right_ankle",
}


class BodyCompletenessFilter:
    def __init__(self, config: dict):
        self.model          = YOLO(config.get("model", "yolov8n-pose.pt"))
        self.det_model      = YOLO("yolov8n.pt")  # bbox fallback for pose failures
        self.min_kp_conf    = config.get("min_keypoint_confidence", 0.10)
        self.required_upper = config.get("required_upper_kpts", [5, 6])
        self.required_lower = config.get("required_lower_kpts", [11, 12])
        self.boundary_margin = config.get("boundary_margin", 0.02)
        self.min_body_span  = config.get("min_body_span_ratio", 0.25)
        self.min_person_conf = config.get("min_person_confidence", 0.15)

    def _check_face_from_keypoints(self, kps, h) -> Dict[str, Any]:
        nose_conf      = float(kps[0, 2])
        left_eye_conf  = float(kps[1, 2])
        right_eye_conf = float(kps[2, 2])
        left_ear_conf  = float(kps[3, 2])
        right_ear_conf = float(kps[4, 2])

        sees_nose = nose_conf > 0.6
        sees_eye  = (left_eye_conf > 0.6) or (right_eye_conf > 0.6)

        # Sunglasses detection: nose + both ears visible but eyes not detected
        # Both ears required to confirm forward-facing (side view only shows one ear)
        both_ears      = left_ear_conf > 0.2 and right_ear_conf > 0.2
        sunglasses_case = sees_nose and both_ears and not sees_eye
        face_visible   = (sees_nose and sees_eye) or sunglasses_case

        if not face_visible:
            return {"face_visible": False, "face_type": "back_facing"}

        both_eyes = left_eye_conf > 0.5 and right_eye_conf > 0.5
        face_type = "frontal" if both_eyes else "side"
        return {"face_visible": True, "face_type": face_type,
                "is_sunglasses": sunglasses_case}

    def _bbox_fallback_check(self, image: np.ndarray) -> Dict[str, Any]:
        """Fallback when YOLO-pose keypoints are unavailable — uses bbox only."""
        h, w = image.shape[:2]
        results = self.det_model(
            image, verbose=False,
            conf=self.min_person_conf, classes=[0]
        )

        if (not results or results[0].boxes is None
                or len(results[0].boxes) == 0):
            return {"passed": False, "reason": "no_person_detected"}

        boxes    = results[0].boxes.xyxy.cpu().numpy()
        confs    = results[0].boxes.conf.cpu().numpy()
        areas    = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        best_idx = int(areas.argmax())
        x1, y1, x2, y2 = boxes[best_idx]
        bw = x2 - x1
        bh = y2 - y1

        # Reject non-standing detections (wide bounding boxes)
        if bh / (bw + 1e-6) < 1.2:
            return {"passed": False, "reason": "not_full_body_aspect_ratio"}

        if bh / h < self.min_body_span:
            return {"passed": False, "reason": "insufficient_body_span",
                    "span": float(bh / h)}

        # Estimate face region — reject if face is in lower 65% of frame
        face_region_y = (y1 + bh * 0.15) / h
        if face_region_y > 0.65:
            return {"passed": False, "reason": "face_not_visible"}

        return {
            "passed":           True,
            "body_span_ratio":  float(bh / h),
            "face_type":        "unknown",
            "yolo_conf":        float(confs[best_idx]),
            "_raw_keypoints":   None,
            "detection_method": "bbox_fallback",
        }

    def _evaluate_result(self, image: np.ndarray,
                         yolo_result) -> Dict[str, Any]:
        # Fall back to bbox if pose keypoints unavailable
        if (yolo_result.keypoints is None
                or len(yolo_result.keypoints.data) == 0
                or yolo_result.boxes is None
                or len(yolo_result.boxes) == 0):
            return self._bbox_fallback_check(image)

        h, w    = image.shape[:2]
        boxes   = yolo_result.boxes
        areas   = ((boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) *
                   (boxes.xyxy[:, 3] - boxes.xyxy[:, 1]))
        best_idx = int(areas.argmax())
        kps     = yolo_result.keypoints.data[best_idx].cpu().numpy()

        # Require at least one shoulder keypoint
        if not any(kps[k, 2] >= self.min_kp_conf for k in [5, 6]):
            return self._bbox_fallback_check(image)

        # Body span calculated before hip check so it's available for dress rescue
        head_y   = next((kps[i, 1] for i in [0, 1, 2, 5, 6]
                         if kps[i, 2] > self.min_kp_conf), kps[5, 1])
        bottom_y = next((kps[i, 1] for i in [15, 16, 13, 14, 11, 12]
                         if kps[i, 2] > self.min_kp_conf), kps[11, 1])
        body_span = (bottom_y - head_y) / h

        # Hip check — rescue if dress/skirt occludes hips but body span is sufficient
        hip_visible = any(kps[k, 2] >= self.min_kp_conf for k in [11, 12])
        if not hip_visible:
            both_shoulders = all(kps[k, 2] >= self.min_kp_conf for k in [5, 6])
            if not (both_shoulders and body_span >= self.min_body_span):
                return self._bbox_fallback_check(image)

        if body_span < self.min_body_span:
            return self._bbox_fallback_check(image)

        # Reject back-facing subjects
        face_result = self._check_face_from_keypoints(kps, h)
        if not face_result["face_visible"]:
            return {"passed": False, "reason": "face_not_visible"}

        return {
            "passed":          True,
            "body_span_ratio": float(body_span),
            "face_type":       face_result["face_type"],
            "is_sunglasses":   face_result.get("is_sunglasses", False),
            "yolo_conf":       float(boxes.conf[best_idx].cpu()),
            "_raw_keypoints":  kps,
            "keypoint_confidences": {
                COCO_KEYPOINTS[i]: float(kps[i, 2]) for i in range(17)
            },
        }

    def check(self, image: np.ndarray) -> Dict[str, Any]:
        """Single-image check. Used by process_image in pipeline.py."""
        results = self.model(image, verbose=False, conf=self.min_person_conf)
        if not results:
            return self._bbox_fallback_check(image)
        return self._evaluate_result(image, results[0])
