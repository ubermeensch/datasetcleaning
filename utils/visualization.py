import cv2
import numpy as np

def draw_result(image: np.ndarray, result: dict) -> np.ndarray:
    vis      = image.copy()
    h, w     = vis.shape[:2]
    accepted = result.get("accepted", False)
    color    = (0, 200, 0) if accepted else (0, 0, 220)

    cv2.rectangle(vis, (3, 3), (w - 3, h - 3), color, 4)

    label = "ACCEPTED" if accepted else f"REJECT: {result.get('rejection_detail', '')}"
    cv2.putText(vis, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2, cv2.LINE_AA)

    if accepted:
        y = 45
        for k, v in [("face", result.get("face_type", "-")),
                     ("age",  result.get("age_estimated", "?"))]:
            cv2.putText(vis, f"{k}: {v}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 18
    return vis
