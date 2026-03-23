from pathlib import Path
import cv2
import numpy as np
from typing import List, Generator

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def collect_images(root_dir: str, recursive: bool = True) -> List[Path]:
    root = Path(root_dir)
    pattern = "**/*" if recursive else "*"
    return [p for p in root.glob(pattern) if p.suffix.lower() in SUPPORTED]

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot load: {path}")
    return img

def batched(lst: list, n: int) -> Generator:
    for i in range(0, len(lst), n):
        yield lst[i: i + n]
