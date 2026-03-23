
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from pipeline import CurationPipeline
from utils.image_utils import collect_images
from utils.visualization import draw_result


def ensure_gdown():
    """Install gdown if not present."""
    try:
        import gdown
        return gdown
    except ImportError:
        print("[Download] gdown not found — installing…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown
        return gdown


def download_dataset(gdrive_url: str, dest_dir: str) -> Path:
    """Download dataset from Google Drive, skip if already exists."""
    dest = Path(dest_dir)
    if dest.exists() and any(dest.iterdir()):
        print(f"[Download] Already exists at {dest} — skipping download.")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    gdown = ensure_gdown()
    print(f"[Download] Downloading → {dest}")
    try:
        gdown.download_folder(url=gdrive_url, output=str(dest),
                              quiet=False, use_cookies=False)
    except Exception as e:
        print(f"[Download] Retrying… ({e})")
        gdown.download_folder(url=gdrive_url, output=str(dest),
                              quiet=False, use_cookies=False, remaining_ok=True)
    image_count = sum(len(list(dest.rglob(f"*{ext}")))
                      for ext in [".jpg", ".jpeg", ".png"])
    print(f"[Download] Done. Found ~{image_count} images.")
    return dest


def _handle_rejected(img, path, result, args):
    """Copy rejected image to rejected_dir if configured."""
    if args.rejected_dir:
        stage = result.get("rejection_reasons", ["unk"])[0]
        dest = Path(args.rejected_dir) / f"{stage}_{Path(path).name}"
        shutil.copy2(path, dest)


def main():
    # ── CLI arguments ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--gdrive_url")
    src.add_argument("--input_dir")
    parser.add_argument("--download_dir",  default="./noisy_dataset")
    parser.add_argument("--output_dir",    required=True)
    parser.add_argument("--config",        default="config/config.yaml")
    parser.add_argument("--batch_size",    type=int, default=32)
    parser.add_argument("--rejected_dir",  default=None)
    parser.add_argument("--visualize",     action="store_true")
    args = parser.parse_args()

    # ── Resolve input ─────────────────────────────────────────────────────────
    if args.gdrive_url:
        input_path = download_dataset(args.gdrive_url, args.download_dir)
    else:
        input_path = Path(args.input_dir)
        if not input_path.exists():
            print(f"[Error] {input_path} does not exist.")
            sys.exit(1)

    # ── Create output dirs ────────────────────────────────────────────────────
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if args.rejected_dir:
        Path(args.rejected_dir).mkdir(parents=True, exist_ok=True)
    if args.visualize:
        (output_path / "visualizations").mkdir(exist_ok=True)

    # ── Load pipeline + images ────────────────────────────────────────────────
    pipeline = CurationPipeline(args.config)
    image_files = collect_images(str(input_path))
    print(f"[Pipeline] Found {len(image_files)} images")
    if not image_files:
        print("[Error] No images found.")
        sys.exit(1)

    all_results = {}
    counts = {
        "total":              len(image_files),
        "accepted":           0,
        "rejected_quality":   0,
        "rejected_body":      0,
        "rejected_not_human": 0,
        "rejected_ad":        0,
        "rejected_age":       0,
        "invalid":            0,
    }

    B = args.batch_size

    for i in tqdm(range(0, len(image_files), B), desc="Curating"):
        batch_paths = image_files[i: i + B]

        # ── Stage 0: Load + quality filter (CPU only) ─────────────────────────
        batch_imgs, valid_paths = [], []
        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is None:
                counts["invalid"] += 1
                all_results[str(p)] = {"accepted": False,
                                       "rejection_reasons": ["invalid_image"]}
                continue
            q = pipeline.quality_filter.check(img)
            if not q["passed"]:
                counts["rejected_quality"] += 1
                result = {"accepted": False, "rejection_reasons": ["quality"],
                          "rejection_detail": q.get("reason"), "path": str(p)}
                all_results[str(p)] = result
                _handle_rejected(img, p, result, args)
            else:
                batch_imgs.append(img)
                valid_paths.append(p)

        if not batch_imgs:
            continue

        # ── Stage 1: Body completeness (YOLOv8-pose, batched) ────────────────
        yolo_results = pipeline.body_filter.model(
            batch_imgs, verbose=False, conf=pipeline.body_filter.min_person_conf)

        s1_pass_imgs, s1_pass_paths, s1_pass_r1 = [], [], []
        for img, path, yres in zip(batch_imgs, valid_paths, yolo_results):
            r1 = pipeline.body_filter._evaluate_result(img, yres)
            if r1["passed"]:
                s1_pass_imgs.append(img)
                s1_pass_paths.append(path)
                s1_pass_r1.append(r1)
            else:
                counts["rejected_body"] += 1
                result = {"accepted": False, "rejection_reasons": ["body"],
                          "rejection_detail": r1.get("reason"), "path": str(path)}
                all_results[str(path)] = result
                _handle_rejected(img, path, result, args)

        if not s1_pass_imgs:
            continue

        # ── Stage 1.5: Real human gate (InsightFace face detection) ──────────
        # Sunglasses bypass: skip InsightFace if YOLO conf > 0.75 and
        # sunglasses detected — avoids false rejections on occluded faces.
        s15_pass_imgs, s15_pass_paths, s15_pass_r1 = [], [], []
        for img, path, r1 in zip(s1_pass_imgs, s1_pass_paths, s1_pass_r1):
            is_sunglasses = r1.get("is_sunglasses", False)
            yolo_conf     = float(r1.get("yolo_conf", 0.0))

            if is_sunglasses and yolo_conf > 0.75:
                passes = True
            else:
                passes = pipeline.age_filter.has_real_face(
                    img, yolo_conf=0.0, n_visible_keypoints=0)

            if passes:
                s15_pass_imgs.append(img)
                s15_pass_paths.append(path)
                s15_pass_r1.append(r1)
            else:
                counts["rejected_not_human"] += 1
                result = {"accepted": False,
                          "rejection_reasons": ["not_real_human"],
                          "rejection_detail": "no_real_face_detected",
                          "path": str(path)}
                all_results[str(path)] = result
                _handle_rejected(img, path, result, args)

        if not s15_pass_imgs:
            continue

        # ── Stage 2: Ad detection (CLIP + Sobel + EasyOCR + bg crop) ─────────
        ad_results = pipeline.ad_filter.check_batch(
            s15_pass_imgs,
            paths=[str(p) for p in s15_pass_paths]
        )

        s2_pass_imgs, s2_pass_paths, s2_pass_r1, s2_pass_ad = [], [], [], []
        for img, path, r1, ad_r in zip(s15_pass_imgs, s15_pass_paths,
                                       s15_pass_r1, ad_results):
            if ad_r["passed"]:
                s2_pass_imgs.append(img)
                s2_pass_paths.append(path)
                s2_pass_r1.append(r1)
                s2_pass_ad.append(ad_r)
            else:
                counts["rejected_ad"] += 1
                result = {"accepted": False,
                          "rejection_reasons": ["ad"],
                          "rejection_detail": ad_r.get("reason"),
                          "fake_probability": ad_r.get("fake_probability"),
                          "path": str(path)}
                all_results[str(path)] = result
                _handle_rejected(img, path, result, args)

        if not s2_pass_imgs:
            continue

        # ── Stage 2.5: Child gate (CLIP zero-shot) ────────────────────────────
        # Fails open on exceptions to avoid silent data loss.
        s25_pass_imgs, s25_pass_paths, s25_pass_r1, s25_pass_ad = [], [], [], []
        for img, path, r1, ad_r in zip(s2_pass_imgs, s2_pass_paths,
                                       s2_pass_r1, s2_pass_ad):
            try:
                is_child = pipeline.ad_filter.is_likely_child(img)
            except Exception as e:
                print(f"[WARN] is_likely_child failed for {path.name}: {e}")
                is_child = False

            if is_child:
                counts["rejected_age"] += 1
                result = {"accepted": False,
                          "rejection_reasons": ["child_detected"],
                          "rejection_detail": "clip_child_gate",
                          "path": str(path)}
                all_results[str(path)] = result
                _handle_rejected(img, path, result, args)
            else:
                s25_pass_imgs.append(img)
                s25_pass_paths.append(path)
                s25_pass_r1.append(r1)
                s25_pass_ad.append(ad_r)

        if not s25_pass_imgs:
            continue

        # ── Stage 3: Age estimation (InsightFace) ─────────────────────────────
        for img, path, r1, ad_r in zip(s25_pass_imgs, s25_pass_paths,
                                       s25_pass_r1, s25_pass_ad):
            age_r = pipeline.age_filter.check(
                img, pose_keypoints=r1.get("_raw_keypoints"))

            if age_r["passed"]:
                counts["accepted"] += 1
                shutil.copy2(path, output_path / Path(path).name)
                result = {
                    "accepted":          True,
                    "rejection_reasons": [],
                    "age_estimated":     age_r.get("age_estimated"),
                    "face_type":         r1.get("face_type"),
                    "body_span_ratio":   r1.get("body_span_ratio"),
                    "fake_probability":  ad_r.get("fake_probability"),
                    "stages": {
                        "advertisement":  ad_r,
                        "age_estimation": age_r,
                    },
                    "path": str(path),
                }
            else:
                counts["rejected_age"] += 1
                result = {
                    "accepted":          False,
                    "rejection_reasons": ["age"],
                    "rejection_detail":  age_r.get("reason"),
                    "age_estimated":     age_r.get("age_estimated"),
                    "path":              str(path),
                }
                _handle_rejected(img, path, result, args)

            all_results[str(path)] = result

            if args.visualize:
                annotated = draw_result(img, result)
                cv2.imwrite(
                    str(output_path / "visualizations" / Path(path).name),
                    annotated)

    # ── Save results + print summary ──────────────────────────────────────────
    with open(output_path / "curation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(output_path / "curation_summary.json", "w") as f:
        json.dump(counts, f, indent=2)

    n = counts["total"]
    print(f"\n{'='*45}")
    print(f"Total              : {n}")
    print(f"Accepted           : {counts['accepted']} ({counts['accepted']/max(n,1)*100:.1f}%)")
    print(f"Rejected quality   : {counts['rejected_quality']}")
    print(f"Rejected body      : {counts['rejected_body']}")
    print(f"Rejected not-human : {counts['rejected_not_human']}")
    print(f"Rejected ad        : {counts['rejected_ad']}")
    print(f"Rejected age       : {counts['rejected_age']}")
    print(f"Invalid/unread     : {counts['invalid']}")
    print(f"Accepted saved     → {output_path}/")
    print(f"Results JSON       → {output_path}/curation_results.json")


if __name__ == "__main__":
    main()
