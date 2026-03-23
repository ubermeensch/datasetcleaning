# Dataset Curation Pipeline

An automated image dataset curation pipeline for filtering noisy real-world human image datasets. The pipeline requires no manual labelling and uses a combination of conventional computer vision techniques and deep learning models to filter out low-quality, incomplete, non-human, advertisement, and underage images.

---

## Pipeline Overview

Images pass through six sequential filtering stages. Each stage only processes images that passed all prior stages, reducing computational load progressively.

| Stage | Filter | Method |
|---|---|---|
| 0 | Quality | Laplacian blur variance + resolution check |
| 1 | Body completeness | YOLOv8-pose keypoint detection |
| 1.5 | Real human gate | InsightFace face detection |
| 2 | Advertisement detection | CLIP + Sobel edge density + EasyOCR + background crop CLIP |
| 2.5 | Child gate | CLIP zero-shot classification |
| 3 | Age estimation | InsightFace age regression |

---

## Setup

### Requirements
- Python 3.11
- CUDA-capable GPU recommended

### Install dependencies

```bash
pip install -r requirements.txt


First run — model weights are downloaded automatically
YOLOv8 pose weights (yolov8n-pose.pt) — downloaded by ultralytics on first run

InsightFace buffalo_l model — downloaded on first run

CLIP ViT-B/32 — downloaded on first run

EasyOCR en + ch_sim models — downloaded on first run

Usage

Run on a local folder

python run_pipeline.py --input_dir ./noisy_dataset --output_dir ./curated_output

Run with Google Drive download

python run_pipeline.py --gdrive_url "https://drive.google.com/drive/folders/YOUR_ID" --output_dir ./curated_output

All options

python run_pipeline.py \
  --input_dir ./noisy_dataset \
  --output_dir ./curated_output \
  --config config/config.yaml \
  --batch_size 32 \
  --rejected_dir ./rejected \
  --visualize

| Argument       | Default            | Description                                        |
| -------------- | ------------------ | -------------------------------------------------- |
| --input_dir    | —                  | Path to input image folder                         |
| --gdrive_url   | —                  | Google Drive folder URL (alternative to input_dir) |
| --output_dir   | required           | Where accepted images are saved                    |
| --config       | config/config.yaml | Path to config file                                |
| --batch_size   | 32                 | Images per GPU batch                               |
| --rejected_dir | None               | Optional folder to copy rejected images into       |
| --visualize    | False              | Save annotated debug images                        |

Configuration

Edit config/config.yaml to adjust thresholds:

pipeline:
  quality_filter:
    min_width: 20
    min_height: 40
    min_blur_variance: 0.0       # 0 = disabled

  body_completeness:
    model: yolov8n-pose.pt
    min_body_span_ratio: 0.25
    min_keypoint_confidence: 0.10
    min_person_confidence: 0.15

  advertisement_detection:
    clip_model: ViT-B/32
    clip_ad_threshold: 0.60
    edge_density_threshold: 0.25
    bg_ad_threshold: 0.48

  age_estimation:
    min_acceptable_age: 13
    age_uncertainty_buffer: 3

Output

curated_output/
├── crop_001.png              accepted images
├── crop_002.png
├── ...
├── curation_results.json     per-image result with stage details
└── curation_summary.json     aggregate counts

Summary format (curation_summary.json)

{
  "total": 1147,
  "accepted": 186,
  "rejected_quality": 319,
  "rejected_body": 468,
  "rejected_not_human": 128,
  "rejected_ad": 44,
  "rejected_age": 2,
  "invalid": 0
}


Project Structure

datasetcleaning/
├── run_pipeline.py              ← main entry point
├── requirements.txt
├── config/
│   └── config.yaml
├── pipeline/
│   ├── __init__.py
│   ├── pipeline.py              ← CurationPipeline class
│   ├── quality_filter.py        ← Stage 0
│   ├── body_completeness.py     ← Stage 1
│   ├── ad_detection.py          ← Stage 2
│   └── age_estimation.py        ← Stage 1.5, 2.5, 3
└── utils/
    ├── image_utils.py
    └── visualization.py
