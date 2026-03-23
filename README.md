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
