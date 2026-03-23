import cv2
import numpy as np
import torch
import clip
import easyocr
from PIL import Image
from typing import Dict, Any


CHILD_PROMPTS = [
    "a young child",
    "a child wearing a school uniform",
    "children in school uniforms",
    "a primary school student",
    "young kids in a classroom",
    "a toddler or infant",
    "a baby or toddler being carried by an adult",
    "a child with a young face and small body",
    "an adult man carrying a toddler in his arms",
    "a parent holding a young child or baby",
    "a man carrying a small child outdoors",
]

REAL_PROMPTS = [
    "a real photograph of a real human being",
    "a candid photo of a real person",
    "a real person standing or walking outdoors",
    "a human being photographed in real life",
]

FAKE_PROMPTS = [
    "an advertisement or commercial poster with a person",
    "a marketing or promotional graphic",
    "a mannequin or dress form wearing clothes",
    "clothes hanging on a rack or hanger with no person",
    "a cardboard cutout or life-size standee of a person",
    "a drawing illustration or cartoon of a person",
    "a statue or sculpture shaped like a person",
    "clothing displayed on a dummy or torso form",
    "a store window display with no real human",
    "a doll or toy shaped like a human",
    "a large printed poster of a celebrity on a wall",
    "a billboard advertisement featuring a famous person",
    "a photograph of a photograph or poster on a wall",
    "a printed photo of a person displayed on a surface",
    "a photograph of a digital LED screen displaying a person",
    "a photo taken of an outdoor billboard with a person on it",
    "a person displayed on a large illuminated screen or display",
    "a restaurant or food advertisement billboard with a person",
    "a photo of a TV or monitor screen showing a person",
    "a glowing digital advertisement board with a human face",
    "a printed poster or banner of a person displayed outdoors in daylight",
    "a large format print advertisement showing a man in a suit with text",
    "a digital billboard advertisement screen",
    "a person standing in front of an advertisement display",
    "an illuminated advertising display board",
    "a person posed in front of a commercial poster",
    "a chinese language advertisement banner",
    "a retail store promotional display with a person",
    "a person photographed next to a glowing LED screen advertisement",
]


class AdvertisementFilter:
    def __init__(self, config: dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(
            config.get("clip_model", "ViT-B/32"), device=self.device)
        self.clip_model.eval()

        self.ad_threshold   = config.get("clip_ad_threshold", 0.60)
        self.edge_threshold = config.get("edge_density_threshold", 0.25)
        self.bg_threshold   = config.get("bg_ad_threshold", 0.48)

        with torch.no_grad():
            all_prompts = REAL_PROMPTS + FAKE_PROMPTS
            tokens      = clip.tokenize(all_prompts).to(self.device)
            feats       = self.clip_model.encode_text(tokens)
            feats       = feats / feats.norm(dim=-1, keepdim=True)
            self._real_feats = feats[:len(REAL_PROMPTS)]
            self._fake_feats = feats[len(REAL_PROMPTS):]
            self._text_feats = torch.cat([self._real_feats, self._fake_feats])

        self._ocr_reader = easyocr.Reader(
            ['en', 'ch_sim'],
            gpu=torch.cuda.is_available(),
            verbose=False
        )

    def _fast_edge_density(self, image: np.ndarray) -> float:
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        return float(np.abs(sobel).mean() / 255.0)

    def _fake_probability(self, sim_row: np.ndarray) -> float:
        n_real     = len(REAL_PROMPTS)
        real_score = float(sim_row[:n_real].mean())
        fake_score = float(sim_row[n_real:].mean())
        temperature = 60.0
        diff = (fake_score - real_score) * temperature
        return float(1.0 / (1.0 + np.exp(-diff)))

    def _check_background(self, image: np.ndarray) -> float:
        """Run CLIP on top 40% of image to detect ad backgrounds."""
        h = image.shape[0]
        bg_crop = image[:int(h * 0.4), :]
        if bg_crop.size == 0:
            return 0.0
        pil_bg = Image.fromarray(cv2.cvtColor(bg_crop, cv2.COLOR_BGR2RGB))
        t = self.preprocess(pil_bg).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.clip_model.encode_image(t)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            sims = (feat @ self._text_feats.T).cpu().numpy()[0]
        n          = len(REAL_PROMPTS)
        real_score = sims[:n].max()
        fake_score = sims[n:].max()
        return float(fake_score - real_score + 0.5)

    def _has_text_overlay(self, img: np.ndarray) -> bool:
        """Detect text overlaid on image using EasyOCR (supports en + zh)."""
        results   = self._ocr_reader.readtext(img, detail=1, paragraph=False)
        confident = [r for r in results if r[2] > 0.3]
        if not confident:
            return False
        total_chars = sum(len(r[1]) for r in confident)
        return total_chars >= 4

    def _evaluate_sim(self, image: np.ndarray,
                      sim_row: np.ndarray, path: str = "") -> Dict[str, Any]:
        fake_prob    = self._fake_probability(sim_row)
        clip_flag    = fake_prob > self.ad_threshold
        borderline   = 0.20 < fake_prob < 0.65

        # Run secondary checks only on flagged or borderline images
        edge_density = self._fast_edge_density(image) if (clip_flag or borderline) else 0.0
        edge_flag    = edge_density > self.edge_threshold
        text_flag    = self._has_text_overlay(image) if (clip_flag or borderline) else False
        bg_score     = self._check_background(image) if (clip_flag or borderline) else 0.0
        bg_flag      = (bg_score >= self.bg_threshold) and (fake_prob > 0.20)

        is_fake = clip_flag or edge_flag or text_flag or bg_flag

        return {
            "passed":           not is_fake,
            "reason":           "advertisement_or_non_human" if is_fake else None,
            "fake_probability": fake_prob,
            "edge_density":     edge_density,
            "clip_flagged":     clip_flag,
            "edge_flagged":     edge_flag,
            "text_flagged":     text_flag,
            "bg_flagged":       bg_flag,
            "bg_score":         bg_score,
        }

    def check(self, image: np.ndarray) -> Dict[str, Any]:
        """Single-image inference. Used by process_image in pipeline.py."""
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            inp      = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            img_feat = self.clip_model.encode_image(inp)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim_row  = (img_feat @ self._text_feats.T).squeeze(0).cpu().numpy()
        return self._evaluate_sim(image, sim_row)

    def check_batch(self, images: list, paths: list = None) -> list:
        """Batched inference for run_pipeline.py main loop."""
        pil_imgs = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    for img in images]
        batch_t  = torch.stack([self.preprocess(p)
                                for p in pil_imgs]).to(self.device)
        with torch.no_grad():
            img_feats = self.clip_model.encode_image(batch_t)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            sims      = (img_feats @ self._text_feats.T).cpu().numpy()

        return [self._evaluate_sim(img, sim,
                                   path=str(paths[i]) if paths is not None else "")
                for i, (img, sim) in enumerate(zip(images, sims))]

    def is_likely_child(self, image: np.ndarray) -> bool:
        """CLIP zero-shot child detection against CHILD_PROMPTS."""
        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        img_tensor  = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(CHILD_PROMPTS).to(self.device)

        with torch.no_grad():
            img_feat = self.clip_model.encode_image(img_tensor)
            txt_feat = self.clip_model.encode_text(text_tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sims     = (img_feat @ txt_feat.T).squeeze(0)

        return float(sims.max()) > 0.30
