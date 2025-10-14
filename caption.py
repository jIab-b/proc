#!/usr/bin/env python3
"""Generate simple heuristic captions for exported WebGPU dataset images."""
from __future__ import annotations

import argparse
import colorsys
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

# Category metadata used for textual descriptions.
CATEGORY_INFO: Dict[str, Dict[str, Iterable[str]]] = {
    "sky": {"label": "sky", "tags": ("sky", "outdoor"), "phrase": "beneath a blue sky"},
    "cloud": {"label": "cloud", "tags": ("cloud",), "phrase": "with pale cloud cover"},
    "grass": {"label": "grass", "tags": ("grass", "vegetation"), "phrase": "highlighting grassy terrain"},
    "foliage": {"label": "foliage", "tags": ("foliage", "vegetation"), "phrase": "showing dense foliage"},
    "dirt": {"label": "dirt", "tags": ("dirt", "soil"), "phrase": "revealing exposed dirt"},
    "sand": {"label": "sand", "tags": ("sand",), "phrase": "covering sandy ground"},
    "stone": {"label": "stone", "tags": ("stone", "rock"), "phrase": "featuring gray stone formations"},
    "water": {"label": "water", "tags": ("water", "river"), "phrase": "including reflective water"},
    "snow": {"label": "snow", "tags": ("snow", "ice"), "phrase": "dusted with snow"},
    "shadow": {"label": "shadow", "tags": ("shadow",), "phrase": "casting long shadows"},
    "lava": {"label": "lava", "tags": ("lava", "glow"), "phrase": "showing glowing magma"},
    "unknown": {"label": "mixed", "tags": ("mixed",), "phrase": "with mixed terrain"},
}

SIGNIFICANT_THRESHOLD = 0.05
TOP_REGION_FRACTION = 0.35
BOTTOM_REGION_FRACTION = 0.35


@dataclass
class ImageAnalysis:
    image: Path
    caption: str
    tags: List[str]
    dominant_colors: List[Dict[str, object]]


def classify_pixel(r: int, g: int, b: int) -> str:
    """Rudimentary color classification tailored to block-world renders."""
    rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rn, gn, bn)
    h_deg = h * 360.0

    if v < 0.12:
        return "shadow"
    if v > 0.9 and s < 0.18:
        return "snow"
    if s < 0.12:
        return "stone"
    if 180.0 <= h_deg <= 255.0:
        return "sky" if v > 0.55 else "water"
    if 70.0 <= h_deg < 160.0:
        return "grass"
    if 40.0 <= h_deg < 70.0:
        return "sand" if v > 0.6 else "dirt"
    if 10.0 <= h_deg < 40.0:
        return "dirt"
    if h_deg < 10.0 or h_deg >= 340.0:
        return "lava" if v > 0.5 else "shadow"
    if 255.0 < h_deg < 300.0 and v > 0.4:
        return "foliage"
    if v > 0.8 and s < 0.2:
        return "cloud"
    return "unknown"


def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((192, 192), Image.LANCZOS)
    return img


def analyse_pixels(img: Image.Image) -> Tuple[Counter, Counter, Counter]:
    width, height = img.size
    pixels = img.load()

    counts = Counter()
    top_counts = Counter()
    bottom_counts = Counter()

    top_cutoff = int(height * TOP_REGION_FRACTION)
    bottom_cutoff = int(height * (1.0 - BOTTOM_REGION_FRACTION))

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            category = classify_pixel(r, g, b)
            counts[category] += 1
            if y < top_cutoff:
                top_counts[category] += 1
            elif y >= bottom_cutoff:
                bottom_counts[category] += 1
    return counts, top_counts, bottom_counts


def dominant_list(counts: Counter) -> List[Tuple[str, float]]:
    total = sum(counts.values()) or 1
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [(cat, count / total) for cat, count in ordered if count / total >= SIGNIFICANT_THRESHOLD]


def build_caption(dominant: List[Tuple[str, float]], top_counts: Counter, bottom_counts: Counter) -> str:
    phrases: List[str] = []
    surface_candidates = [cat for cat, _ in dominant if cat not in {"sky", "cloud", "shadow", "unknown"}]

    if surface_candidates:
        first = surface_candidates[0]
        phrases.append(CATEGORY_INFO.get(first, {}).get("phrase", "showing varied terrain"))
        if len(surface_candidates) > 1:
            second = surface_candidates[1]
            second_phrase = CATEGORY_INFO.get(second, {}).get("phrase")
            if second_phrase and second_phrase not in phrases:
                phrases.append(second_phrase)
    else:
        phrases.append("showing mixed terrain")

    def region_ratio(region_counts: Counter, key: str) -> float:
        total = sum(region_counts.values()) or 1
        return region_counts.get(key, 0) / total

    if region_ratio(top_counts, "sky") >= 0.4:
        phrases.append("under a bright sky")
    elif region_ratio(top_counts, "cloud") >= 0.35:
        phrases.append("beneath light cloud cover")

    if region_ratio(bottom_counts, "water") >= 0.2:
        phrases.append("near reflective water")
    elif region_ratio(bottom_counts, "grass") >= 0.35:
        phrases.append("above a grassy foreground")

    phrase_text = ", ".join(dict.fromkeys(phrases))
    caption = f"Isometric voxel landscape {phrase_text}. Captured from an elevated camera angle looking across the scene."
    return " ".join(caption.split())


def convert_tags(dominant: List[Tuple[str, float]]) -> List[str]:
    tags: List[str] = []
    for cat, _ in dominant:
        info = CATEGORY_INFO.get(cat)
        if not info:
            continue
        for tag in info.get("tags", (cat,)):
            if tag not in tags:
                tags.append(tag)
    return tags


def analyse_image(path: Path) -> ImageAnalysis:
    img = load_image(path)
    counts, top_counts, bottom_counts = analyse_pixels(img)
    dom = dominant_list(counts)
    caption = build_caption(dom, top_counts, bottom_counts)

    dom_colors = [
        {
            "category": cat,
            "label": CATEGORY_INFO.get(cat, {}).get("label", cat),
            "percent": round(frac, 4),
        }
        for cat, frac in dom
    ]
    tags = convert_tags(dom)
    return ImageAnalysis(image=path, caption=caption, tags=tags, dominant_colors=dom_colors)


def gather_images(root: Path) -> List[Path]:
    if root.is_dir():
        return sorted(p for p in root.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    raise FileNotFoundError(f"No images found at {root}")


def resolve_paths(dataset_path: Path, images_path: Path | None, output_path: Path | None) -> Tuple[Path, Path, Path]:
    if images_path:
        image_dir = images_path
    else:
        candidate = dataset_path / "images"
        if candidate.is_dir():
            image_dir = candidate
        else:
            image_dir = dataset_path
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if output_path:
        out_file = output_path
    else:
        if dataset_path.is_dir():
            out_file = dataset_path / "captions.json"
        else:
            out_file = image_dir.parent / "captions.json"
    return dataset_path, image_dir, out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate heuristic captions for dataset images.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/7"), help="Dataset directory containing images and metadata.")
    parser.add_argument("--images", type=Path, default=None, help="Explicit images directory (overrides dataset/images).")
    parser.add_argument("--output", type=Path, default=None, help="Output captions JSON file path.")
    args = parser.parse_args()

    dataset_path, image_dir, output_path = resolve_paths(args.dataset.resolve(), args.images, args.output)
    images = gather_images(image_dir)

    analyses = [analyse_image(img_path) for img_path in images]
    output_payload = [
        {
            "image": str(analysis.image.relative_to(dataset_path)),
            "caption": analysis.caption,
            "tags": analysis.tags,
            "dominant_colors": analysis.dominant_colors,
        }
        for analysis in analyses
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output_payload, fh, indent=2)
    print(f"Saved {len(output_payload)} captions to {output_path}")


if __name__ == "__main__":
    main()
