#!/usr/bin/env python3
"""
Generate a mock psoriasis skin image dataset for pipeline validation.

Creates sharp, blurry, and near-duplicate mock clinical skin photographs
distributed across Mild, Moderate, and Severe directories to verify data discovery,
blur filtering, near-duplicate detection, split stratification, training,
XAI generation, and metric evaluation.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def generate_mock_image(size: int = 256, blur: bool = False, draw_lesion: bool = True, severity: str = "Mild"):
    """
    Generate a mock clinical skin image.
    
    Skin color base, optional red psoriasis-like lesions with scaling,
    and optional Gaussian blur to simulate low quality.
    """
    # Skin tone background
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = [180, 210, 240]  # Skin-like BGR (peach-pinkish)

    # Draw lesions if requested
    if draw_lesion:
        # Severity controls lesion size and count
        if severity == "Mild":
            num_lesions = 1
            max_radius = 20
            color = [100, 100, 200]  # Light reddish-pink
        elif severity == "Moderate":
            num_lesions = 3
            max_radius = 35
            color = [80, 80, 220]    # Medium red
        else:  # Severe
            num_lesions = 6
            max_radius = 50
            color = [50, 50, 240]    # Dark angry red

        for _ in range(num_lesions):
            center = (np.random.randint(40, size - 40), np.random.randint(40, size - 40))
            radius = np.random.randint(10, max_radius)
            # Draw irregular lesion shapes using filled polygons
            points = []
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                r = radius + np.random.randint(-int(radius*0.3), int(radius*0.3) + 1)
                x = int(center[0] + r * np.cos(angle))
                y = int(center[1] + r * np.sin(angle))
                points.append([x, y])
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(img, [pts], color)
            
            # Add some skin scaling texture (white flakes)
            for _ in range(radius * 2):
                tx = np.random.randint(center[0] - radius, center[0] + radius)
                ty = np.random.randint(center[1] - radius, center[1] + radius)
                if 0 <= tx < size and 0 <= ty < size:
                    cv2.circle(img, (tx, ty), 1, [240, 240, 240], -1)

    # Add background skin noise (texture)
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Apply blur if requested
    if blur:
        img = cv2.GaussianBlur(img, (15, 15), 0)
    else:
        # Add high frequency details to make sure it's sharp
        cv2.putText(img, "SHARP", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [100, 100, 100], 1)

    return img


def main():
    parser = argparse.ArgumentParser(description="Generate mock data for validation")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Target raw directory")
    parser.add_argument("--samples-per-class", type=int, default=15, help="Number of sharp samples per class")
    parser.add_argument("--blurry-samples", type=int, default=3, help="Number of blurry samples to generate")
    parser.add_argument("--duplicate-samples", type=int, default=2, help="Number of duplicate groups to inject")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    print(f"[MockData] Generating mock psoriasis images in: {output_path.resolve()}")

    severities = ["Mild", "Moderate", "Severe"]
    for sev in severities:
        sev_dir = output_path / sev
        sev_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate sharp samples
        for idx in range(args.samples_per_class):
            img = generate_mock_image(size=256, blur=False, draw_lesion=True, severity=sev)
            img_path = sev_dir / f"sample_sharp_{idx:03d}.jpg"
            cv2.imwrite(str(img_path), img)

        # 2. Generate blurry samples (should be filtered out by blur threshold)
        for idx in range(args.blurry_samples):
            img = generate_mock_image(size=256, blur=True, draw_lesion=True, severity=sev)
            img_path = sev_dir / f"sample_blurry_{idx:03d}.jpg"
            cv2.imwrite(str(img_path), img)

        # 3. Generate near-duplicate samples (keeps the first, drops the second)
        for idx in range(args.duplicate_samples):
            img_orig = generate_mock_image(size=256, blur=False, draw_lesion=True, severity=sev)
            path_orig = sev_dir / f"sample_dup_{idx:03d}_a.jpg"
            cv2.imwrite(str(path_orig), img_orig)

            # Add minor noise to make it a near-duplicate rather than byte-exact
            noise = np.random.normal(0, 1, img_orig.shape).astype(np.int16)
            img_dup = np.clip(img_orig.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            path_dup = sev_dir / f"sample_dup_{idx:03d}_b.jpg"
            cv2.imwrite(str(path_dup), img_dup)

    print("[MockData] Data generation complete!")
    print(f"Total directories created:")
    for sd in sorted(output_path.iterdir()):
        if sd.is_dir():
            print(f"  └── {sd.name}: {len(list(sd.glob('*.jpg')))} images")


if __name__ == "__main__":
    main()
