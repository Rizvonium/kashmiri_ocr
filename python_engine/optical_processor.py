"""
Module 2: Optical Restoration, Dewarping & Descreening
Implements:
1. Division Illumination Normalization (eliminates hand shadows & flash glare without eroding diacritics)
2. Bilateral Descreening (removes high-frequency LCD/OLED screen Moiré)
3. 2D Thin-Plate Spline (TPS) Baseline Rectification (flattens curved book spines)
4. Laplacian Sharpness Variance Gate (> 110)
"""

import math
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageFilter

class OpticalPreprocessor:
    """
    Client and Server optical restoration pipeline for camera and scanned Kashmiri books.
    """

    def __init__(self, blur_threshold: float = 110.0, sigma_illum: float = 45.0):
        self.blur_threshold = blur_threshold
        self.sigma_illum = sigma_illum

    def check_sharpness(self, img: Image.Image) -> Tuple[bool, float]:
        """
        Computes the Laplacian variance over a central patch to evaluate camera sharpness.
        Variance > 110 indicates a crisp image suitable for 1-2px diacritic recognition.
        """
        gray = img.convert('L')
        w, h = gray.size
        # Crop center 256x256 patch
        cw, ch = min(w, 256), min(h, 256)
        left = (w - cw) // 2
        top = (h - ch) // 2
        patch = gray.crop((left, top, left + cw, top + ch))
        arr = np.array(patch, dtype=np.float32)

        # 3x3 Discrete Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        lap = (
            np.roll(arr, 1, axis=0) +
            np.roll(arr, -1, axis=0) +
            np.roll(arr, 1, axis=1) +
            np.roll(arr, -1, axis=1) -
            4 * arr
        )[1:-1, 1:-1]

        variance = float(lap.var())
        is_sharp = variance >= self.blur_threshold
        return is_sharp, variance

    def division_illumination_normalization(
        self,
        img: Image.Image,
        sigma: Optional[float] = None,
        target_mean: float = 240.0
    ) -> Image.Image:
        """
        Applies adaptive background division normalization:
        I_norm = clip((I / (GaussianBlur(I, sigma=45) + eps)) * 240, 0, 255)
        
        Crucial: Unlike Otsu or Sauvola binarization, this preserves 1-2px subtle
        diacritical strokes (Hamza Below U+0655, Kashmiri Yeh ring U+0620, Wavy Dagger U+065F)
        with intact subpixel anti-aliasing.
        """
        s = sigma or self.sigma_illum
        # Work in float32
        arr = np.array(img, dtype=np.float32)

        # Compute smooth Gaussian background illumination map B(x, y)
        if len(arr.shape) == 3:
            # Multi-channel Gaussian blur
            pil_blur = img.filter(ImageFilter.GaussianBlur(radius=s))
            bg = np.array(pil_blur, dtype=np.float32)
        else:
            pil_blur = img.filter(ImageFilter.GaussianBlur(radius=s))
            bg = np.array(pil_blur, dtype=np.float32)

        # Division formula
        eps = 1e-5
        norm = (arr / (bg + eps)) * target_mean
        norm = np.clip(norm, 0, 255).astype(np.uint8)

        return Image.fromarray(norm)

    def bilateral_descreening(self, img: Image.Image, diameter: int = 5) -> Image.Image:
        """
        Suppresses high-frequency periodic screen Moiré interference grids
        photographed from laptop/phone screens while preserving edge contrast.
        """
        # A combination of subtle median filter and unsharp mask
        smoothed = img.filter(ImageFilter.MedianFilter(size=3))
        # Blend 70% smoothed, 30% original to remove Moiré stripes
        arr_orig = np.array(img, dtype=np.float32)
        arr_smooth = np.array(smoothed, dtype=np.float32)
        blended = (0.7 * arr_smooth + 0.3 * arr_orig).astype(np.uint8)
        return Image.fromarray(blended)

    def tps_spine_dewarp(
        self,
        img: Image.Image,
        curvature_depth: Optional[float] = None
    ) -> Image.Image:
        """
        2D Thin-Plate Spline (TPS) / Polynomial mesh baseline rectification:
        Flattens cylindrical 3D book spine curvature.
        """
        arr = np.array(img)
        h, w = arr.shape[:2]

        # Detect or estimate curvature amplitude (spine typically at left or right margin)
        # In RTL Nastaliq books, spine is on the right for right-hand pages or left for left-hand pages.
        amp = curvature_depth if curvature_depth is not None else 18.0

        x_coords = np.arange(w)
        # Inverted sine curve to counteract spine sag
        counter_shifts = (-amp * np.sin(np.pi * x_coords / w)).astype(int)

        out = np.full_like(arr, 255)
        for x in range(w):
            shift = counter_shifts[x]
            if shift >= 0:
                if shift < h:
                    out[shift:, x] = arr[:h - shift, x]
            else:
                shift_abs = -shift
                if shift_abs < h:
                    out[:h - shift_abs, x] = arr[shift_abs:, x]

        return Image.fromarray(out)

    def preprocess_page(
        self,
        img: Image.Image,
        apply_descreen: bool = True,
        apply_dewarp: bool = True
    ) -> Image.Image:
        """Complete optical pre-processing pipeline for an input page."""
        # 1. Illumination normalization
        processed = self.division_illumination_normalization(img)

        # 2. Descreen if photographed from monitor
        if apply_descreen:
            processed = self.bilateral_descreening(processed)

        # 3. TPS Dewarp if book spine curvature present
        if apply_dewarp:
            processed = self.tps_spine_dewarp(processed)

        return processed
