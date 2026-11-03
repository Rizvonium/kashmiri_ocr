"""
Module 1: SynthCamera-KS v2.0 Synthetic Dataset Generator
Renders Kashmiri text with HarfBuzz/Pango or Pillow, applying:
1. Physical Book 3D Cylindrical Spine Warping, Perspective, and Hand-Shadow Gradients
2. Screen Photo RGB Subpixel Grids and Sinusoidal Moiré Interference
3. Randomized inter-word kerning (0.6x - 1.5x)
"""

import math
import random
from typing import Tuple, List, Optional, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

from python_engine.normalizer import normalize_kashmiri

class SynthCameraKS:
    """
    Synthetic Kashmiri Camera & Document Line Generator v2.0.
    Produces paired (image, normalized_canonical_text) samples with photorealistic
    curved book spine distortions, shadows, and screen Moiré artifacts.
    """

    FONT_CANDIDATES = [
        "Noto Nastaliq Urdu",
        "Gulmarg Nastaleeq",
        "SIL Awami Nastaliq",
        "Mehr Nastaliq",
        "Afan Koshur Naksh",
        "Noto Sans Arabic"
    ]

    SAMPLE_CORPUS = [
        "کٲشِر زَبانہِ ہٕنٛز تَوٲریخ چھےٚ واریاہ پرٛٲنؠ۔",
        "اَتھ مَنٛز چھےٚ شیٚنٛٹھ، شیو مَت تہٕ صُوفی شٲعِری ہٕنٛز خوٗشبوٗ۔",
        "آزاد کٔشیٖر تہٕ وادیِ کٔشیٖر کؠن لوٗکَن ہُنٛد فخر۔",
        "کٲشُر لُغَت — جِلد گۄڈٕنِیُک آب تہٕ آزٲدی۔",
        "لَل دؠد تہٕ نُند رؠشؠن کٔر شَنٛگار وادی مَنٛز۔",
        "مَحجوٗر تہٕ حَبہٕ خاتوٗنَن دِیُت کٲشِرِ کلامَس نٔو زِندگی۔",
        "ترٛومُت صفت آبَس مَنٛز کٔڈِتھ بَرٛوٛنٛہہ ترٛاوُن۔",
        "دۄپھ آواز زمینَس پؠٹھ کِہین پؠنہٕ چِھ چھَنٛڈ۔"
    ]

    def __init__(self, font_path: Optional[str] = None, default_font_size: int = 32):
        self.default_font_size = default_font_size
        self.font = self._load_best_font(font_path, default_font_size)

    def _load_best_font(self, font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
        """Finds and loads an installed Nastaliq or Arabic font."""
        if font_path:
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass

        # Try system fonts
        system_paths = [
            "/usr/share/fonts/opentype/noto/NotoNastaliqUrdu-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf"
        ]
        for p in system_paths:
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue

        # Fallback to default
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    def render_clean_line(
        self,
        text: str,
        font_size: Optional[int] = None,
        kerning_mult: float = 1.0,
        pad_vert: int = 40,
        pad_horiz: int = 40
    ) -> Image.Image:
        """
        Renders a baseline clean Kashmiri text strip with vertical padding (+40px)
        to protect interlocking ascenders and deep descenders.
        """
        normalized_text = normalize_kashmiri(text)
        size = font_size or self.default_font_size
        font = self.font
        if font_size and hasattr(font, 'path'):
            font = ImageFont.truetype(font.path, font_size)

        # Estimate dimension
        words = normalized_text.split()
        space_width = int(24 * kerning_mult)

        # Measure text with font
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)

        # Render word-by-word RTL
        rendered_words = []
        total_text_width = 0
        max_h = 50

        for word in words:
            bbox = draw.textbbox((0, 0), word, font=font)
            w = max(bbox[2] - bbox[0], 20)
            h = max(bbox[3] - bbox[1], 30)
            rendered_words.append((word, w, h))
            total_text_width += w
            if h > max_h:
                max_h = h

        total_width = total_text_width + (len(words) - 1) * space_width + (pad_horiz * 2)
        total_height = max_h + (pad_vert * 2)

        img = Image.new("RGB", (max(total_width, 100), max(total_height, 80)), color=(255, 255, 255))
        img_draw = ImageDraw.Draw(img)

        # Draw Right-to-Left
        curr_x = total_width - pad_horiz
        y_pos = pad_vert + 5

        for word, w, h in rendered_words:
            curr_x -= w
            img_draw.text((curr_x, y_pos), word, font=font, fill=(15, 23, 42))
            curr_x -= space_width

        return img

    def apply_cylindrical_spine_warp(self, img: Image.Image, max_amplitude: float = 25.0) -> Image.Image:
        """
        Simulates 3D curved physical book spine deformation using cylindrical sine wave:
        y' = y + A * sin(pi * x / W)
        """
        arr = np.array(img)
        h, w, c = arr.shape
        out = np.full_like(arr, 255)

        # Amplitude varies across horizontal dimension
        x_indices = np.arange(w)
        y_shifts = (max_amplitude * np.sin(np.pi * x_indices / w)).astype(int)

        for x in range(w):
            shift = y_shifts[x]
            if shift >= 0:
                if shift < h:
                    out[shift:, x, :] = arr[:h - shift, x, :]
            else:
                shift_abs = -shift
                if shift_abs < h:
                    out[:h - shift_abs, x, :] = arr[shift_abs:, x, :]

        return Image.fromarray(out)

    def apply_perspective_keystoning(self, img: Image.Image, angle_deg: float = 8.0) -> Image.Image:
        """Applies 4-point perspective warp representing mobile camera tilt."""
        w, h = img.size
        rad = math.radians(angle_deg)
        tilt_x = int(math.tan(rad) * h * 0.5)

        src_quad = [(0, 0), (w, 0), (w, h), (0, h)]
        dst_quad = [(tilt_x, 0), (w - tilt_x, 0), (w, h), (0, h)]

        # Find coefficients
        def find_coeffs(pa, pb):
            matrix = []
            for p1, p2 in zip(pa, pb):
                matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
                matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
            A = np.matrix(matrix, dtype=float)
            B = np.array(pb).reshape(8)
            res = np.dot(np.linalg.pinv(A), B)
            return np.array(res).reshape(8)

        coeffs = find_coeffs(dst_quad, src_quad)
        return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC, fillcolor=(255, 255, 255))

    def apply_shadow_illumination(self, img: Image.Image, shadow_intensity: float = 0.5) -> Image.Image:
        """
        Applies non-uniform room lighting and soft hand-shadow gradients:
        I_shadow(x, y) = I(x, y) * G(x, y)
        """
        w, h = img.size
        # Generate 2D smooth parabolic gradient
        cx = random.uniform(0.1, 0.9) * w
        cy = random.uniform(0.1, 0.9) * h
        max_dist = math.sqrt(w*w + h*h)

        y_coords, x_coords = np.ogrid[:h, :w]
        dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2) / max_dist
        gradient = 1.0 - (shadow_intensity * 0.7 * (1.0 - dist))
        gradient = np.clip(gradient, 0.25, 1.0)[:, :, np.newaxis]

        arr = np.array(img, dtype=np.float32) * gradient
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def apply_screen_moire(self, img: Image.Image, freq: float = 0.25, intensity: float = 0.35) -> Image.Image:
        """
        Overlays RGB subpixel stripes and high-frequency sinusoidal Moiré interference
        to simulate photographing laptop/mobile screens.
        """
        arr = np.array(img, dtype=np.float32)
        h, w, c = arr.shape

        # 1. Sinusoidal 2D interference mesh
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        moire_pattern = 0.5 + 0.5 * np.sin(2 * np.pi * (freq * xx + 0.1 * yy))
        moire_factor = 1.0 - (intensity * moire_pattern)[:, :, np.newaxis]

        # 2. RGB subpixel column alternation (R-G-B stripes)
        subpixel_mask = np.ones((h, w, 3), dtype=np.float32)
        subpixel_mask[:, 0::3, 0] *= 1.15  # Red boost
        subpixel_mask[:, 1::3, 1] *= 1.15  # Green boost
        subpixel_mask[:, 2::3, 2] *= 1.15  # Blue boost

        arr = arr * moire_factor * subpixel_mask
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def apply_sensor_noise_and_blur(self, img: Image.Image, blur_sigma: float = 0.8, noise_level: float = 0.05) -> Image.Image:
        """Applies Gaussian sensor blur and Poisson-like ISO noise."""
        if blur_sigma > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

        if noise_level > 0:
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, noise_level * 255, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img

    def generate_sample(
        self,
        text: Optional[str] = None,
        track: str = "random"
    ) -> Tuple[Image.Image, str]:
        """
        Generates a synthetic camera sample with ground truth canonical Kashmiri text.
        Tracks:
        - 'book' (50%): 3D cylindrical spine curl + hand shadows + keystoning
        - 'screen' (30%): RGB subpixel stripes + high-frequency Moiré
        - 'clean' (20%): Flat scan
        """
        raw_text = text or random.choice(self.SAMPLE_CORPUS)
        canonical_text = normalize_kashmiri(raw_text)

        # Kerning variation (0.6x to 1.5x)
        kerning_mult = random.uniform(0.7, 1.4)
        img = self.render_clean_line(canonical_text, kerning_mult=kerning_mult)

        if track == "random":
            p = random.random()
            if p < 0.50:
                track = "book"
            elif p < 0.80:
                track = "screen"
            else:
                track = "clean"

        if track == "book":
            # Book Spine Track
            img = self.apply_cylindrical_spine_warp(img, max_amplitude=random.uniform(15, 35))
            img = self.apply_perspective_keystoning(img, angle_deg=random.uniform(-10, 10))
            img = self.apply_shadow_illumination(img, shadow_intensity=random.uniform(0.3, 0.7))
            img = self.apply_sensor_noise_and_blur(img, blur_sigma=random.uniform(0.4, 1.2), noise_level=random.uniform(0.02, 0.08))

        elif track == "screen":
            # Screen Photo Track
            img = self.apply_screen_moire(img, freq=random.uniform(0.18, 0.32), intensity=random.uniform(0.25, 0.5))
            img = self.apply_sensor_noise_and_blur(img, blur_sigma=random.uniform(0.2, 0.8), noise_level=random.uniform(0.01, 0.04))

        else:
            # Clean Scan Track with slight noise
            img = self.apply_sensor_noise_and_blur(img, blur_sigma=0.3, noise_level=0.01)

        return img, canonical_text
