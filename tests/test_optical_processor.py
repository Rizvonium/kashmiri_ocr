import pytest
import numpy as np
from PIL import Image
from python_engine.optical_processor import OpticalPreprocessor

def test_sharpness_check():
    preprocessor = OpticalPreprocessor(blur_threshold=110.0)
    
    # Sharp synthetic image with high edge variance
    arr_sharp = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    img_sharp = Image.fromarray(arr_sharp)
    is_sharp, var_sharp = preprocessor.check_sharpness(img_sharp)
    assert is_sharp is True
    assert var_sharp > 110.0
    
    # Flat blurry image (low variance)
    arr_blur = np.full((256, 256), 200, dtype=np.uint8)
    img_blur = Image.fromarray(arr_blur)
    is_blurry, var_blur = preprocessor.check_sharpness(img_blur)
    assert is_blurry is False
    assert var_blur < 10.0

def test_division_illumination_normalization():
    preprocessor = OpticalPreprocessor()
    
    # Create image with artificial shadow ramp (256x256)
    base = np.full((256, 256), 220, dtype=np.float32)
    # Add dark left side
    ramp = np.linspace(0.4, 1.0, 256)[np.newaxis, :]
    shadowed = (base * ramp).astype(np.uint8)
    img = Image.fromarray(shadowed)
    
    normalized = preprocessor.division_illumination_normalization(img, sigma=20.0)
    norm_arr = np.array(normalized)
    
    # Normalized image should have significantly reduced horizontal brightness difference
    left_mean = np.mean(norm_arr[:, :50])
    right_mean = np.mean(norm_arr[:, -50:])
    assert abs(left_mean - right_mean) < 40
