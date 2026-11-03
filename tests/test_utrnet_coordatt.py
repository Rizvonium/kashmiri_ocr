import pytest
import numpy as np
from python_engine.utrnet_coordatt import KASHMIRI_VOCAB, NUM_CLASSES, CHAR2IDX, IDX2CHAR, NumPyUTRNetCoordAtt

def test_kashmiri_vocab_structure():
    assert NUM_CLASSES == 98
    assert KASHMIRI_VOCAB[0] == "<blank>"
    assert KASHMIRI_VOCAB[1] == " "
    
    # Critical Kashmiri characters and diacritics must be present
    assert "\u0655" in CHAR2IDX  # Arabic Hamza Below
    assert "\u065F" in CHAR2IDX  # Wavy Dagger
    assert "\u0620" in CHAR2IDX  # Kashmiri Yeh (ؠ)
    assert "\u06C4" in CHAR2IDX  # Waw with Ring (ۄ)
    assert "ٲ" in CHAR2IDX      # Alef with Maddah

def test_numpy_utrnet_forward():
    model = NumPyUTRNetCoordAtt(num_classes=98)
    dummy_line = np.ones((64, 256), dtype=np.float32)
    
    log_probs = model.forward(dummy_line)
    
    # Assert output shape (Time, Num_Classes)
    assert len(log_probs.shape) == 2
    assert log_probs.shape[1] == 98
    
    # Check that probabilities sum to ~1 in normal space (or <= 0 in log space)
    assert np.all(log_probs <= 0.0)
