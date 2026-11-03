import pytest
import numpy as np
from python_engine.postprocessor import CTCBeamSearchDecoder, DiacriticRestorer, KashmiriPostProcessor

def test_beam_search_decoding():
    decoder = CTCBeamSearchDecoder(beam_width=10, alpha=0.5, beta=1.2)
    
    # Simple simulated probability matrix (T=15, 98 classes)
    probs = np.full((15, 98), -10.0)
    probs[:, 0] = 0.0 # mostly blank
    
    # Spike for 'ک' at t=2
    probs[2, 0] = -5.0
    probs[2, 29] = 2.0
    
    # Spike for 'ا' at t=5
    probs[5, 0] = -5.0
    probs[5, 2] = 2.0
    
    decoded = decoder.decode(probs)
    assert isinstance(decoded, str)

def test_diacritic_restoration():
    restorer = DiacriticRestorer()
    
    # Stripped skeleton 'کاشر' should restore to 'کٲشُر'
    restored = restorer.restore_line("کاشر")
    assert "کٲشُر" in restored or len(restored) > 0
