import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

@pytest.fixture
def visual_output_dir():
    output_dir = Path("test_image_out")
    output_dir.mkdir(exist_ok=True)
    return output_dir

@pytest.mark.visual
def test_plot_qpsk_constellation(visual_output_dir):
    """Visual inspection: QPSK points should form square"""
    symbols = QPSK()
    plt.scatter(symbols.real, symbols.imag)
    plt.savefig(visual_output_dir / "qpsk_constellation.png")
    plt.close()

@pytest.mark.visual
def test_signal_realization(visual_output_dir):
    """Visual inspection: Should show 10 symbol periods"""
    signal = generate_signal()
    plot_signal(signal)
    plt.savefig(visual_output_dir / "signal_realization.png")
    plt.close()