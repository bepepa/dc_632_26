import numpy as np
import pytest

from dc_632_26.match_filter import MatchFilter
from dc_632_26.pulses import (
    CosineSquaredPulse,
    HalfSinePulse,
    RectangularPulse,
    TrianglePulse,
)

NUM_SAMPLES = 64
RECT = RectangularPulse(NUM_SAMPLES)
RECT_MF = MatchFilter(RECT)

HSIN = HalfSinePulse(NUM_SAMPLES)
HSIN_MF = MatchFilter(HSIN)

SQCO = CosineSquaredPulse(NUM_SAMPLES)
SQCO_MF = MatchFilter(SQCO)

TRIA = TrianglePulse(NUM_SAMPLES)
TRIA_MF = MatchFilter(TRIA)

ALL_CASES = [
    (RECT, RECT_MF),
    (HSIN, HSIN_MF),
    (SQCO, SQCO_MF),
    (TRIA, TRIA_MF),
]


@pytest.mark.parametrize("pulse, mf", ALL_CASES)
class TestTaps:
    def test_taps_are_reversed_conjugate(self, pulse, mf):
        np.testing.assert_array_equal(mf.taps, np.conj(pulse.samples[::-1]))

    def test_num_taps_matches_pulse_length(self, pulse, mf):
        assert mf.num_taps == len(pulse.samples) == pulse.num_samples


@pytest.mark.parametrize("pulse, mf", ALL_CASES)
class TestPeakValue:
    def test_peak_value_equals_unit_energy(self, pulse, mf):
        corr = mf.filter(pulse.samples)
        assert corr[mf.delay] == pytest.approx(1.0)

    def test_peak_is_global_max_magnitude(self, pulse, mf):
        corr = mf.filter(pulse.samples)
        assert np.argmax(np.abs(corr)) == mf.delay


@pytest.mark.parametrize("pulse, mf", ALL_CASES)
class TestOutputShape:
    def test_output_length(self, pulse, mf):
        rng = np.random.default_rng(0)
        test_case = rng.integers(1, 100, size=5)
        for length in test_case:
            x = np.zeros(length)
            out = mf.filter(x)
            assert len(out) == length + mf.num_taps - 1

@pytest.mark.parametrize("pulse, mf", ALL_CASES)
class TestLinearity:
    def test_superposition(self, pulse, mf):
        rng = np.random.default_rng(0)
        x1 = rng.standard_normal(3 * mf.num_taps)
        x2 = rng.standard_normal(3 * mf.num_taps)
        a, b = 2.5, -1.3

        combined = mf.filter(a * x1 + b * x2)
        separate = a * mf.filter(x1) + b * mf.filter(x2)
        np.testing.assert_allclose(combined, separate, atol=1e-10)