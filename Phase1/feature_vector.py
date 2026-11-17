import numpy as np

from features import EventCharacteristics


FEATURE_NAMES = [
    "peak_to_peak",
    "peak_amplitude",
    "num_peaks",
    "mean_peak_spacing",
    "rms",
    "energy",
    "duration",
    "rise_time",
    "decay_time",
    "zero_crossings",
    "dominant_frequency",
    "spectral_centroid",
    "spectral_bandwidth",
    "low_band_energy",
    "mid_band_energy",
    "high_band_energy",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
]


def to_feature_vector(
    characteristics: EventCharacteristics
) -> np.ndarray:

    return np.array([
        characteristics.peak_to_peak,
        characteristics.peak_amplitude,
        characteristics.num_peaks,
        characteristics.mean_peak_spacing,
        characteristics.rms,
        characteristics.energy,
        characteristics.duration,
        characteristics.rise_time,
        characteristics.decay_time,
        characteristics.zero_crossings,
        characteristics.dominant_frequency,
        characteristics.spectral_centroid,
        characteristics.spectral_bandwidth,
        characteristics.low_band_energy,
        characteristics.mid_band_energy,
        characteristics.high_band_energy,
        characteristics.low_band_ratio,
        characteristics.mid_band_ratio,
        characteristics.high_band_ratio,
    ], dtype=float)


def feature_dict(characteristics: EventCharacteristics) -> dict[str, float]:

    vector = to_feature_vector(characteristics)

    return {
        name: float(value)
        for name, value in zip(FEATURE_NAMES, vector)
    }