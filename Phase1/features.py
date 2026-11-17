from dataclasses import dataclass
import numpy as np

from event import Event

from scipy.signal import find_peaks


@dataclass
class EventCharacteristics:
    baseline: float
    peak_to_peak: float
    peak_amplitude: float
    peak_index: int
    num_peaks: int
    mean_peak_spacing: float
    rms: float
    energy: float
    duration: float
    rise_time: float
    decay_time: float
    zero_crossings: int

    dominant_frequency: float
    spectral_centroid: float
    spectral_bandwidth: float
    low_band_energy: float
    mid_band_energy: float
    high_band_energy: float
    low_band_ratio: float
    mid_band_ratio: float
    high_band_ratio: float

@dataclass
class FeatureExtractionConfig:
    activity_threshold: float = 3.0
    quiet_sample_count: int = 20
    peak_prominence: float = 2.0
    zero_crossing_threshold: float = 1.0

    low_band_max_hz: float = 150.0
    mid_band_max_hz: float = 500.0


def extract_features(event: Event, config: FeatureExtractionConfig) -> EventCharacteristics:

    samples = event.samples.astype(float)

    baseline_samples = samples[:event.trigger_index]
    baseline = np.median(baseline_samples)

    centered = samples - baseline
    if len(centered) == 0:
        raise ValueError("Cannot extract features from an empty event.")

    peak_to_peak = np.max(samples) - np.min(samples)

    peak_amplitude = np.max(
        np.abs(centered)
    )

    peak_index = int(np.argmax(np.abs(centered)))

    rms = np.sqrt(np.mean(centered ** 2))

    energy = np.sum(centered ** 2)

    # duration
    active_indices = np.where(np.abs(centered) >= config.activity_threshold)[0]

    if len(active_indices) > 0:
        duration = (active_indices[-1] - active_indices[0]) / event.sample_rate
    else:
        duration = 0.0

    # rise time
    if len(active_indices) > 0:
        rise_time = (
            (peak_index - active_indices[0]) / event.sample_rate
            if peak_index >= active_indices[0]
            else 0.0
        )
    else:
        rise_time = 0.0

    # decay
    decay_end_index = None
    quiet_count = 0

    for i in range(peak_index, len(centered)):

        if abs(centered[i]) < config.activity_threshold:
            quiet_count += 1
        else:
            quiet_count = 0

        if quiet_count >= config.quiet_sample_count:
            decay_end_index = i - config.quiet_sample_count + 1
            break

    if decay_end_index is not None:
        decay_time = (decay_end_index - peak_index) / event.sample_rate
    else:
        decay_time = (len(centered) - 1 - peak_index) / event.sample_rate

    # zero crossings
    if len(active_indices) > 0:
        signal_region = centered[
            active_indices[0] : active_indices[-1] + 1
        ]
    else:
        signal_region = centered

    zero_crossings = 0
    last_sign = 0

    for sample in signal_region:

        if sample > config.zero_crossing_threshold:
            current_sign = 1

        elif sample < -config.zero_crossing_threshold:
            current_sign = -1

        else:
            continue

        if last_sign != 0 and current_sign != last_sign:
            zero_crossings += 1

        last_sign = current_sign


    peaks, _ = find_peaks(
        np.abs(centered),
        prominence=config.peak_prominence
    )

    num_peaks = int(len(peaks))

    if len(peaks) >= 2:
        mean_peak_spacing = float(
            np.mean(np.diff(peaks)) / event.sample_rate
        )
    else:
        mean_peak_spacing = 0.0

    # frequency-domain features
    # analyze the first 30 ms after the trigger consistently
    frequency_start = event.trigger_index
    frequency_length = int(round(0.030 * event.sample_rate))

    frequency_signal = centered[
        frequency_start : frequency_start + frequency_length
    ]

    if len(frequency_signal) < 2:
        raise ValueError("Not enough post-trigger samples for frequency analysis.")

    window = np.hanning(len(frequency_signal))
    windowed = frequency_signal * window

    spectrum = np.fft.rfft(windowed)
    magnitudes = np.abs(spectrum)

    frequencies = np.fft.rfftfreq(
        len(frequency_signal),
        d=1.0 / event.sample_rate
    )

    # Ignore DC component when finding dominant frequency
    if len(magnitudes) > 1:
        dominant_index = int(np.argmax(magnitudes[1:]) + 1)
        dominant_frequency = float(frequencies[dominant_index])
    else:
        dominant_frequency = 0.0

    magnitude_sum = np.sum(magnitudes)

    if magnitude_sum > 0:
        spectral_centroid = float(
            np.sum(frequencies * magnitudes) / magnitude_sum
        )

        spectral_bandwidth = float(
            np.sqrt(
                np.sum(
                    ((frequencies - spectral_centroid) ** 2) * magnitudes
                )
                / magnitude_sum
            )
        )
    else:
        spectral_centroid = 0.0
        spectral_bandwidth = 0.0

    power = magnitudes ** 2

    low_mask = frequencies < config.low_band_max_hz

    mid_mask = (
        (frequencies >= config.low_band_max_hz)
        & (frequencies < config.mid_band_max_hz)
    )

    high_mask = frequencies >= config.mid_band_max_hz

    low_band_energy = float(np.sum(power[low_mask]))
    mid_band_energy = float(np.sum(power[mid_mask]))
    high_band_energy = float(np.sum(power[high_mask]))

    total_band_energy = (
        low_band_energy
        + mid_band_energy
        + high_band_energy
    )

    if total_band_energy > 0:
        low_band_ratio = low_band_energy / total_band_energy
        mid_band_ratio = mid_band_energy / total_band_energy
        high_band_ratio = high_band_energy / total_band_energy
    else:
        low_band_ratio = 0.0
        mid_band_ratio = 0.0
        high_band_ratio = 0.0

    return EventCharacteristics(
        baseline=baseline,
        peak_to_peak=peak_to_peak,
        peak_amplitude=peak_amplitude,
        peak_index=peak_index,
        num_peaks=num_peaks,
        mean_peak_spacing=mean_peak_spacing,
        rms=rms,
        energy=energy,
        duration=duration,
        rise_time=rise_time,
        decay_time=decay_time,
        zero_crossings=zero_crossings,
        dominant_frequency=dominant_frequency,
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
        low_band_energy=low_band_energy,
        mid_band_energy=mid_band_energy,
        high_band_energy=high_band_energy,
        low_band_ratio=low_band_ratio,
        mid_band_ratio=mid_band_ratio,
        high_band_ratio=high_band_ratio
    )

def prepare_waveform(samples, sample_rate, trigger_index):
    samples = np.asarray(samples, dtype=float)

    if not np.isclose(sample_rate, 8300.0):
        raise ValueError("Expected an 8300 Hz sample rate.")

    if not 0 < trigger_index < len(samples):
        raise ValueError("Invalid trigger index.")

    baseline = np.median(samples[:trigger_index])
    length = round(0.030 * sample_rate)

    waveform = (
        samples[trigger_index:trigger_index + length] - baseline
    )

    if len(waveform) != length:
        raise ValueError("Not enough post-trigger samples.")

    norm = np.linalg.norm(waveform)

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid or empty waveform.")

    return waveform / norm