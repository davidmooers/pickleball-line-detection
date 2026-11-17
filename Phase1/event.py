from dataclasses import dataclass
import numpy as np

@dataclass
class Event:
    samples: np.ndarray     # raw ADC samples
    sample_rate: float      # samples per second
    trigger_index: int      # sample index where event occurred