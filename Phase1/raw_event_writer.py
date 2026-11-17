from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from event import Event
from features import EventCharacteristics
from feature_vector import FEATURE_NAMES, to_feature_vector


def save_raw_event(
    directory,
    event: Event,
    characteristics: EventCharacteristics,
    label: str,
    source: str,
    session_id: str
) -> Path:

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    event_id = uuid4().hex
    output_path = directory / f"{event_id}.npz"

    np.savez_compressed(
        output_path,
        event_id=event_id,
        session_id=session_id,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        label=label,
        source=source,
        samples=np.asarray(event.samples),
        sample_rate=event.sample_rate,
        trigger_index=event.trigger_index,
        feature_names=np.array(FEATURE_NAMES),
        feature_values=to_feature_vector(characteristics)
    )

    return output_path