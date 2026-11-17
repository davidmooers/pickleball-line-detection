import argparse
import csv
from pathlib import Path
from uuid import uuid4

import joblib

from serial_reader import SerialReader
from event_detector import EventDetector
from features import (
    prepare_waveform,
    extract_features,
    FeatureExtractionConfig
)
from raw_event_writer import save_raw_event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--threshold", type=float, default=20.0)
    args = parser.parse_args()

    project_directory = Path(__file__).resolve().parent
    model_path = project_directory / "classifier.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "Run train_classifier.py before classifying live."
        )

    saved_model = joblib.load(model_path)

    if saved_model["preprocessing_version"] != 1:
        raise ValueError("Model preprocessing version does not match.")

    model = saved_model["model"]

    session_id = uuid4().hex
    directory = project_directory / "live_events" / session_id
    directory.mkdir(parents=True, exist_ok=True)

    detector = EventDetector()
    detector.trigger_threshold = args.threshold
    detector.post_trigger_samples = 830

    config = FeatureExtractionConfig()
    reader = SerialReader(port=args.port)

    print(f"Loaded model: {saved_model['training_events']} training events.")
    print("Listening. Ctrl+C stops.")

    try:
        with (directory / "predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["raw_file", "predicted_label"])

            while True:
                sample = reader.read_sample()

                if sample is None:
                    continue

                event = detector.add_sample(sample)

                if event is None:
                    continue

                waveform = prepare_waveform(
                    event.samples,
                    event.sample_rate,
                    event.trigger_index
                )
                prediction = str(
                    model.predict(waveform[None, :])[0]
                )

                characteristics = extract_features(event, config)

                raw_path = save_raw_event(
                    directory=directory,
                    event=event,
                    characteristics=characteristics,
                    label="unlabelled",
                    source="live",
                    session_id=session_id
                )

                writer.writerow([raw_path.name, prediction])
                file.flush()

                print(prediction)

    except KeyboardInterrupt:
        print("\nStopped.")
        print(f"Missing serial samples: {reader.missing_samples}")

    finally:
        reader.serial.close()


if __name__ == "__main__":
    main()