import argparse
from pathlib import Path
from uuid import uuid4

from serial_reader import SerialReader
from event_detector import EventDetector
from features import extract_features, FeatureExtractionConfig
from raw_event_writer import save_raw_event


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        required=True,
        choices=["pickleball", "footstep", "paddle_tap", "background"]
    )
    parser.add_argument(
        "--split",
        choices=["train", "test"],
        default="train"
    )
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--distance-cm", type=float)
    parser.add_argument("--threshold", type=float, default=20.0)
    args = parser.parse_args()

    label = (
        "pickleball"
        if args.source == "pickleball"
        else "non-pickleball"
    )

    # Include the distance in the saved session metadata.
    distance = (
        f"{args.distance_cm:g}cm"
        if args.distance_cm is not None
        else "unspecified"
    )
    session_id = f"{args.source}_{distance}_{uuid4().hex}"

    project_directory = Path(__file__).resolve().parent
    directory = (
        project_directory / "raw_events" / args.split / session_id
    )

    detector = EventDetector()
    detector.trigger_threshold = args.threshold
    detector.post_trigger_samples = 830

    config = FeatureExtractionConfig()
    reader = SerialReader(port=args.port)
    count = 0

    print(f"Recording: {args.source}")
    print(f"Dataset: {args.split}")
    print(f"Session: {session_id}")
    print("Count physical impacts separately. Ctrl+C stops.")

    try:
        while True:
            sample = reader.read_sample()

            if sample is None:
                continue

            event = detector.add_sample(sample)

            if event is None:
                continue

            characteristics = extract_features(event, config)

            save_raw_event(
                directory=directory,
                event=event,
                characteristics=characteristics,
                label=label,
                source=args.source,
                session_id=session_id
            )

            count += 1
            print(f"Saved event {count}")

    except KeyboardInterrupt:
        print(f"\nSaved {count} detected events.")
        print(f"Missing serial samples: {reader.missing_samples}")
        print(f"Bad serial bytes: {reader.bad_packets}")

    finally:
        reader.serial.close()


if __name__ == "__main__":
    main()