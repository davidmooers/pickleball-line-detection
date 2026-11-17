from pathlib import Path

import joblib
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from features import prepare_waveform


project_directory = Path(__file__).resolve().parent
labels = ["non-pickleball", "pickleball"]


def load_events(directory):
    waveforms = []
    targets = []
    sessions = set()

    for path in sorted(directory.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as saved:
            label = str(saved["label"].item())

            if label not in labels:
                raise ValueError(f"Unexpected label in {path}: {label}")

            waveform = prepare_waveform(
                saved["samples"],
                float(saved["sample_rate"]),
                int(saved["trigger_index"])
            )

            sessions.add(str(saved["session_id"].item()))

        waveforms.append(waveform)
        targets.append(label)

    return np.asarray(waveforms), np.asarray(targets), sessions


def main():
    X_train, y_train, train_sessions = load_events(
        project_directory / "raw_events" / "train"
    )
    X_test, y_test, test_sessions = load_events(
        project_directory / "raw_events" / "test"
    )

    if train_sessions & test_sessions:
        raise ValueError("A session appears in both training and test data.")

    for label in labels:
        count = int(np.sum(y_train == label))
        print(f"Training {label}: {count}")

        if count < 3:
            raise ValueError(
                f"Record at least 3 training events for {label}."
            )

    model = KNeighborsClassifier(
        n_neighbors=3,
        weights="distance",
        algorithm="brute",
        metric="euclidean"
    )
    model.fit(X_train, y_train)

    model_path = project_directory / "classifier.joblib"

    joblib.dump(
        {
            "model": model,
            "preprocessing_version": 1,
            "training_events": len(y_train)
        },
        model_path
    )

    print(f"\nSaved model trained on {len(y_train)} events.")

    if len(y_test) == 0:
        print("No separate test recordings yet.")
        return

    predictions = model.predict(X_test)

    print(f"\nSeparate test events: {len(y_test)}")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    print("Rows: actual; columns: predicted")
    print("Label order:", labels)
    print(confusion_matrix(
        y_test, predictions, labels=labels
    ))


if __name__ == "__main__":
    main()