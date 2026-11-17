"""
Piezo Visual Trainer v6 — aeon DTW k-NN
========================================

Uses:
    aeon.classification.distance_based.KNeighborsTimeSeriesClassifier

Instead of hand-written correlation thresholds, approved events are used to train
a k-nearest-neighbour time-series classifier with Dynamic Time Warping (DTW).

IMPORTANT:
Train BOTH classes:
    B = arm one PICKLEBALL training capture
    O = arm one OTHER / NON-PICKLEBALL training capture

Then:
    Y = approve the displayed candidate with the armed label
    N = reject it
    D = toggle live detection
    S = print status
    Q = quit

Arduino:
    Uses the 2-byte self-synchronizing binary stream at 250000 baud.

Install:
    pip install pyserial numpy matplotlib aeon
"""

import time
import threading
from collections import deque

import numpy as np
import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from aeon.classification.distance_based import (
    KNeighborsTimeSeriesClassifier,
)


# ============================================================
# CONFIG
# ============================================================

PORT = "COM3"
BAUD = 250000

ROLLING_BUFFER_SAMPLES = 40000
LIVE_DISPLAY_SAMPLES = 1500

PP_WINDOW = 32

PRE_SAMPLES = 80
POST_SAMPLES = 220
EVENT_SAMPLES = PRE_SAMPLES + POST_SAMPLES

TRAIN_CAPTURE_SECONDS = 2.5

# This is ONLY the sensitive event trigger.
# It does not decide whether something is a pickleball.
CANDIDATE_PP = 3

COOLDOWN_SAMPLES = 700

# Search enough recent history that short impacts cannot disappear
# between GUI refreshes.
DETECTION_LOOKBACK = 1200

# aeon k-NN settings
MAX_NEIGHBORS = 5

# Optional minimum confidence for printing PICKLEBALL.
# Start fairly permissive so we can see what the model is doing.
PICKLEBALL_PROBABILITY_THRESHOLD = 0.60


# ============================================================
# SHARED STATE
# ============================================================

samples = deque(maxlen=ROLLING_BUFFER_SAMPLES)
sample_lock = threading.Lock()

running = True

total_samples_received = 0
stream_rate = 0.0

bad_packets = 0
missing_samples = 0
last_sequence = None

# Training data
training_events = []
training_labels = []

pending_training_event = None
pending_training_label = None

training_armed = False
training_start_count = 0
training_start_time = 0.0
armed_label = None

# Model
classifier = None
model_ready = False

# Detection
detection_enabled = False
last_detection_index = -COOLDOWN_SAMPLES

# GUI review event
display_event = None
display_event_kind = "No event yet"
display_probability = None
display_prediction = None


# ============================================================
# SIGNAL HELPERS
# ============================================================

def center_event(event):
    """
    Remove the event's local DC baseline but preserve amplitude.
    We deliberately do NOT scale every waveform to unit amplitude:
    a tiny noise waveform should not become equivalent to a strong impact.
    """
    x = np.asarray(event, dtype=np.float64)

    baseline_region = x[:max(12, PRE_SAMPLES // 2)]
    baseline = np.median(baseline_region)

    return x - baseline


def event_features(event):
    raw = np.asarray(event, dtype=np.float64)
    x = center_event(raw)

    baseline = float(
        np.median(raw[:max(12, PRE_SAMPLES // 2)])
    )

    return {
        "baseline": baseline,
        "minimum": float(np.min(raw)),
        "maximum": float(np.max(raw)),
        "pp": float(np.max(raw) - np.min(raw)),
        "rms": float(np.sqrt(np.mean(x * x))),
        "peak": float(np.max(np.abs(x))),
    }


def recent_pp(data, window=PP_WINDOW):
    if len(data) < window:
        return 0

    x = np.asarray(data[-window:], dtype=np.int16)
    return int(np.max(x) - np.min(x))


def event_for_model(event):
    """
    aeon expects one event as shape:
        (channels, timepoints)

    We have one channel, so:
        (1, EVENT_SAMPLES)
    """
    return center_event(event)[None, :]


# ============================================================
# MODEL
# ============================================================

def rebuild_model():
    global classifier
    global model_ready

    model_ready = False
    classifier = None

    if len(training_events) < 2:
        return

    unique_labels = set(training_labels)

    # We want a real two-class classifier.
    if "pickleball" not in unique_labels or "other" not in unique_labels:
        return

    X = np.stack(
        [event_for_model(e) for e in training_events],
        axis=0,
    )

    y = np.asarray(training_labels)

    k = min(MAX_NEIGHBORS, len(training_events))

    # Prefer odd k when possible so votes are less likely to tie.
    if k > 1 and k % 2 == 0:
        k -= 1

    k = max(1, k)

    classifier = KNeighborsTimeSeriesClassifier(
        distance="dtw",
        n_neighbors=k,
        weights="distance",
        n_jobs=1,
    )

    classifier.fit(X, y)
    model_ready = True

    print(
        f"Model rebuilt: {len(training_events)} examples, "
        f"k={k}, classes={sorted(unique_labels)}"
    )


def classify_event(event):
    """
    Returns:
        prediction, pickleball_probability
    """
    if not model_ready or classifier is None:
        return None, None

    X = event_for_model(event)[None, :, :]

    prediction = str(classifier.predict(X)[0])

    probabilities = classifier.predict_proba(X)[0]
    classes = list(classifier.classes_)

    if "pickleball" in classes:
        p_index = classes.index("pickleball")
        pickleball_probability = float(probabilities[p_index])
    else:
        pickleball_probability = 0.0

    return prediction, pickleball_probability


# ============================================================
# BINARY SERIAL READER
# ============================================================

def serial_reader(ser):
    global running
    global total_samples_received
    global bad_packets
    global missing_samples
    global last_sequence

    buffer = bytearray()

    while running:
        try:
            waiting = ser.in_waiting
            chunk = ser.read(max(64, waiting))

            if chunk:
                buffer.extend(chunk)

            while len(buffer) >= 2:
                first = buffer[0]
                second = buffer[1]

                # First byte must have high bit set.
                if (first & 0x80) == 0:
                    del buffer[0]
                    bad_packets += 1
                    continue

                # Second byte must have high bit clear.
                if (second & 0x80) != 0:
                    del buffer[0]
                    bad_packets += 1
                    continue

                sequence = (first >> 3) & 0x0F

                adc_high = first & 0x07
                adc_low = second & 0x7F

                value = (adc_high << 7) | adc_low

                del buffer[:2]

                if value > 1023:
                    bad_packets += 1
                    continue

                if last_sequence is not None:
                    expected = (last_sequence + 1) & 0x0F

                    if sequence != expected:
                        gap = (sequence - expected) & 0x0F
                        missing_samples += gap

                last_sequence = sequence

                with sample_lock:
                    samples.append(value)
                    total_samples_received += 1

        except serial.SerialException as exc:
            print(f"\nSerial error: {exc}")
            running = False
            return


# ============================================================
# TRAINING
# ============================================================

def arm_training(label):
    global training_armed
    global training_start_count
    global training_start_time
    global pending_training_event
    global pending_training_label
    global armed_label

    if training_armed:
        print("Training is already armed.")
        return

    pending_training_event = None
    pending_training_label = None

    armed_label = label

    with sample_lock:
        training_start_count = total_samples_received

    training_start_time = time.time()
    training_armed = True

    if label == "pickleball":
        action = "DROP ONE PICKLEBALL"
    else:
        action = "MAKE ONE NON-PICKLEBALL EVENT"

    print(
        f"\nTRAINING ARMED [{label.upper()}]: "
        f"{action} within {TRAIN_CAPTURE_SECONDS:.1f} seconds."
    )


def finish_training_capture():
    global training_armed
    global pending_training_event
    global pending_training_label

    global display_event
    global display_event_kind
    global display_probability
    global display_prediction

    with sample_lock:
        end_count = total_samples_received
        data = np.asarray(list(samples), dtype=np.float64)

    captured_count = end_count - training_start_count
    training_armed = False

    if captured_count < EVENT_SAMPLES:
        print("Not enough samples were captured.")
        return

    captured_count = min(captured_count, len(data))
    interval = data[-captured_count:]

    search_start = PRE_SAMPLES
    search_end = len(interval) - POST_SAMPLES

    if search_end <= search_start:
        print("Capture interval was too short.")
        return

    baseline = np.median(interval)
    centered = interval - baseline

    safe = centered[search_start:search_end]

    peak_index = (
        search_start
        + int(np.argmax(np.abs(safe)))
    )

    start = peak_index - PRE_SAMPLES
    end = peak_index + POST_SAMPLES

    event = interval[start:end].astype(np.int16)

    pending_training_event = event
    pending_training_label = armed_label

    display_event = event
    display_event_kind = (
        f"TRAINING CANDIDATE [{armed_label.upper()}] "
        "— inspect, then Y/N"
    )
    display_probability = None
    display_prediction = None

    f = event_features(event)

    print(f"\nTRAINING CANDIDATE [{armed_label.upper()}]")
    print(f"  baseline = {f['baseline']:.1f}")
    print(f"  min      = {f['minimum']:.0f}")
    print(f"  max      = {f['maximum']:.0f}")
    print(f"  PP       = {f['pp']:.1f}")
    print(f"  RMS      = {f['rms']:.2f}")
    print(f"  peak     = {f['peak']:.1f}")
    print("Press Y to approve or N to reject.")


def approve_training_candidate():
    global pending_training_event
    global pending_training_label
    global display_event_kind

    if pending_training_event is None:
        print("No training candidate is waiting.")
        return

    label = pending_training_label

    training_events.append(
        pending_training_event.copy()
    )
    training_labels.append(label)

    pending_training_event = None
    pending_training_label = None

    count_for_label = training_labels.count(label)

    display_event_kind = (
        f"APPROVED {label.upper()} EVENT "
        f"#{count_for_label}"
    )

    print(
        f"Approved {label} training example "
        f"#{count_for_label}."
    )

    rebuild_model()


def reject_training_candidate():
    global pending_training_event
    global pending_training_label
    global display_event_kind

    if pending_training_event is None:
        print("No training candidate is waiting.")
        return

    label = pending_training_label

    pending_training_event = None
    pending_training_label = None

    display_event_kind = (
        f"REJECTED {label.upper()} CANDIDATE"
    )

    print("Rejected. It was not added to training data.")


# ============================================================
# LIVE CANDIDATE EXTRACTION
# ============================================================

def extract_recent_candidate():
    """
    Search ALL recent samples for a short peak-to-peak excursion.
    This avoids missing impacts between GUI refreshes.
    """
    with sample_lock:
        needed = max(
            DETECTION_LOOKBACK + POST_SAMPLES,
            EVENT_SAMPLES + 200,
        )

        if len(samples) < needed:
            return None

        data = np.asarray(
            list(samples),
            dtype=np.int16,
        )

        global_end = total_samples_received

    lookback_len = min(
        len(data),
        DETECTION_LOOKBACK + PRE_SAMPLES + POST_SAMPLES,
    )

    local = data[-lookback_len:].astype(np.float64)

    windows = np.lib.stride_tricks.sliding_window_view(
        local,
        PP_WINDOW,
    )

    rolling_pp = (
        np.max(windows, axis=1)
        - np.min(windows, axis=1)
    )

    max_trigger_end = len(local) - POST_SAMPLES

    valid_last_window_index = (
        max_trigger_end - PP_WINDOW
    )

    if valid_last_window_index <= 0:
        return None

    valid_pp = rolling_pp[
        :valid_last_window_index + 1
    ]

    strongest_window_index = int(
        np.argmax(valid_pp)
    )

    strongest_pp = float(
        valid_pp[strongest_window_index]
    )

    if strongest_pp < CANDIDATE_PP:
        return None

    window_start = strongest_window_index
    window_end = window_start + PP_WINDOW

    baseline_region_start = max(
        0,
        window_start - PRE_SAMPLES,
    )

    if window_start > baseline_region_start:
        baseline_region = local[
            baseline_region_start:window_start
        ]
    else:
        baseline_region = local[:PRE_SAMPLES]

    baseline = np.median(baseline_region)

    trigger_window = local[
        window_start:window_end
    ]

    centered_trigger = (
        trigger_window - baseline
    )

    peak_in_window = int(
        np.argmax(np.abs(centered_trigger))
    )

    peak_index = (
        window_start + peak_in_window
    )

    start = peak_index - PRE_SAMPLES
    end = peak_index + POST_SAMPLES

    if start < 0 or end > len(local):
        return None

    event = local[
        start:end
    ].astype(np.int16)

    event_global_index = (
        global_end
        - len(local)
        + peak_index
    )

    return (
        event,
        event_global_index,
        int(strongest_pp),
    )


# ============================================================
# LIVE DETECTION
# ============================================================

def process_detection():
    global last_detection_index

    global display_event
    global display_event_kind
    global display_probability
    global display_prediction

    if not detection_enabled:
        return

    if not model_ready:
        return

    result = extract_recent_candidate()

    if result is None:
        return

    event, event_index, trigger_pp = result

    if (
        abs(event_index - last_detection_index)
        < COOLDOWN_SAMPLES
    ):
        return

    last_detection_index = event_index

    prediction, p_pickleball = classify_event(event)

    f = event_features(event)

    display_event = event
    display_prediction = prediction
    display_probability = p_pickleball

    # Use the probability threshold as an extra guard.
    # The model prediction is still shown in the console.
    is_pickleball = (
        prediction == "pickleball"
        and p_pickleball is not None
        and p_pickleball >= PICKLEBALL_PROBABILITY_THRESHOLD
    )

    if is_pickleball:
        display_event_kind = (
            "LIVE DETECTION: PICKLEBALL"
        )

        print(
            f"PICKLEBALL  "
            f"p={p_pickleball:.3f}  "
            f"PP={f['pp']:.1f}  "
            f"triggerPP={trigger_pp}  "
            f"RMS={f['rms']:.2f}"
        )

    else:
        display_event_kind = (
            "LIVE DETECTION: OTHER"
        )

        print(
            f"OTHER  "
            f"model={prediction}  "
            f"p_pickleball={p_pickleball:.3f}  "
            f"PP={f['pp']:.1f}  "
            f"triggerPP={trigger_pp}  "
            f"RMS={f['rms']:.2f}"
        )


# ============================================================
# STATUS
# ============================================================

def print_status():
    with sample_lock:
        data = list(samples)

    pp32 = recent_pp(data)

    pickleball_count = training_labels.count(
        "pickleball"
    )

    other_count = training_labels.count(
        "other"
    )

    print("\n--- STATUS ---")
    print(
        f"Stream rate: "
        f"{stream_rate:.0f} samples/s"
    )
    print(
        f"Current 32-sample PP: {pp32}"
    )
    print(
        f"Pickleball training events: "
        f"{pickleball_count}"
    )
    print(
        f"Other training events: "
        f"{other_count}"
    )
    print(
        f"Model ready: "
        f"{'YES' if model_ready else 'NO'}"
    )
    print(
        f"Detection: "
        f"{'ON' if detection_enabled else 'OFF'}"
    )
    print(
        f"Bad packets: {bad_packets}"
    )
    print(
        f"Missing samples: {missing_samples}"
    )
    print(
        f"Candidate PP threshold: "
        f"{CANDIDATE_PP}"
    )
    print(
        f"Pickleball probability threshold: "
        f"{PICKLEBALL_PROBABILITY_THRESHOLD:.2f}"
    )

    if training_events:
        print("Training examples:")

        for i, (event, label) in enumerate(
            zip(training_events, training_labels),
            1,
        ):
            f = event_features(event)

            print(
                f"  #{i} {label:10s} "
                f"PP={f['pp']:.1f} "
                f"RMS={f['rms']:.2f}"
            )

    print("--------------")


# ============================================================
# GUI
# ============================================================

plt.close("all")

fig, (ax_live, ax_pp, ax_event) = plt.subplots(
    3,
    1,
    figsize=(12, 9),
    gridspec_kw={
        "height_ratios": [2.0, 1.0, 2.0]
    },
)

fig.canvas.manager.set_window_title(
    "Piezo Trainer v6 — aeon DTW k-NN"
)

live_line, = ax_live.plot([], [])

live_baseline_line = ax_live.axhline(
    0,
    linestyle="--",
)

pp_line, = ax_pp.plot([], [])

pp_threshold_line = ax_pp.axhline(
    CANDIDATE_PP,
    linestyle="--",
)

event_line, = ax_event.plot(
    [],
    [],
    linewidth=1.5,
)

event_zero_line = ax_event.axhline(
    0,
    linestyle="--",
)

training_overlay_lines = []

ax_live.set_title("LIVE RAW A0")
ax_live.set_xlabel("Recent sample")
ax_live.set_ylabel("ADC")

ax_pp.set_title(
    f"ROLLING {PP_WINDOW}-SAMPLE PEAK-TO-PEAK"
)
ax_pp.set_xlabel("Recent sample")
ax_pp.set_ylabel("PP")

ax_event.set_title(
    "EVENT REVIEW — no candidate yet"
)
ax_event.set_xlabel("Event sample")
ax_event.set_ylabel(
    "ADC relative to baseline"
)

status_text = fig.text(
    0.01,
    0.01,
    "",
    fontsize=9,
    verticalalignment="bottom",
)

plt.tight_layout(
    rect=[0, 0.12, 1, 1]
)


def update_gui(_frame):
    global training_overlay_lines

    if training_armed:
        elapsed = (
            time.time()
            - training_start_time
        )

        if elapsed >= TRAIN_CAPTURE_SECONDS:
            finish_training_capture()

    process_detection()

    with sample_lock:
        live = np.asarray(
            list(samples)[
                -LIVE_DISPLAY_SAMPLES:
            ],
            dtype=np.float64,
        )

    baseline = 0.0
    live_display_pp = 0.0
    pp32_now = 0

    if len(live) > 0:
        x = np.arange(len(live))

        live_line.set_data(x, live)

        baseline = float(
            np.median(live)
        )

        live_baseline_line.set_ydata(
            [baseline, baseline]
        )

        low = float(np.min(live))
        high = float(np.max(live))

        live_display_pp = high - low

        pad = max(
            4.0,
            (high - low) * 0.20,
        )

        ax_live.set_xlim(
            0,
            max(10, len(live) - 1),
        )

        ax_live.set_ylim(
            low - pad,
            high + pad,
        )

        if len(live) >= PP_WINDOW:
            windows = (
                np.lib.stride_tricks
                .sliding_window_view(
                    live,
                    PP_WINDOW,
                )
            )

            rolling_pp = (
                np.max(windows, axis=1)
                - np.min(windows, axis=1)
            )

            pp_x = np.arange(
                PP_WINDOW - 1,
                len(live),
            )

            pp_line.set_data(
                pp_x,
                rolling_pp,
            )

            pp32_now = int(
                rolling_pp[-1]
            )

            pp_high = max(
                float(np.max(rolling_pp)),
                CANDIDATE_PP + 1,
            )

            ax_pp.set_xlim(
                0,
                max(10, len(live) - 1),
            )

            ax_pp.set_ylim(
                -0.25,
                pp_high
                + max(
                    1.0,
                    pp_high * 0.15,
                ),
            )

    # Remove old training overlays.
    for line in training_overlay_lines:
        try:
            line.remove()
        except Exception:
            pass

    training_overlay_lines = []

    # Draw approved examples faintly.
    for event, label in zip(
        training_events,
        training_labels,
    ):
        centered = center_event(event)

        overlay, = ax_event.plot(
            np.arange(len(centered)),
            centered,
            alpha=0.12,
            linewidth=1.0,
        )

        training_overlay_lines.append(
            overlay
        )

    if display_event is not None:
        centered = center_event(
            display_event
        )

        event_x = np.arange(
            len(centered)
        )

        event_line.set_data(
            event_x,
            centered,
        )

        low = float(
            np.min(centered)
        )

        high = float(
            np.max(centered)
        )

        pad = max(
            2.0,
            (high - low) * 0.20,
        )

        ax_event.set_xlim(
            0,
            len(centered) - 1,
        )

        ax_event.set_ylim(
            low - pad,
            high + pad,
        )

        f = event_features(
            display_event
        )

        title = (
            f"{display_event_kind}\n"
            f"baseline={f['baseline']:.1f}  "
            f"PP={f['pp']:.1f}  "
            f"RMS={f['rms']:.2f}"
        )

        if display_probability is not None:
            title += (
                f"  model={display_prediction}  "
                f"P(ball)="
                f"{display_probability:.3f}"
            )

        ax_event.set_title(title)

    else:
        event_line.set_data([], [])

        ax_event.set_title(
            "EVENT REVIEW — "
            "no candidate yet"
        )

    pickleball_count = (
        training_labels.count(
            "pickleball"
        )
    )

    other_count = (
        training_labels.count(
            "other"
        )
    )

    status = (
        f"Stream: ~{stream_rate:.0f} samples/s   |   "
        f"Baseline: {baseline:.1f}   |   "
        f"Current {PP_WINDOW}-sample PP: {pp32_now}   |   "
        f"Displayed-window PP: {live_display_pp:.1f}\n"
        f"Training: {pickleball_count} ball / "
        f"{other_count} other   |   "
        f"Model: {'READY' if model_ready else 'NEEDS BOTH CLASSES'}   |   "
        f"Detection: {'ON' if detection_enabled else 'OFF'}   |   "
        f"bad: {bad_packets}   missing: {missing_samples}\n"
        f"B=train BALL   O=train OTHER   "
        f"Y=approve   N=reject   D=detection   "
        f"S=status   Q=quit"
    )

    if training_armed:
        remaining = max(
            0.0,
            TRAIN_CAPTURE_SECONDS
            - (
                time.time()
                - training_start_time
            ),
        )

        status += (
            f"\nTRAINING ARMED [{armed_label.upper()}] "
            f"— perform event now "
            f"({remaining:.1f}s remaining)"
        )

    elif pending_training_event is not None:
        status += (
            f"\nCANDIDATE [{pending_training_label.upper()}] "
            "WAITING — inspect bottom graph, "
            "then Y or N"
        )

    status_text.set_text(status)

    return (
        live_line,
        pp_line,
        event_line,
        status_text,
    )


def on_key(event):
    global running
    global detection_enabled

    if event.key is None:
        return

    key = event.key.lower()

    if key == "b":
        arm_training("pickleball")

    elif key == "o":
        arm_training("other")

    elif key == "y":
        approve_training_candidate()

    elif key == "n":
        reject_training_candidate()

    elif key == "d":
        if not model_ready:
            print(
                "Train and approve at least "
                "one PICKLEBALL and one OTHER "
                "example first."
            )
        else:
            detection_enabled = (
                not detection_enabled
            )

            print(
                f"Detection "
                f"{'ON' if detection_enabled else 'OFF'}."
            )

    elif key == "s":
        print_status()

    elif key == "q":
        running = False
        plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    global running
    global stream_rate

    print(
        f"Opening {PORT} at "
        f"{BAUD} baud..."
    )

    try:
        ser = serial.Serial(
            PORT,
            BAUD,
            timeout=0.05,
        )

    except serial.SerialException as exc:
        print(
            f"Could not open {PORT}: {exc}"
        )
        return

    time.sleep(1.0)
    ser.reset_input_buffer()

    reader = threading.Thread(
        target=serial_reader,
        args=(ser,),
        daemon=True,
    )

    reader.start()

    print(
        "Binary A0 stream connected."
    )

    print(
        "This version uses aeon DTW k-NN."
    )

    print(
        "B=train pickleball, "
        "O=train other, "
        "Y=approve, N=reject, "
        "D=detection, S=status, Q=quit"
    )

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )

    def rate_worker():
        global stream_rate

        last_time = time.time()
        last_count = 0

        while running:
            time.sleep(1.0)

            now = time.time()

            with sample_lock:
                count = (
                    total_samples_received
                )

            elapsed = now - last_time

            if elapsed > 0:
                stream_rate = (
                    count - last_count
                ) / elapsed

            last_count = count
            last_time = now

    threading.Thread(
        target=rate_worker,
        daemon=True,
    ).start()

    animation = FuncAnimation(
        fig,
        update_gui,
        interval=50,
        blit=False,
        cache_frame_data=False,
    )

    try:
        plt.show()

    finally:
        running = False
        ser.close()
        print("Stopped.")


if __name__ == "__main__":
    main()
