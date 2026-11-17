from collections import deque
import numpy as np

from event import Event

class EventDetector:

    def __init__(self):

        # number of samples used to decide whether a vibration occurred
        self.trigger_window = deque(maxlen=32)

        # samples kept from immediately before the trigger
        self.pre_trigger = deque(maxlen=80)

        # peak-to-peak amount required to start capturing an event
        self.trigger_threshold = 20

        # capture 100 ms after triggering at 8300 samples per second
        self.post_trigger_samples = 830

        self.waiting_for_rearm = False
        self.rearm_threshold = 5
        self.rearm_quiet_samples = 40
        self.rearm_quiet_count = 0

        # state of detector
        self.capturing = False

        # samples belonging to event currently being captured
        self.current_event_samples = []

        # number of samples collected since trigger occurred
        self.samples_after_trigger = 0

        # index first trigger for current event
        self.current_trigger_index = 0


    def add_sample(self, sample):

        self.trigger_window.append(sample)
        self.pre_trigger.append(sample)

        # wait until the signal stays quiet before allowing another event
        if self.waiting_for_rearm:
            peak_to_peak = max(self.trigger_window) - min(self.trigger_window)

            if peak_to_peak < self.rearm_threshold:
                self.rearm_quiet_count += 1
            else:
                self.rearm_quiet_count = 0

            if self.rearm_quiet_count >= self.rearm_quiet_samples:
                self.waiting_for_rearm = False
                self.rearm_quiet_count = 0

            return None


        # STATE 1 - CAPTURING AN EVENT
        if self.capturing:

            self.current_event_samples.append(sample)
            self.samples_after_trigger += 1

            # check if we collected enough samples after the trigger
            if self.samples_after_trigger >= self.post_trigger_samples:

                event = Event(
                    samples=np.array(self.current_event_samples),
                    sample_rate=8300.0,
                    trigger_index=self.current_trigger_index
                )

                # finish capture and require quiet before detecting another event
                self.capturing = False
                self.waiting_for_rearm = True
                self.rearm_quiet_count = 0

                self.current_event_samples = []
                self.samples_after_trigger = 0
                self.current_trigger_index = 0

                return event

            return None

        # STATE 2 - WAITING FOR EVENT

        # wait until the trigger window is full
        if len(self.trigger_window) < self.trigger_window.maxlen:
            return None

        peak_to_peak = (
            max(self.trigger_window) - min(self.trigger_window)
        )

        # check if vibration has crossed threshold (TODO: make this more robust against noise?)
        if peak_to_peak >= self.trigger_threshold:
                        
            # copy the samples that happened before the trigger
            self.current_event_samples = list(self.pre_trigger)

            # index of the trigger sample already included in the buffer
            self.current_trigger_index = len(self.current_event_samples) - 1

            # begin collecting post-trigger samples
            self.capturing = True
            self.samples_after_trigger = 0

        return None