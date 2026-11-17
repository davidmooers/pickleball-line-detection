import serial

class SerialReader:

    def __init__(self, port="COM3", baud_rate=250000):
        self.serial = serial.Serial(
            port,
            baud_rate,
            timeout=0.1
        )

        # temporarily store bytes from Arduino
        self.buffer = bytearray()

        # sequence number of last valid packet
        self.last_sequence = None

        # diagnostic counters
        self.bad_packets = 0
        self.missing_samples = 0

    def read_sample(self):
        """ 
        Attempts to read one valid ADC sample from Arduino

        Returns:
            int -> ADC value from 0 to 1023
            None -> no complete sample available
        """

        # read all bytes waiting in buffer
        waiting = self.serial.in_waiting

        if waiting > 0:
            s = self.serial.read(waiting)
            self.buffer.extend(s)

        # one sample is 2 bytes
        while len(self.buffer) >= 2:
            first = self.buffer[0]
            second = self.buffer[1]

            # ensure first byte begins with 1xxxxxxx
            if (first & 0x80) == 0:
                del self.buffer[0]
                self.bad_packets += 1
                continue

            # ensure second byte begins with 0xxxxxxx
            if (second & 0x80) != 0:
                del self.buffer[0]
                self.bad_packets += 1
                continue

            # extract the packet
            sequence = (first >> 3) & 0x0F  # 4-bit packet number
            adc_high = first & 0x07         # upper 3 bits of ADC
            adc_low = second & 0x7F         # lower 7 bits of ADC

            value = (adc_high << 7) | adc_low

            del self.buffer[:2]

            # check if we missed any samples by comparing packet sequence number with last sequence number
            if self.last_sequence is not None:

                expected = (self.last_sequence + 1) & 0x0F

                if sequence != expected:
                    missing = (sequence - expected) & 0x0F
                    self.missing_samples += missing

            # update sequence counter
            self.last_sequence = sequence

            return value

        return None

    