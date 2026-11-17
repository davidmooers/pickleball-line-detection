const int SENSOR_PIN = A0;

uint8_t sequenceNumber = 0;

void setup()
{
    Serial.begin(250000);
}

void loop()
{
    uint16_t value = analogRead(SENSOR_PIN);

    /*
      Packet is exactly 2 bytes:

      BYTE 1:
        bit 7    = 1 (start marker)
        bits 6-3 = 4-bit sequence number
        bits 2-0 = upper 3 ADC bits

      BYTE 2:
        bit 7    = 0
        bits 6-0 = lower 7 ADC bits

      This lets Python recover alignment automatically.
    */

    // Serial.print(value);
    // Serial.print(" : ");
    // Serial.print(value);
    // Serial.print('\n');

    uint8_t first =
        0x80 |
        ((sequenceNumber & 0x0F) << 3) |
        ((value >> 7) & 0x07);

    uint8_t second =
        value & 0x7F;

    Serial.write(first);
    Serial.write(second);

    sequenceNumber =
        (sequenceNumber + 1) & 0x0F;
}