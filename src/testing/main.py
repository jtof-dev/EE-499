import serial


def main():
    # 1. Connect to the virtual serial port
    try:
        # MindWave Mobile operates at 57600 baud
        ser = serial.Serial("/dev/rfcomm0", 57600, timeout=1)
        print("Connected to MindWave! Reading data... (Press Ctrl+C to stop)")
        print("-" * 50)
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("Make sure the headset is on, paired, and bound to /dev/rfcomm0.")
        return

    try:
        while True:
            # 2. Wait for the sync bytes (0xAA 0xAA) which indicate a new packet
            if ser.read(1) == b"\xaa":
                if ser.read(1) == b"\xaa":
                    # 3. Read the payload length
                    p_len_byte = ser.read(1)
                    if not p_len_byte:
                        continue

                    p_len = p_len_byte[0]
                    # Packets larger than 169 bytes are invalid
                    if p_len > 169:
                        continue

                    # 4. Read the actual payload and the checksum
                    payload = ser.read(p_len)
                    checksum = ser.read(1)[0]

                    # 5. Verify the checksum to ensure data isn't corrupted
                    calculated_checksum = (~(sum(payload) & 0xFF)) & 0xFF
                    if calculated_checksum == checksum:
                        parse_payload(payload)

    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        if "ser" in locals() and ser.is_open:
            ser.close()
            print("Serial connection closed cleanly.")


def parse_payload(payload):
    """Parses the ThinkGear protocol payload into readable values."""
    i = 0
    while i < len(payload):
        code = payload[i]
        i += 1

        # Extended code level (skip)
        if code == 0x55:
            continue

        # Single-byte data values
        if code < 0x80:
            value = payload[i]
            i += 1
            if code == 0x02:
                # 0 means perfect signal. 200 means the headset is off.
                if value > 0:
                    print(f"[{value}/200] Adjust headset for better contact...")
            elif code == 0x04:
                print(f"Attention:  {value}")
            elif code == 0x05:
                print(f"Meditation: {value}")

        # Multi-byte data values
        else:
            v_len = payload[i]
            i += 1

            # Code 0x80 is the Raw EEG signal
            if code == 0x80 and v_len == 2:
                raw_val = (payload[i] << 8) | payload[i + 1]
                # Convert to a signed 16-bit integer
                if raw_val >= 32768:
                    raw_val -= 65536
                print(f"Raw EEG: {raw_val}")

            # Skip past any other multi-byte values we aren't using yet
            i += v_len


if __name__ == "__main__":
    main()
