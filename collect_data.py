#!/usr/bin/env python3
"""
Gesture Data Collector
======================
Captures IMU data streamed from an STM32 + MPU-6050 over a serial (COM) port
and saves it to a CSV file compatible with Edge Impulse.

Usage:
    python collect_data.py

The script will prompt you for:
    1. COM port  (e.g. COM3 on Windows, /dev/ttyACM0 on Linux/Mac)
    2. Baud rate (default 115200)
    3. Output filename (e.g. circle_1.csv)

Press Ctrl+C to stop recording and close the file cleanly.
"""

import serial
import serial.tools.list_ports
import csv
import sys
import os
from datetime import datetime


# ── Helpers ──────────────────────────────────────────────────────────────────

def list_serial_ports():
    """Print all available serial ports to help the user pick one."""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("  (no serial ports detected)")
        return
    for p in ports:
        print(f"  {p.device:15s}  {p.description}")


def validate_line(line: str) -> bool:
    """Return True if the line looks like valid sensor data (7 comma-separated numbers)."""
    parts = line.split(",")
    if len(parts) != 7:
        return False
    try:
        [float(x) for x in parts]
        return True
    except ValueError:
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Gesture Data Collector  —  STM32 + MPU-6050")
    print("=" * 55)
    print()

    # --- 1. Choose COM port ---
    print("Available serial ports:")
    list_serial_ports()
    print()

    port = input("Enter COM port (e.g. COM3 or /dev/ttyACM0): ").strip()
    if not port:
        print("No port entered. Exiting.")
        sys.exit(1)

    baud_input = input("Baud rate [115200]: ").strip()
    baud = int(baud_input) if baud_input else 115200

    # --- 2. Choose output file ---
    filename = input("Output filename (e.g. circle_1.csv): ").strip()
    if not filename:
        filename = f"gesture_{datetime.now():%Y%m%d_%H%M%S}.csv"
        print(f"  → using default: {filename}")
    if not filename.endswith(".csv"):
        filename += ".csv"

    # --- 3. Open serial port ---
    try:
        ser = serial.Serial(port, baud, timeout=1)
        print(f"\nConnected to {port} @ {baud} baud.")
    except serial.SerialException as e:
        print(f"\nERROR: Could not open {port}: {e}")
        sys.exit(1)

    # Flush any stale data sitting in the buffer
    ser.reset_input_buffer()

    # --- 4. Record to CSV ---
    header = ["timestamp", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"]
    sample_count = 0

    print(f"Recording to '{filename}'...")
    print("Press and hold the BLUE BUTTON on the Nucleo to stream data.")
    print("Press Ctrl+C to stop recording.\n")

    try:
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            while True:
                raw = ser.readline()
                if not raw:
                    continue  # timeout, no data (button not pressed)

                try:
                    line = raw.decode("ascii", errors="ignore").strip()
                except UnicodeDecodeError:
                    continue

                if not line:
                    continue

                if not validate_line(line):
                    # Might be a debug message — print it but don't save
                    print(f"  [info] {line}")
                    continue

                values = line.split(",")
                writer.writerow(values)
                sample_count += 1

                # Live feedback every 10 samples (~200 ms at 50 Hz)
                if sample_count % 10 == 0:
                    print(f"\r  Samples collected: {sample_count}", end="", flush=True)

    except KeyboardInterrupt:
        print()  # newline after the \r counter

    finally:
        ser.close()
        print(f"\nDone. {sample_count} samples saved to '{filename}'")

        if sample_count > 0:
            duration_s = sample_count * 0.02  # 50 Hz → 20 ms per sample
            print(f"  Approximate duration: {duration_s:.1f} s")
            file_size = os.path.getsize(filename)
            print(f"  File size: {file_size:,} bytes")
        else:
            print("  (no data was captured — was the button pressed?)")


if __name__ == "__main__":
    main()