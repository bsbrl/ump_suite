"""
Stream HEKA monitor samples to the Linux machine over UDP.

Wiring:
    Voltage monitor BNC center/+ -> NI AI0
    Voltage monitor BNC shield/- -> NI AI GND
    Current monitor BNC center/+ -> NI AI1
    Current monitor BNC shield/- -> NI AI GND

Packet format:
    header = struct.Struct("<5sdfH")
        magic: b"HEKA1"
        first_sample_time: float64
        sample_rate_hz: float32
        sample_count: uint16

    payload:
        repeated float32 pairs [voltage_raw_v, current_pA]

Linux unpack sketch:
    import struct
    import numpy as np

    HEADER = struct.Struct("<5sdfH")

    data, addr = sock.recvfrom(2048)
    magic, t0, rate, n = HEADER.unpack_from(data, 0)
    samples = np.frombuffer(data, dtype="<f4", offset=HEADER.size).reshape(n, 2)

    voltage_raw_v = samples[:, 0]
    current_pA = samples[:, 1]
"""

# windows_send_heka_data.py
import socket
import struct
import time

import nidaqmx
import numpy as np
from nidaqmx.constants import AcquisitionType, TerminalConfiguration


UBUNTU_IP = "192.168.50.2"
PORT = 5005

DEVICE = "Dev1"
VOLTAGE_CHANNEL = f"{DEVICE}/ai0"
CURRENT_CHANNEL = f"{DEVICE}/ai1"

SAMPLE_RATE = 10000
READ_SECONDS = 0.01
SAMPLES_PER_READ = int(SAMPLE_RATE * READ_SECONDS)

# Same conversion used by windows_plot_heka_monitors.py.
CURRENT_GAIN_MV_PER_PA = 20.0
VOLTAGE_MONITOR_SCALE = 1.0

INPUT_TERMINAL_CONFIG = TerminalConfiguration.RSE

# Binary UDP packet:
#   header:  magic, first_sample_time, sample_rate_hz, sample_count
#   payload: repeated float32 pairs: voltage_raw_v, current_pA
PACKET_MAGIC = b"HEKA1"
PACKET_HEADER = struct.Struct("<5sdfH")


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(
        VOLTAGE_CHANNEL,
        terminal_config=INPUT_TERMINAL_CONFIG,
        min_val=-10.0,
        max_val=10.0,
    )
    task.ai_channels.add_ai_voltage_chan(
        CURRENT_CHANNEL,
        terminal_config=INPUT_TERMINAL_CONFIG,
        min_val=-10.0,
        max_val=10.0,
    )

    task.timing.cfg_samp_clk_timing(
        rate=SAMPLE_RATE,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=SAMPLES_PER_READ,
    )

    print(
        f"Streaming voltage={VOLTAGE_CHANNEL}, current={CURRENT_CHANNEL} "
        f"at {SAMPLE_RATE} Hz"
    )
    print(f"Sending raw monitor packets to {UBUNTU_IP}:{PORT}")
    print(f"Samples per UDP packet: {SAMPLES_PER_READ}")
    print(f"NI terminal configuration = {INPUT_TERMINAL_CONFIG.name}")
    print("Wire voltage monitor BNC center/+ to AI0 and shield/- to AI GND")
    print("Wire current monitor BNC center/+ to AI1 and shield/- to AI GND")
    print("Packet payload columns: voltage_raw_v, current_pA")

    while True:
        data = task.read(
            number_of_samples_per_channel=SAMPLES_PER_READ,
            timeout=2.0,
        )
        read_time = time.time()

        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise RuntimeError(f"Expected 2 DAQ channels, got shape {arr.shape}")

        voltage_raw_v = arr[0] * VOLTAGE_MONITOR_SCALE
        current_pA = arr[1] * 1000.0 / CURRENT_GAIN_MV_PER_PA

        samples = np.column_stack((voltage_raw_v, current_pA)).astype("<f4")
        first_sample_time = read_time - (samples.shape[0] - 1) / SAMPLE_RATE

        packet = PACKET_HEADER.pack(
            PACKET_MAGIC,
            first_sample_time,
            float(SAMPLE_RATE),
            samples.shape[0],
        ) + samples.tobytes()

        sock.sendto(packet, (UBUNTU_IP, PORT))

        print(
            f"{read_time:.6f},"
            f"latest_voltage_raw_v={float(voltage_raw_v[-1]):.9f},"
            f"latest_current_pA={float(current_pA[-1]):.6f},"
            f"v_raw_ptp={float(np.ptp(voltage_raw_v)):.6f},"
            f"i_pA_ptp={float(np.ptp(current_pA)):.6f},"
            f"samples={samples.shape[0]}"
        )
