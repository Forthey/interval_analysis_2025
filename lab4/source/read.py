import struct
import numpy as np

def read_bin_file(path):
    adc_den = 16384.0

    with open(path, "rb") as f:
        header = f.read(256)
        frames = []
        point_dtype = np.dtype("<8H")
        side, mode, frame_count = struct.unpack("<BBH", header[:4])
        for _ in range(frame_count):
            frame_header_data = f.read(16)
            if len(frame_header_data) < 16:
                break
            stop_point, timestamp = struct.unpack("<HL", frame_header_data[:6])
            frame_data = np.frombuffer(f.read(1024 * 16), dtype=point_dtype)
            frames.append(frame_data)
    frames = np.array(frames)
    volts = frames / adc_den - 0.5
    return volts
