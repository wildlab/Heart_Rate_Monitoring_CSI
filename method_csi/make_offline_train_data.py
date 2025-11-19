import sys
import csv
import json
import argparse
import numpy as np
import ast
from scipy.signal import butter, filtfilt, savgol_filter
import re
import pandas as pd
# This code basically replicates the code written by Nick Bild: https://github.com/nickbild/csi_hr?tab=readme-ov-file
WINDOW = 100
buffer = []
COLLECT_TRAINING_DATA = False

CSI_VAID_SUBCARRIER_INTERVAL = 1
csi_vaid_subcarrier_len = 0

CSI_DATA_INDEX = 200
CSI_DATA_COLUMNS = 490

DATA_COLUMNS_NAMES_C5C6 = [
    "type",
    "id",
    "mac",
    "rssi",
    "rate",
    "noise_floor",
    "fft_gain",
    "agc_gain",
    "channel",
    "local_timestamp",
    "sig_len",
    "rx_state",
    "len",
    "first_word",
    "data",
]
DATA_COLUMNS_NAMES = [
    "type",
    "id",
    "mac",
    "rssi",
    "rate",
    "sig_mode",
    "mcs",
    "bandwidth",
    "smoothing",
    "not_sounding",
    "aggregation",
    "stbc",
    "fec_coding",
    "sgi",
    "noise_floor",
    "ampdu_cnt",
    "channel",
    "secondary_channel",
    "local_timestamp",
    "ant",
    "sig_len",
    "rx_state",
    "len",
    "first_word",
    "data",
]

CSI_REC = re.compile(r'CSI_DATA,[^\r\n]*?,"\[.*?\]"')


def parse_csi_amplitudes(csi_str: str):
    """csi_str is a stringified list of ints [imag0,real0, imag1,real1, ...]"""
    values = ast.literal_eval(csi_str)
    complex_pairs = np.array(values).reshape(-1, 2)
    csi_complex = complex_pairs[:, 1] + 1j * complex_pairs[:, 0]
    amplitudes = np.abs(csi_complex)
    return amplitudes


def remove_dc(signal, fs, lowcut=2.0, highcut=5.0, order=3):
    """WiHear-style 2-5 Hz bandpass (optional)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, signal)


def butter_bandpass_filter(
    signal: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 3
) -> np.ndarray:
    x = np.asarray(signal, dtype=float).copy()
    if x.size == 0:
        return x
    nyq = 0.5 * fs
    if not (0 < lowcut < highcut < nyq):
        raise ValueError(
            f"Invalid bandpass: low={lowcut}, high={highcut}, Nyquist={nyq} with fs={fs}"
        )
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)


def savitzky_golay_smooth(
    signal: np.ndarray, window_length: int = 15, polyorder: int = 3
) -> np.ndarray:
    x = np.asarray(signal, dtype=float).copy()
    n = x.size
    if n == 0:
        return x
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    if wl < (polyorder + 2):
        return x
    if wl >= n:
        wl_candidate = n - 1
        if wl_candidate % 2 == 0:
            wl_candidate -= 1
        if wl_candidate < (polyorder + 2):
            return x
        wl = wl_candidate
    return savgol_filter(x, wl, polyorder)


def process_log_file_with_timestamps(
    log_csv_path: str,
    out_csv_path: str = "./data/csi_logs/frames_with_ts_4_1.csv",
    fs: float = 20.0,
    hr_band=(0.8, 2.17),
    apply_remove_dc: bool = True,
):
    rows_out = []
    with open(log_csv_path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            raw = row["raw"]
            ts = row["timestamp_utc"]
            if not raw:
                continue
            records = CSI_REC.findall(raw) or []
            for rec in records:
                try:
                    parts = next(csv.reader([rec]))
                    csi_len = int(parts[-3])
                    try:
                        csi_raw = json.loads(parts[-1])
                    except json.JSONDecodeError:
                        csi_raw = ast.literal_eval(parts[-1])
                except:
                    continue
                if not isinstance(csi_raw, list) or len(csi_raw) != csi_len:
                    continue
                amps = parse_csi_amplitudes(parts[-1])
                x = amps
                if apply_remove_dc:
                    x = remove_dc(x, fs)
                x = butter_bandpass_filter(x, hr_band[0], hr_band[1], fs, order=3)
                shaped = savitzky_golay_smooth(x, 15, 3)
                rows_out.append([ts] + shaped.tolist())
    cols = ["timestamp_utc"] + [f"v{i}" for i in range(len(rows_out[0]) - 1)]
    df = pd.DataFrame(rows_out, columns=cols)
    df.to_csv(out_csv_path, index=False)
    print(f"Wrote {len(df)} frames to {out_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline CSI → processed vectors")
    parser.add_argument(
        "--from-log",
        dest="from_log",
        help="Path to csi_*.csv log created by log_csi.py",
    )

    args = parser.parse_args()

    if args.from_log:
        process_log_file_with_timestamps(
            log_csv_path=args.from_log,
            # out_csv_path="./data/csi_logs/frames_with_ts_2_2.csv",
            fs=20.0,
            hr_band=(0.8, 2.17),  # just like in pulsefi
            apply_remove_dc=True,
        )
        print("Done.")
        sys.exit(0)

    print("No --from-log provided! pls provide :P --> exiting now")
