import csv
import json
import argparse
import numpy as np
from io import StringIO
import ast
from scipy.signal import butter, filtfilt, savgol_filter
import re
from tensorflow import keras

model = keras.models.load_model("csi_hr.keras", safe_mode=False)
stats = np.load("train_stats.npz")
mean = stats["mean"]  # shape (1,1,192)
std = stats["std"]  # shape (1,1,192)
WINDOW = 100
buffer = []
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
csi_data_complex = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)
agc_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)
fft_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)
CSI_REC = re.compile(r'CSI_DATA,[^\r\n]*?,"\[.*?\]"')


def parse_csi_amplitudes(csi_str: str):
    values = ast.literal_eval(csi_str)
    complex_pairs = np.array(values).reshape(-1, 2)
    csi_complex = complex_pairs[:, 1] + 1j * complex_pairs[:, 0]
    return np.abs(csi_complex)


def remove_dc(signal, fs, lowcut=2.0, highcut=5.0, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass")
    return filtfilt(b, a, signal)


def butter_bandpass_filter(signal, lowcut, highcut, fs, order=3):
    x = np.asarray(signal, dtype=float)
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, x)


def savitzky_golay_smooth(signal, window_length=15, polyorder=3):
    x = np.asarray(signal, dtype=float)
    n = len(x)
    if n <= window_length:
        return x
    return savgol_filter(x, window_length, polyorder)


def csi_data_read_parse(port: str, csv_writer, log_file_fd):
    import serial

    ser = serial.Serial(port=port, baudrate=921600, bytesize=8, parity="N", stopbits=1)
    if not ser.isOpen():
        print("open failed")
        return
    print("open success")
    while True:
        s = str(ser.readline()).lstrip("b'").rstrip("\\r\\n'")
        if "CSI_DATA" not in s:
            log_file_fd.write(s + "\n")
            log_file_fd.flush()
            continue
        csi_data = next(csv.reader(StringIO(s)))
        if len(csi_data) not in (len(DATA_COLUMNS_NAMES), len(DATA_COLUMNS_NAMES_C5C6)):
            continue
        try:
            csi_data_len = int(csi_data[-3])
            try:
                csi_raw_data = json.loads(csi_data[-1])
            except json.JSONDecodeError:
                csi_raw_data = ast.literal_eval(csi_data[-1])
        except:
            continue
        if len(csi_raw_data) != csi_data_len:
            continue
        csv_writer.writerow(csi_data)
        amp = parse_csi_amplitudes(csi_data[-1])
        x = remove_dc(amp, fs=20.0)
        x = butter_bandpass_filter(x, 0.8, 2.17, 20.0, order=3)
        shaped = savitzky_golay_smooth(x, 15, 3)
        buffer.append(shaped)
        if len(buffer) > WINDOW:
            buffer.pop(0)
        if len(buffer) == WINDOW:
            X = np.array(buffer, dtype=np.float32).reshape(1, WINDOW, 192)
            X = (X - mean) / std
            hr = model.predict(X, verbose=0)[0, 0]
            print(f"HR: {hr:.1f} bpm")
        else:
            print(f"Collecting CSI… {len(buffer)}/{WINDOW}")

    ser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", required=True)
    parser.add_argument("-s", "--store", default="./csi_data.csv")
    parser.add_argument("-l", "--log", default="./csi_log.txt")
    args = parser.parse_args()

    csi_data_read_parse(
        args.port, csv.writer(open(args.store, "w")), open(args.log, "w")
    )
