import pandas as pd
import numpy as np

def select_columns(df):
    columns = ["id","utcTimestamp","utcDate","accInGBurst","imuFreqHz","axes"]
    df = df[columns].copy()
    df["utcDate"] = pd.to_datetime(df["utcDate"],format = "%d.%m.%y %H:%M:%S",utc=True) # format: 09.04.24 12:10:00
    return df

def parse_burst(acc,init_time,fs):
    """
    Parse one IMU burst string into a DataFrame with columns:
    ['X', 'Y', 'Z', 'time'].

    acc_str: string of space-separated XYZ values
    init_time: timestamp of the burst (scalar)
    fs: sampling frequency in Hz (default 50 Hz)
    """
    acc_list = acc.split(" ")
    acc_list = [float(i) for i in acc_list]
    if len(acc_list) % 3 != 0:
        raise ValueError(f"Burst length {len(vals)} not divisible by 3 (XYZ)")
    acc_list = np.array(acc_list).reshape((len(acc_list)//3, 3))
    df_acc = pd.DataFrame(acc_list, columns=["X", "Y", "Z"])
    dt = 1.0/fs
    n = df_acc.shape[0]
    df_acc["time"] = init_time + pd.to_timedelta(np.arange(n) * dt, unit="s")
    return df_acc

def concat_bursts(df,fs):
    """
    Concatenate all bursts in the input DataFrame into a single
    long-form DataFrame with columns ['X', 'Y', 'Z', 'time', 'burst_idx'].
    """
    for i, row in df.iterrows():
        burst = row["accInGBurst"]
        init_time = row["utcDate"]
        df_burst = parse_burst(burst,init_time,fs) # (X,Y,Z,time)
        df_burst["burst_idx"] = i
        if i == 0:
            df_all = df_burst
        else:
            df_all = pd.concat([df_all, df_burst], ignore_index=True)
    return df_all

def long_bursts(df,thr=200):
    """
    Some bursts are longer than others, these are more useful for peak detection. Preserve only these.
    """
    burst_lengths = []
    for i, row in df.iterrows():
        burst_lengths.append(len(row["accInGBurst"].split(" ")))
    m = [i > thr for i in burst_lengths]
    long_bursts = df[m].copy().reset_index(drop=True)
    return long_bursts

    

