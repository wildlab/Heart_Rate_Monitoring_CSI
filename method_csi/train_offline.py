import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CSI = "/Users/ninabodenstab/esp/csi_hr/data/csi_logs/frames_with_ts_4_1.csv"  # timestamp_utc, v0,v1,...,v191 (already processed 192 features)
DEFAULT_HR = (
    "/Users/ninabodenstab/esp/csi_hr/data/hr_logs/hr_4_1.csv"  # timestamp + hr column
)
SAVE_HISTORY_NAME = "history_loss.npy"  # for the stats
WINDOW_SIZE = 100
MSE_THRESHOLD = 0.5
EPOCHS = 30
BATCH_SIZE = 32
VAL_SPLIT = 0.2
TOLERANCE = pd.Timedelta("1s")
CONTINUE_TRAINING_MODEL = ""  # csi_hr.keras

class StopCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get("val_loss", np.inf) <= MSE_THRESHOLD or os.path.isfile(
            "training.stop"
        ):
            print(f"\nReached {MSE_THRESHOLD} MSE; stopping training.")
            self.model.stop_training = True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csi", default=DEFAULT_CSI)
    ap.add_argument("--hr", default=DEFAULT_HR")
    ap.add_argument(
        "--tolerance_s",
        type=float,
        default=1.0,
    )
    args = ap.parse_args()
    csi = pd.read_csv(args.csi)
    if "timestamp_utc" not in csi.columns:
        raise RuntimeError("no timestamps utc column in the datafile :(")
    csi["timestamp_utc"] = pd.to_datetime( # cast
        csi["timestamp_utc"], utc=True, errors="coerce"
    )
    csi = (
        csi.dropna(subset=["timestamp_utc"])
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    feat_cols = [c for c in csi.columns if c.startswith("v")]
    if len(feat_cols) != 192:
        raise RuntimeError(
            f"Expected 192 CSI features v0..v191, found {len(feat_cols)}"
        )

    X_feat = csi[feat_cols].to_numpy(dtype=np.float32)
    X_times = csi["timestamp_utc"]
    hr = pd.read_csv(args.hr)
    hr["timestamp_utc"] = pd.to_datetime(hr["timestamp_utc"], utc=True, errors="coerce")
    hr["hr_bpm"] = pd.to_numeric(hr["hr_bpm"], errors="coerce")
    hr = (
        hr.dropna(subset=["timestamp_utc", "hr_bpm"])
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    tol = pd.Timedelta(seconds=args.tolerance_s)
    aligned = pd.merge_asof(
        csi[["timestamp_utc"]],
        hr,
        on="timestamp_utc",
        direction="nearest",
        tolerance=tol,
    )
    mask = aligned["hr_bpm"].notna().to_numpy()
    X_feat = X_feat[mask]
    X_times_kept = X_times[mask].reset_index(drop=True)
    y_per_frame = aligned.loc[mask, "hr_bpm"].to_numpy(dtype=np.float32)
    deltas = ((aligned.loc[mask, "timestamp_utc"] - X_times_kept).dt.total_seconds().abs())
    # training_data.txt  : one line per frame, 192 comma-separated floats
    # hr_data.txt        : one line per frame, single float bpm
    with open("training_data.txt", "w") as fX, open("hr_data.txt", "w") as fY:
        for row, bpm in zip(X_feat, y_per_frame):
            fX.write(",".join(map(lambda v: f"{float(v):.6f}", row)) + "\n")
            fY.write(f"{float(bpm):.6f}\n")
    print("Wrote training_data.txt and hr_data.txt (just like nick bild's implementation).")

    data = []
    data_hr = []
    with open("training_data.txt", "r") as f:
        for line in f:
            s = line.strip()
            if s:
                data.append(s.split(","))
    with open("hr_data.txt", "r") as f:
        for line in f:
            s = line.strip()
            if s:
                data_hr.append(float(s))

    data = np.array(data, dtype=np.float32)  # shape: (N,192)
    data_hr = np.array(data_hr, dtype=np.float32)  # shape: (N,)

    train_x = []
    train_y = []
    for i in range(len(data) - WINDOW_SIZE):
        train_x.append(data[i : i + WINDOW_SIZE])  # (100,192)
        avg_hr = float(np.mean(data_hr[i : i + WINDOW_SIZE]))
        train_y.append(avg_hr)
    train_x = np.asarray(train_x, dtype=np.float32)  # (M,100,192)
    train_y = np.asarray(train_y, dtype=np.float32)  # (M,)

    print(f"Training data X shape: {train_x.shape}")
    print(f"Training data Y shape: {train_y.shape}")

    if CONTINUE_TRAINING_MODEL != "":
        print("Loading saved model: {0}".format(CONTINUE_TRAINING_MODEL))
        model = keras.models.load_model(CONTINUE_TRAINING_MODEL)
    else:
        main_input = keras.Input(shape=(WINDOW_SIZE, 192), name="main_input")
        x = keras.layers.LSTM(64, return_sequences=True, name="lstm_1")(main_input)
        x = keras.layers.Dropout(0.2, name="dropout_1")(x)
        x = keras.layers.LSTM(32, name="lstm_2")(x)
        x = keras.layers.Dropout(0.2, name="dropout_2")(x)
        x = keras.layers.Dense(16, activation="relu", name="dense_1")(x)
        hr_output = keras.layers.Dense(1, name="hr_output")(x)

        model = keras.Model(inputs=main_input, outputs=hr_output)
        opt = keras.optimizers.legacy.Adam(learning_rate=1e-3)
        model.compile(optimizer=opt, loss={"hr_output": "mse"})
    model.summary()
    callbacks_list = [StopCallback()]
    history = model.fit(
        train_x,
        train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        validation_split=VAL_SPLIT,
        callbacks=callbacks_list,
    )
    if os.path.exists(SAVE_HISTORY_NAME):
        old_loss = np.load(SAVE_HISTORY_NAME).tolist()
        old_val_loss = np.load("val" + SAVE_HISTORY_NAME).tolist()

        combined_loss = old_loss + history.history["loss"]
        combined_val_loss = old_val_loss + history.history["val_loss"]
    else:
        combined_loss = history.history["loss"]
        combined_val_loss = history.history["val_loss"]

    np.save(SAVE_HISTORY_NAME, np.array(combined_loss))
    np.save("val" + SAVE_HISTORY_NAME, np.array(combined_val_loss))

    model.save("csi_hr.keras")
    print("Model training complete! Saved csi_hr.keras")
    loss = np.load(SAVE_HISTORY_NAME)
    val_loss = np.load("val" + SAVE_HISTORY_NAME)
    plt.figure(figsize=(10, 6))
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training + Validation Loss Across Both Phases")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
