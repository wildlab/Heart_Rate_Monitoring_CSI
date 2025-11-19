# log_csi.py
import pathlib, datetime, time, sys, re
import serial

DEV = "/dev/cu.SLAB_USBtoUART" # don't forget to adapt this to your use case if you're using this
BAUD = 921600
CSI_REC = re.compile(r'CSI_DATA,[^\r\n]*?,"\[.*?\]"', re.DOTALL)
outdir = pathlib.Path.home() / "data" / "csi_logs"
outdir.mkdir(parents=True, exist_ok=True)
ts_run = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
fp = outdir / f"csi_{ts_run}Z.csv"

print(f"\n Opening {DEV} @ {BAUD} …")
try:
    ser = serial.Serial(
        DEV,
        BAUD,
        timeout=0.2,          
        rtscts=False,
        dsrdtr=False,
        xonxoff=False,
        write_timeout=0
    )
except Exception as e:
    print(f"Could not open serial port: {e}")
    sys.exit(1)
try:
    ser.setDTR(False)
    ser.setRTS(False)
except Exception:
    pass

print("Serial open.")
print(f"Saving to: {fp}\n")

with ser, open(fp, "w", buffering=1) as f:
    f.write("timestamp_utc,raw\n")
    line_count = 0
    first_byte_deadline = time.time() + 3.0
    bbuf = bytearray()
    tbuf = ""  
    frag_fp = outdir / f"csi_{ts_run}Z.fragments.log"
    frag = open(frag_fp, "w", buffering=1)

    try:
        while True:
            n = ser.in_waiting
            chunk = ser.read(n if n else 1)
            if chunk:
                bbuf.extend(chunk)
                tbuf += chunk.decode("utf-8", errors="ignore")

                last_end = 0
                for m in CSI_REC.finditer(tbuf):
                    rec = m.group(0)              
                    last_end = m.end()
                    ts = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z")
                    raw = rec.replace('"','""')
                    f.write(f'{ts},"{raw}"\n')
                    print(f"[{ts}] {rec}")
                    line_count += 1
                    if line_count % 200 == 0:
                        print(f"--- {line_count} frames logged ---")
                if last_end:
                    head = tbuf[:last_end - len(m.group(0))] if 'm' in locals() else tbuf[:last_end]
                    if head.strip() and "CSI_DATA" not in head:
                        frag.write(head)
                    tbuf = tbuf[last_end:]
                if len(tbuf) > 2_000_000:
                    frag.write(f"\n--- truncating oversized partial buffer at {len(tbuf)} bytes ---\n")
                    tbuf = tbuf[-200_000:]

            else:
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n STOPPP.")
        print(f"Total frames logged: {line_count}")
        print(f"File saved at: {fp}")
        print(f"Fragments/non-CSI saved at: {frag_fp}")
    finally:
        frag.close()
