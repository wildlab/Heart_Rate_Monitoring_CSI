# print_garmin_hr_log.py
import asyncio, datetime, pathlib
from bleak import BleakScanner, BleakClient
HRS_UUID="0000180d-0000-1000-8000-00805f9b34fb"; HRM_CHAR="00002a37-0000-1000-8000-00805f9b34fb" # Bluetooth GATT Heart Rate Service

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00","Z")
def parse_hr(data: bytes) -> int: return int.from_bytes(data[1:3],"little") if (data[0]&1) else data[1]

async def find_forerunner(timeout=6.0):
    for d in await BleakScanner.discover(timeout=timeout):
        n = (d.name or "")
        if "Forerunner" in n or "GARMIN" in n: return d
    return None

async def main():
    outdir = pathlib.Path.home()/ "data" / "hr_logs"; outdir.mkdir(parents=True, exist_ok=True)
    fp = (outdir / f"hr_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}Z.csv").open("w", buffering=1)
    fp.write("timestamp_utc,hr_bpm\n")
    dev = await find_forerunner()
    if not dev: print("No Forerunner found."); return
    async with BleakClient(dev) as c:
        async def cb(_, data: bytes):
            try: 
                hr = parse_hr(data)
                ts = utc()
                fp.write(f"{utc()},{parse_hr(data)}\n")
                print(ts, hr)
            except Exception: pass
        await c.start_notify(HRM_CHAR, cb)
        print("HR logging… Ctrl+C to stop."); 
        try:
            while True: await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await c.stop_notify(HRM_CHAR); fp.close()

if __name__=="__main__": asyncio.run(main())
