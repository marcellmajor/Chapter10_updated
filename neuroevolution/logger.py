"""Minimal replacement for the original tabular_logger: prints a key/value table per
iteration and appends it to ``progress.csv`` in the log directory."""

import csv
import os
import sys
import time

_log_dir = None
_row = {}
_csv_keys = None


def set_log_dir(path):
    global _log_dir, _csv_keys
    _log_dir = path
    _csv_keys = None
    os.makedirs(path, exist_ok=True)


def log_dir():
    return _log_dir


def info(*args):
    msg = " ".join(str(a) for a in args)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_dir:
        with open(os.path.join(_log_dir, "log.txt"), "a") as f:
            f.write(line + "\n")


def record_tabular(key, value):
    _row[key] = value


def dump_tabular():
    global _csv_keys
    if not _row:
        return
    width = max(len(k) for k in _row)
    lines = ["-" * (width + 20)]
    for key, value in _row.items():
        text = f"{value:.6g}" if isinstance(value, float) else str(value)
        if len(text) > 60:
            text = text[:57] + "..."
        lines.append(f"| {key:<{width}} | {text}")
    lines.append("-" * (width + 20))
    print("\n".join(lines), file=sys.stdout, flush=True)

    if _log_dir:
        path = os.path.join(_log_dir, "progress.csv")
        if _csv_keys is None:
            _csv_keys = list(_row.keys())
            new_file = not os.path.exists(path)
        else:
            new_file = False
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_csv_keys, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow({k: _row.get(k) for k in _csv_keys})
    _row.clear()
