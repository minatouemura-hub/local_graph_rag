"""Record peak RSS for a command; useful for model-worker benchmarks."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=Path("logs/memory_benchmark.jsonl"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required after --")
    process = subprocess.Popen(args.command)
    child = psutil.Process(process.pid)
    peak = 0
    while process.poll() is None:
        try:
            peak = max(peak, child.memory_info().rss)
        except psutil.Error:
            break
        time.sleep(0.05)
    record = {"command": args.command, "returncode": process.returncode, "peak_rss_bytes": peak}
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
