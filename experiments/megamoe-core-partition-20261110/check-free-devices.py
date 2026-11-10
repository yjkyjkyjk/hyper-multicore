#!/usr/bin/env python3
"""Reject occupied devices immediately before launching this experiment."""
import re
import subprocess
import sys
import time

devices = {int(value) for value in sys.argv[1].split(',')}
while True:
    output = subprocess.check_output(['npu-smi', 'info'], text=True)
    busy = []
    for line in output.splitlines():
        match = re.match(r'\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|', line)
        if match and int(match[1]) in devices:
            busy.append((int(match[1]), int(match[2])))
    if not busy:
        print(output, flush=True)
        break
    print(f'Waiting for devices {busy}; no experiment launched.', flush=True)
    time.sleep(10)
