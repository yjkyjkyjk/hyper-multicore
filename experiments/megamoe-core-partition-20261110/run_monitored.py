#!/usr/bin/env python3
"""Record device ownership while running an isolated scheduling measurement."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

repo, python, devices, mode, phase, output, port = sys.argv[1:]
root = Path(__file__).resolve().parent
output = Path(output)
output.mkdir(parents=True, exist_ok=True)
selected = {int(value) for value in devices.split(',')}
subprocess.run([python, str(root/'check-free-devices.py'), devices],
               stdout=(output/'devices-before.log').open('w'), check=True)
samples = []
with (output/'launcher.log').open('w') as log, (output/'device-samples.jsonl').open('w') as samples_file:
    process = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=15s', '600s',
                                'bash', str(root/'run_core_partition.sh'), repo, python,
                                devices, mode, phase, str(output), port], stdout=log, stderr=subprocess.STDOUT)
    while process.poll() is None:
        before = time.time_ns()
        probe = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=15)
        pids = []
        for line in probe.stdout.splitlines():
            match = re.match(r'\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|', line)
            if match and int(match[1]) in selected:
                device, pid = int(match[1]), int(match[2])
                try:
                    os.kill(pid, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                except PermissionError:
                    alive = True
                pids.append(dict(device=device, pid=pid, alive=alive))
        sample = dict(start_ns=before, end_ns=time.time_ns(), returncode=probe.returncode, pids=pids)
        samples.append(sample)
        samples_file.write(json.dumps(sample)+'\n')
        samples_file.flush()
        time.sleep(0.4)
status = process.wait()
if status:
    raise SystemExit(status)
result = json.loads((output/'result.json').read_text())
own = {rank['provenance']['pid'] for rank in result['ranks']}
foreign = []
for sample in samples:
    foreign.extend(dict(start_ns=sample['start_ns'],end_ns=sample['end_ns'],**entry)
                   for entry in sample['pids'] if entry['alive'] and entry['pid'] not in own)
audit = dict(passed=not foreign and all(s['returncode']==0 for s in samples),
             own_pids=sorted(own), foreign=foreign, samples=len(samples),
             interval_seconds=0.4, scope='entire process, including warmup and timed samples')
(output/'device-ownership.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(dict(mode=mode, phase=phase, output=str(output), **audit)),flush=True)
if not audit['passed']:
    raise SystemExit(3)
