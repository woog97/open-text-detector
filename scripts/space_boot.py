"""Print the training Space's boot time (UTC epoch seconds, rounded to the minute), /tmp contents and GPU use. Run via remote.py."""
import json, os, subprocess, time

up = float(open('/proc/uptime').read().split()[0])
gpus = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()
print('BOOT ' + json.dumps({'boot_epoch_min': int((time.time() - up) // 60 * 60), 'uptime_s': int(up),
                            'tmp': sorted(os.listdir('/tmp'))[:20], 'gpus_busy': sum(int(x) > 1000 for x in gpus)}))
