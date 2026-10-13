import os, subprocess, json, time
def sh(c): print('$', c); print(subprocess.run(c, shell=True, capture_output=True, text=True, timeout=120).stdout[-3000:])
for d in ['/data/workspace/backbone-launch-20261003', '/tmp/pangram-space-fast10']:
    sh(f'ls -la {d} | head -60')
sh('ls /data/workspace/backbone-launch-20261003/sweeps | head -80; ls /data/workspace/backbone-launch-20261003/sweeps | wc -l')
sh('ls -la /data/workspace/backbone-launch-20261003/runs/qwen35-4b/ /data/workspace/backbone-launch-20261003/runs/qwen35-4b/prepared-v2 2>&1 | head -30')
sh('ls -la /data/workspace/backbone-launch-20261003/assets/ 2>&1 | head; ls /data/workspace/backbone-launch-20261003/assets/qwen35-4b 2>&1 | head -30')
sh('nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader')
sh('df -h /tmp /data | head; free -g | head -3')
print(time.strftime('%H:%M UTC', time.gmtime()))
