import subprocess
print(subprocess.run('cat /tmp/pangram-splice-20261006/setup.log; tail -c 1500 /tmp/pangram-splice-20261006/setup.out; ls /proc/64792 >/dev/null 2>&1 && echo ALIVE || echo EXITED; date -u +%H:%M', shell=True, capture_output=True, text=True).stdout[-6000:])
