import subprocess
print(subprocess.run('cat /tmp/pangram-splice-20261006/setup-wave2.log; tail -c 800 /tmp/pangram-splice-20261006/setup-wave2.out; ls /proc/77360 >/dev/null 2>&1 && echo ALIVE || echo EXITED', shell=True, capture_output=True, text=True).stdout[-8000:])
