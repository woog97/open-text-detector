"""Resume the two existing chunks in order, preserving their frozen manifests."""
import subprocess,sys
for chunk in ['000002','000003']:
 subprocess.run([sys.executable,'-u','scripts/run_iclr2027_pipeline.py','--chunk',chunk],check=True)
