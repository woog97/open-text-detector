import subprocess, sys
def sh(c): print('$', c); r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=120); print((r.stdout + r.stderr)[-2500:])
sh('ls /tmp/pangram-space-fast10/vendor /tmp/pangram-space-fast10/assets /tmp/pangram-space-fast10/assets/* | head -40; tail -5 /tmp/pangram-space-fast10/install.log; ls /tmp | head -40')
sh('ls -d /tmp/pangram-wandb-vendor 2>&1; python -c "import sys;print(sys.version)"')
sh("cd /tmp && PYTHONPATH=/tmp/pangram-space-fast10/vendor:/tmp/pangram-wandb-vendor python -c 'import transformers,peft,bitsandbytes,torch;print(transformers.__version__,peft.__version__,bitsandbytes.__version__,torch.__version__)'")
sh("cd /tmp && PYTHONPATH=/tmp/pangram-space-fast10/vendor:/tmp/pangram-wandb-vendor python -c 'import wandb;print(wandb.__version__, wandb.__file__)'")
sh("python -c 'import transformers,torch;print(transformers.__version__,torch.__version__)'; pip list 2>/dev/null | grep -i -E 'peft|bitsandbytes|wandb|transformers|safetensors|torch ' ")
