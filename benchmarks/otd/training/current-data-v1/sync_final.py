from pathlib import Path
import base64,sys
sys.path.insert(0,__import__('os').path.expanduser('~/.config/open-text-detector'))
from remote import run
r=Path(__file__).resolve().parent
payload={n:base64.b64encode((r/n).read_bytes()).decode() for n in ['launch.py','test_encoding.py','PROTOCOL.txt']}
run('PAYLOAD='+repr(payload)+'\n'+'''
from pathlib import Path
import base64
r=Path('/data/workspace/current-data-v1');assert not (r/'READY.json').exists(),'Do not alter frozen setup'
for n,v in PAYLOAD.items():(r/n).write_bytes(base64.b64decode(v))
print('Setup documentation, tests and explicit launcher copied; launcher not executed')
''')
