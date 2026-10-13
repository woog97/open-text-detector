"""Build page.html (the Final Runs results page) from page.template.html and data.json (run collect.py first)."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
data = json.loads((H / 'data.json').read_text())
t = (H / 'page.template.html').read_text()
assert '/*DATA*/null' in t
(H / 'page.html').write_text(t.replace('/*DATA*/null', json.dumps(data, separators=(',', ':'))))
print('page.html', (H / 'page.html').stat().st_size)
