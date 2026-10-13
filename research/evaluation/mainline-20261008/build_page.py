"""Build page.html from page.template.html + data.json (collect.py) + facts.json (written after reading the results)."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
data = json.loads((H / 'data.json').read_text())
if (H / 'facts.json').exists(): data['facts'] = json.loads((H / 'facts.json').read_text())
t = (H / 'page.template.html').read_text(); assert '/*DATA*/null' in t
(H / 'page.html').write_text(t.replace('/*DATA*/null', json.dumps(data, separators=(',', ':'))))
print('page.html', (H / 'page.html').stat().st_size)
