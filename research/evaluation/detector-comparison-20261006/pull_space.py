import json, glob
S = '/tmp/pangram-splice-20261006/sweeps'; out = {}
for f in glob.glob(S + '/q4b-*/eval/*.json') + glob.glob(S + '/q4b-*/status.json'):
    if f.endswith('-rows.json'): continue
    out[f[len(S) + 1:]] = json.load(open(f))
print('JSONSTART' + json.dumps(out) + 'JSONEND')
