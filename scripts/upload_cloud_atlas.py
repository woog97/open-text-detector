"""Resumable upload using the user's existing Wrangler authorization; never prints tokens."""
import concurrent.futures, hashlib, json, time, tomllib, urllib.request, urllib.parse
from pathlib import Path
root=Path(__file__).resolve().parents[1];folder=root/'app/.sites-runtime/atlas'
manifest=json.loads((folder/'manifest.json').read_text())
record=folder/'uploaded.json';done=json.loads(record.read_text()) if record.exists() else {}
config=Path.home()/'Library/Preferences/.wrangler/config/default.toml'
base='https://api.cloudflare.com/client/v4/accounts/6147916993a1cbfa4e3898aaeed27ae3/r2/buckets/pangram-paper-atlas/objects/'
def upload(item):
    data=Path(item['path']).read_bytes();digest=hashlib.sha256(data).hexdigest()
    if done.get(item['key'])==digest:return item['key'],digest
    for attempt in range(4):
        try:
            token=tomllib.loads(config.read_text())['oauth_token']
            req=urllib.request.Request(base+urllib.parse.quote(item['key'],safe='/'),data=data,method='PUT',headers={'Authorization':'Bearer '+token,'Content-Type':item['type'],'cf-r2-storage-class':'Standard'})
            with urllib.request.urlopen(req,timeout=180) as res:res.read()
            return item['key'],digest
        except Exception as e:
            if attempt==3:raise RuntimeError(f"Upload failed for {item['key']}: {type(e).__name__}") from None
            time.sleep(2**attempt)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    futures=[pool.submit(upload,i) for i in manifest]
    for index,f in enumerate(concurrent.futures.as_completed(futures),1):
        key,digest=f.result();done[key]=digest
        record.write_text(json.dumps(done))
        if index%50==0 or index==len(manifest):print(f'{index}/{len(manifest)} objects verified/uploaded',flush=True)
