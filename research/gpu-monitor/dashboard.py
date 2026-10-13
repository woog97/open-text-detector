#!/usr/bin/env python3
"""Private localhost dashboard for recorded samples. Reach through SSH forwarding."""
import datetime as dt
import json
from pathlib import Path
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
ROOT=Path(__file__).resolve().parent

def history(start,end):
    records=[];latest=None;invalid=0
    for path in sorted((ROOT/'data').glob('samples-*.jsonl')):
        day=dt.datetime.strptime(path.stem[8:],'%Y-%m-%d').replace(tzinfo=dt.timezone.utc).timestamp()
        if day>end or day+86400<start:continue
        for line in path.open():
            try:r=json.loads(line)
            except ValueError:invalid+=1;continue
            if start<=r['timestamp']<=end:records.append(r)
    # Keep chart response bounded; retain full raw records on disk.
    records.sort(key=lambda r:r['timestamp'])
    total=len(records);step=max(1,(total+2999)//3000)
    reduced=records[::step]
    if records and (not reduced or reduced[-1]['timestamp']!=records[-1]['timestamp']):reduced.append(records[-1])
    last_index=0
    for row in reduced:
        while last_index<len(records) and records[last_index]['timestamp']<row['timestamp']:
            last_index+=1
        lo=max(1,last_index-step+1)
        row['gap_before']=any(records[i]['timestamp']-records[i-1]['timestamp']>90 for i in range(lo,last_index+1))
    return {'samples':reduced,'total_samples':total,'display_stride':step,'incomplete_lines':invalid,'server_time':time.time()}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url=urlparse(self.path)
        try:
            if url.path=='/api/history':
                q=parse_qs(url.query);end=float(q.get('end',[time.time()])[0]);start=float(q.get('start',[end-86400])[0])
                if end<start or end-start>32*86400:raise ValueError('Choose a range of at most 32 days.')
                body=json.dumps(history(start,end),separators=(',',':')).encode();kind='application/json'
            elif url.path in ['/','/index.html']:
                body=(ROOT/'dashboard.html').read_bytes();kind='text/html; charset=utf-8'
            else:self.send_error(404);return
        except (ValueError,OverflowError):self.send_error(400,'Invalid time range (maximum 32 days)');return
        self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass

if __name__=='__main__':
    import fcntl,os
    lock=open('/tmp/gpu-dashboard-'+str(os.getuid())+'.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    (ROOT/'dashboard-port.txt').write_text(str(server.server_address[1]))
    (ROOT/'dashboard.pid').write_text(str(os.getpid()))
    server.serve_forever()
