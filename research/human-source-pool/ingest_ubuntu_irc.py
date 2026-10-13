"""Pinned Ubuntu IRC: unchanged human conversation spans, automatic bot filters."""
import argparse
from bisect import bisect_left
from collections import Counter
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import time

from collect_pool import BINS, LENGTH_WEIGHTS, OPEN_LICENSE, STOPWORDS, apportion, atomic_json, date_year, digest, now
from expand_pool import add, connect

SID = 'ubuntu_irc'
REPO = 'common-pile/ubuntu_irc'
REVISION = '35a9ef530c682965b53d4167e37b32f6a85aa87c'
VERSION = 'ubuntu-irc-human-conversation-v1'
TURN = re.compile(r'^\[(\d{2}):(\d{2})\]\s+<([^>\s]+)>\s*(.*)$')
BOTS = re.compile(r'^(?:ubot\w*|ubuntulog\w*|floodbot\w*|meetingology\w*|chanserv|nickserv|mootbot\w*|locobot\w*|logbot\w*|.*bot\d*)$',re.I)
CHANNELS = set('#ubuntu #ubuntu-offtopic #ubuntu-devel #ubuntu-server #ubuntu-desktop #ubuntu-kernel #ubuntu-bugs #ubuntu-motu #ubuntu-app-devel #ubuntu-doc #ubuntu+1 #ubuntu-uk #ubuntu-ca #ubuntu-au #ubuntu-nz #kubuntu #kubuntu-devel #xubuntu #xubuntu-devel #lubuntu #edubuntu #ubuntu-touch #ubuntu-quality #ubuntu-meeting #ubuntu-release'.split())
ENGLISH = STOPWORDS | set('you your we our do does did how can could would should will if why when what where which has had have yes no need want using use help please thanks'.split())


def channel_ok(channel):
    return channel in CHANNELS or bool(re.fullmatch(r'#ubuntu-us(?:-[a-z]{2})?',channel))


def parsed_runs(text):
    """Consecutive genuine-looking turns only; excluded lines always split runs."""
    parsed = []
    for match in re.finditer(r'[^\n]*(?:\n|$)',text):
        if not match.group(): continue
        value = match.group().rstrip('\r\n'); turn = TURN.fullmatch(value)
        item = None
        if turn:
            hh,mm,nick,body = turn.groups(); words = re.findall(r'[A-Za-z]+',body)
            if (int(hh) < 24 and int(mm) < 60 and not BOTS.fullmatch(nick)
                and words and not body.lstrip().startswith(('!','$ ','>>>','sudo ','apt-get '))
                and sum(c.isalpha() for c in body) / max(1,len(body)) >= .4
                and len(re.findall(r'https?://',body)) < 3):
                item = (match.start(),match.start()+len(value),body,nick,int(hh)*60+int(mm))
        parsed.append(item)
    repeated = Counter(' '.join(p[2].casefold().split()) for p in parsed if p and len(p[2].split())>=8)
    runs=[];run=[]
    for item in parsed:
        if item and repeated[' '.join(item[2].casefold().split())] > 3: item=None
        if item is None or (run and (item[4] < run[-1][4] or item[4] - run[-1][4] > 20)):
            if run:runs.append(run)
            run=[]
        if item:run.append(item)
    if run:runs.append(run)
    return runs


def english_prose(bodies):
    value=' '.join(bodies); words=re.findall(r'[A-Za-z]+',value)
    letters=sum(c.isalpha() for c in value)
    return (len(words)>=30 and sum(w.lower() in ENGLISH for w in words)/len(words)>=.1
            and sum(c.isalpha() and ord(c)>127 for c in value)/max(1,letters)<.05)


def select_span(text,doc_id,remaining):
    runs=parsed_runs(text); rng=random.Random(digest('27183'+doc_id)); rng.shuffle(runs)
    bins=[i for i in range(4) if remaining[i]>0]
    order=[]
    while bins:
        i=rng.choices(bins,weights=[remaining[j] for j in bins])[0];order.append(i);bins.remove(i)
    for bin_id in order:
        lo,hi=BINS[bin_id]; target=(lo+hi)//2
        for run in runs:
            prefix=[0]
            for a,b,body,nick,minute in run:prefix.append(prefix[-1]+len(text[a:b].split()))
            if prefix[-1]<lo:continue
            starts=list(range(len(run)));rng.shuffle(starts)
            for start in starts[:128]:
                end=bisect_left(prefix,prefix[start]+target,lo=start+1)
                end=min(end,len(run))
                if prefix[end]-prefix[start]>hi:end-=1
                if end<=start or not lo<=prefix[end]-prefix[start]<=hi:continue
                turns=run[start:end]
                if not english_prose([t[2] for t in turns]):continue
                a,b=turns[0][0],turns[-1][1];wc=len(text[a:b].split())
                if lo<=wc<=hi:return a,b,wc,bin_id,sorted({t[3] for t in turns})
    return None


def make_pair(row,filename,index,remaining,archive_hash):
    meta=row.get('metadata') or {};channel=str(meta.get('channel') or '')
    text=row.get('text');date=str(row.get('created') or '');year=date_year(date)
    if not isinstance(text,str) or not channel_ok(channel) or not year or year>2021:return None
    if not OPEN_LICENSE.search(str(meta.get('license') or '')):return None
    doc_id=REPO+':'+str(row.get('id') or '')
    if not row.get('id'):return None
    selected=select_span(text,doc_id,remaining)
    if selected is None:return None
    a,b,wc,bin_id,speakers=selected;passage=text[a:b];retrieved=now();raw_hash=digest(text)
    original={'source_id':SID,'source_dataset':REPO,'source_revision':REVISION,'source_file':filename,
              'source_row':index,'retrieved_at':retrieved,'raw_text_sha256':raw_hash,
              'archive_sha256':archive_hash,'record':row}
    record={'record_id':digest(SID+doc_id+str(a)+str(b)),'source_id':SID,'category':'social','text':passage,
            'word_count':wc,'length_bin':bin_id,'source_dataset':REPO,'source_revision':REVISION,
            'source_file':filename,'source_row':index,'original_id':str(row['id']),
            'source_url':str(meta.get('url') or ''),'title':channel+' '+date,
            'author_attribution_json':json.dumps(speakers),'license_evidence':str(meta['license']),
            'claimed_original_date':date,'retrieved_at':retrieved,'raw_text_sha256':raw_hash,
            'passage_sha256':digest(passage),'raw_start':a,'raw_end':b,'offset_unit':'unicode_codepoints',
            'extraction_method':'unchanged_contiguous_human_irc_turns','parent_document_id':doc_id,
            'provisional_family_id':digest(channel+':'+date),'channel':channel,'channel_day':channel+':'+date,
            'speaker_family_ids':[digest(channel+':'+nick.casefold()) for nick in speakers],
            'admission_status':'quarantined_candidate','training_eligible':False,
            'provenance_basis':'dated_archived_irc_channel_log_automated_human_turn_filter',
            'protected_overlap_status':'not_fully_audited',
            'reason_codes':['unknown_bot_and_human_authorship_audit_pending','conversation_privacy_audit_pending',
                            'protected_overlap_audit_pending','genre_and_extraction_review_pending'],
            'sampling_seed':27183,'pipeline_version':VERSION}
    return record,original


def download(folder,filename):
    import requests
    path=folder/Path(filename).name;manifest=path.with_suffix(path.suffix+'.manifest.json')
    if not path.exists():
        url='https://huggingface.co/datasets/'+REPO+'/resolve/'+REVISION+'/'+filename
        with requests.get(url,stream=True,timeout=(20,180)) as response:
            response.raise_for_status()
            with path.with_suffix('.partial').open('wb') as output:
                for chunk in response.iter_content(4*1024*1024):output.write(chunk)
        path.with_suffix('.partial').rename(path)
    h=hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda:source.read(4*1024*1024),b''):h.update(chunk)
    value={'repo_id':REPO,'revision':REVISION,'source_file':filename,'sha256':h.hexdigest(),'bytes':path.stat().st_size}
    if manifest.exists() and json.loads(manifest.read_text())!=value:raise ValueError('Ubuntu shard manifest mismatch')
    atomic_json(manifest,value)
    return path,value['sha256']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args();base=args.base;folder=base/'source-downloads/ubuntu_irc';folder.mkdir(parents=True,exist_ok=True)
    plan=json.loads((base/'pipeline/sampling-plan.json').read_text())
    quota=next(s['planned_passages'] for s in plan['source_quotas'] if s['source_id']==SID)
    targets=apportion(quota,{str(i):v for i,v in enumerate(LENGTH_WEIGHTS['social'])})
    spec=json.loads((base/'pipeline/collection-sources.json').read_text())[SID]
    if spec['repo_id']!=REPO or spec['revision']!=REVISION:raise ValueError('Unexpected Ubuntu revision')
    began=time.monotonic();scanned=0
    with (base/(SID+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=connect(base/'collection.sqlite3')
        saved=db.execute("SELECT value FROM settings WHERE key='plan_hash'").fetchone()
        if saved and saved[0]!=digest(json.dumps(plan,sort_keys=True)):raise ValueError('Plan migration must complete first')
        count=db.execute('SELECT count(*) FROM passages WHERE source=?',(SID,)).fetchone()[0];before=count
        bins=dict(db.execute('SELECT bin,count(*) FROM passages WHERE source=? GROUP BY bin',(SID,)))
        for filename in sorted(f for f in spec['files'] if f.startswith('raw/documents/')):
            if count>=quota:break
            cursor_file=VERSION+':'+filename
            cursor=db.execute('SELECT position,done FROM cursors WHERE source=? AND file=?',(SID,cursor_file)).fetchone()
            if cursor and cursor[1]:continue
            path,archive_hash=download(folder,filename);index=-1
            with gzip.open(path,'rt') as stream:
                for index,line in enumerate(stream):
                    if cursor and index<=cursor[0]:continue
                    remaining=[max(0,targets[str(i)]-bins.get(i,0)) for i in range(4)]
                    row=json.loads(line);scanned+=1
                    doc_id=REPO+':'+str(row.get('id') or '')
                    pair=None if db.execute('SELECT 1 FROM passages WHERE source=? AND doc=?',(SID,doc_id)).fetchone() else make_pair(row,filename,index,remaining,archive_hash)
                    if pair or scanned%100==0:
                        db.execute('BEGIN IMMEDIATE')
                        try:
                            if pair and add(db,*pair,quota):count+=1;bins[pair[0]['length_bin']]=bins.get(pair[0]['length_bin'],0)+1
                            db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,?,?)',(SID,cursor_file,index,0));db.commit()
                        except BaseException:db.rollback();raise
                    if scanned%100==0:
                        atomic_json(base/'progress/ubuntu_irc.json',{'source_id':SID,'state':'collecting','count':count,'target':quota,'scanned_this_run':scanned,'updated_at':now()})
                    if count>=quota:break
            with db:db.execute('INSERT OR REPLACE INTO cursors VALUES (?,?,?,?)',(SID,cursor_file,index,int(count<quota)))
        db.close()
        status={'source_id':SID,'state':'quota_filled' if count==quota else 'source_exhausted','count':count,'target':quota,
                'new_rows':count-before,'length_counts':bins,'scanned_this_run':scanned,'elapsed_seconds':time.monotonic()-began,'updated_at':now(),'all_quarantined':True}
        atomic_json(base/'progress/ubuntu_irc.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':main()
