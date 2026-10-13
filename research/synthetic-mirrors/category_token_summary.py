"""Comparable text-only token statistics for frozen human and snapshot AI pools."""
import collections,datetime,hashlib,json,sqlite3
from pathlib import Path
from tokenizers import Tokenizer

def summarize(human_db,ai_directory,tokenizer_path,out):
    snapshot_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
    ai_files=sorted(Path(ai_directory).glob('*.json'))
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation();tokenizer.no_padding()
    aggregates={pool:collections.defaultdict(lambda:[0,0]) for pool in ['human','ai']}
    def consume(pool,rows):
        batch=[]
        def flush():
            encodings=tokenizer.encode_batch([r['text'] for r in batch],add_special_tokens=False)
            for row,encoded in zip(batch,encodings):
                bucket=aggregates[pool][row['category']];bucket[0]+=1;bucket[1]+=len(encoded.ids)
        for row in rows:
            batch.append(row)
            if len(batch)==256:flush();batch.clear()
        if batch:flush()
    db=sqlite3.connect('file:'+str(human_db)+'?mode=ro',uri=True)
    consume('human',(json.loads(row[0]) for row in db.execute('SELECT row FROM passages')));db.close()
    consume('ai',(json.loads(p.read_text()) for p in ai_files))
    totals={pool:sum(x[0] for x in cats.values()) for pool,cats in aggregates.items()}
    assert totals['human']==100000
    def row(category):
        result={'category':category}
        for pool in ['human','ai']:
            count,tokens=aggregates[pool].get(category,[0,0])
            result.update({pool+'_documents':count,pool+'_percent':count*100/totals[pool] if totals[pool] else 0,pool+'_average_tokens':tokens/count if count else None,pool+'_total_tokens':tokens})
        return result
    rows=[row(c) for c in sorted(set(aggregates['human'])|set(aggregates['ai']))]
    total={'category':'Total'}
    for pool in ['human','ai']:
        tokens=sum(v[1] for v in aggregates[pool].values())
        total.update({pool+'_documents':totals[pool],pool+'_percent':100.0,pool+'_average_tokens':tokens/totals[pool],pool+'_total_tokens':tokens})
    result={'snapshot_at':snapshot_at,'tokenizer':'Existing MELD-v8 training tokenizer','tokenizer_sha256':hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),'count_method':'Stored text, no special tokens, no truncation or padding; excludes API prompts and hidden reasoning','human_db':str(human_db),'ai_directory':str(ai_directory),'ai_snapshot_files_sha256':hashlib.sha256('\n'.join(p.name for p in ai_files).encode()).hexdigest(),'filtering_performed':False,'rows':rows,'total':total}
    Path(out).write_text(json.dumps(result,indent=2)+'\n')
    return result
