import sys, hashlib, pandas as pd
src, out = sys.argv[1], sys.argv[2]
p = pd.read_parquet(src + '/papers.parquet')
print(len(p), p.label.value_counts().to_dict(), p.source.value_counts().to_dict())
for view, col in [('scislop-prose', 'body_txt'), ('scislop-tex', 'body_tex')]:
    d = pd.DataFrame({'id': p.paper_id, 'text': p[col].fillna('')})
    d['text_sha256'] = [hashlib.sha256(t.encode()).hexdigest() for t in d.text]
    d['pdf_sha256'] = ''
    d.to_parquet(f'{out}/{view}/train-00000.parquet', index=False)
    print(view, (d.text.str.len()==0).sum(), 'empty;', int(d.text.str.len().sum()), 'chars')
p[['paper_id','label','source','role']].to_parquet(out + '/labels.parquet', index=False)
