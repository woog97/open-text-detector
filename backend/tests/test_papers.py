import asyncio
import hashlib
import sqlite3
import zlib
from .conftest import TEXT


def seed(app, tmp_path):
    root = tmp_path / 'reviewbench'
    root.mkdir()
    app.state.papers.root = root
    conn = sqlite3.connect(root / 'catalogue.sqlite3')
    conn.execute('CREATE TABLE papers(id TEXT PRIMARY KEY,title TEXT,conference TEXT,year INTEGER,decision TEXT,authors TEXT,abstract TEXT,word_count INTEGER,characters INTEGER,text_hash TEXT,preview TEXT,text BLOB,source_file TEXT)')
    for pid,title,year,text in [('iclr:one','Example paper',2023,TEXT),('iclr:two','Missing paper',2026,'')]:
        conn.execute('INSERT INTO papers VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(pid,title,'iclr',year,'Accept','["An Author"]','Abstract',len(text.split()),len(text),hashlib.sha256(text.encode()).hexdigest(),text[:260],zlib.compress(text.encode()),'test.parquet'))
    conn.commit()
    conn.close()


def test_paper_read_only_and_cached_model_results(configured,tmp_path):
    c,admin,app,provider,model,_=configured
    seed(app,tmp_path)
    assert c.get('/v1/papers').status_code==401
    result=c.get('/v1/papers?year=2023',headers=admin).json()
    assert result['total']==1 and 'text' not in result['items'][0]
    assert c.get('/v1/papers?q=author',headers=admin).json()['total']==2
    body={'model_id':model['id']}
    first=c.post('/v1/papers/iclr:one/compute',headers=admin,json=body)
    assert first.status_code==202,first.text
    scan=first.json()['scan']
    assert scan['source_locked']==1
    asyncio.run(app.state.service.process_one())
    again=c.post('/v1/papers/iclr:one/compute',headers=admin,json=body).json()
    assert again['reused'] and again['scan']['id']==scan['id'] and len(provider.calls)==1
    detail=c.get('/v1/papers/iclr:one',headers=admin).json()
    assert detail['text']==TEXT and detail['read_only'] and len(detail['classifications'])==1
    assert detail['classifications'][0]['result']['label']=='ai_assisted'
    assert c.patch('/v1/scans/'+scan['id'],headers=admin,json={'text':'edited'}).status_code==422
    assert c.get('/v1/papers',headers=admin).json()['items'][1]['classifications'][0]['id']==scan['id']
    assert c.get('/v1/papers/iclr:absent',headers=admin).status_code==404
    assert c.post('/v1/papers/iclr:two/compute',headers=admin,json=body).status_code==422


def test_paper_links_preexisting_report(configured,tmp_path):
    c,admin,app,_,_,_=configured
    seed(app,tmp_path)
    scan=c.post('/v1/scans',headers=admin,json={'text':TEXT}).json()
    detail=c.get('/v1/papers/iclr:one',headers=admin).json()
    assert detail['classifications'][0]['id']==scan['id']
    assert c.get('/v1/scans/'+scan['id'],headers=admin).json()['source_locked']==1


def test_classified_model_filter_counts_completed_only(configured,tmp_path):
    c,admin,app,_,model,payload=configured
    seed(app,tmp_path)
    first=c.post('/v1/papers/iclr:one/compute',headers=admin,json={'model_id':model['id']}).json()['scan']
    def filtered(value):
        return c.get('/v1/papers',headers=admin,params={'classified_by':value}).json()
    assert filtered('')['total']==2
    assert filtered(model['id'])['total']==0
    asyncio.run(app.state.service.process_one())
    assert filtered(model['id'])['total']==1
    assert filtered('any')['total']==1
    assert filtered('missing')['total']==0
    assert c.get('/v1/papers',headers=admin,params={'classified_by':model['id'],'year':2026}).json()['total']==0
    detail=c.get('/v1/scans/'+first['id'],headers=admin).json()
    metrics=detail['result']['performance']
    assert metrics['classification_seconds']>=0
    assert metrics['processing_seconds']>=metrics['classification_seconds']
    assert metrics['attempt']==1 and metrics['queue_seconds']>=0
    assert metrics['input_utf8_bytes']==len(TEXT.encode()) and metrics['stored_result_bytes']>0
    assert detail['result']['completed_at']
    app.state.db.execute("UPDATE scans SET deleted_at='removed' WHERE id=?",(first['id'],))
    assert filtered('any')['total']==0
