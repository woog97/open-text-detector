"""Read-only access to the separately built ReviewBench catalogue."""
import hashlib
import json
from .sqlite_runtime import sqlite3
import zlib
from fastapi import HTTPException
from .service import public_scan


class Papers:
    def __init__(self, findings):
        self.findings, self.db = findings, findings.db
        self.root = findings.root / 'reviewbench'
        self.synced = False

    def connect(self):
        path = self.root / 'catalogue.sqlite3'
        if not path.is_file():
            raise HTTPException(503, 'ReviewBench is still downloading or being indexed. Please try again shortly.')
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def list(self, q='', conference='', year=None, offset=0, limit=20, classified_by='', ids_only=False):
        clauses, args = ['1=1'], []
        if q:
            clauses.append('(instr(lower(title),lower(?))>0 OR instr(lower(authors),lower(?))>0 OR instr(lower(id),lower(?))>0)')
            args.extend([q, q, q])
        if conference:
            clauses.append('conference=?')
            args.append(conference)
        if year:
            clauses.append('year=?')
            args.append(year)
        where = ' AND '.join(clauses)
        conn = self.connect()
        try:
            if not self.synced:
                for scan in self.db.all("SELECT id,text FROM scans WHERE kind='text' AND deleted_at IS NULL AND NOT EXISTS (SELECT 1 FROM paper_scans WHERE scan_id=scans.id)"):
                    digest = hashlib.sha256(scan['text'].encode()).hexdigest()
                    for match in conn.execute('SELECT id FROM papers WHERE text_hash=?', (digest,)):
                        self.link(match['id'], scan['id'])
                self.synced = True
            if classified_by:
                sql = "SELECT DISTINCT p.paper_id FROM paper_scans p JOIN scans s ON s.id=p.scan_id WHERE s.deleted_at IS NULL AND s.status='completed'"
                params = []
                exclude = classified_by.startswith('not:')
                selected = classified_by[4:] if exclude else classified_by
                if selected != 'any':
                    sql += " AND json_extract(s.model_snapshot,'$.model.id')=?"
                    params.append(selected)
                ids = [r['paper_id'] for r in self.db.all(sql, params)]
                where += ' AND id ' + ('NOT IN' if exclude else 'IN') + ' (SELECT value FROM json_each(?))'
                args.append(json.dumps(ids))
            if ids_only:
                return [r[0] for r in conn.execute('SELECT id FROM papers WHERE '+where+' ORDER BY year DESC,conference,title,id', args)]
            total = conn.execute('SELECT count(*) FROM papers WHERE '+where, args).fetchone()[0]
            rows = [dict(r) for r in conn.execute('SELECT id,title,conference,year,decision,authors,word_count,characters,preview FROM papers WHERE '+where+' ORDER BY year DESC,conference,title,id LIMIT ? OFFSET ?', [*args,limit,offset])]
            facets = [dict(r) for r in conn.execute('SELECT conference,year,count(*) AS count FROM papers GROUP BY conference,year ORDER BY year DESC,conference')]
        finally:
            conn.close()
        for row in rows:
            row['authors'] = json.loads(row['authors'])
            row['classifications'] = self.results(row['id'])
        return {'items':rows,'total':total,'facets':facets}

    def results(self, paper_id):
        rows = self.db.all('SELECT s.* FROM scans s JOIN paper_scans p ON s.id=p.scan_id WHERE p.paper_id=? AND s.deleted_at IS NULL ORDER BY s.created_at DESC', (paper_id,))
        return [public_scan(row, False) for row in rows]

    def get(self, paper_id):
        conn = self.connect()
        try:
            row = conn.execute('SELECT * FROM papers WHERE id=?', (paper_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            raise HTTPException(404, 'Paper not found')
        result = dict(row)
        result['text'] = zlib.decompress(result['text']).decode()
        result['authors'] = json.loads(result['authors'])
        result['source_url'] = 'https://openreview.net/forum?id=' + paper_id.split(':', 1)[1]
        # Attach previously computed exact-text reports, including the original baselines.
        for scan in self.db.all("SELECT id FROM scans WHERE text=? AND kind='text' AND deleted_at IS NULL", (result['text'],)):
            self.link(paper_id, scan['id'])
        result['classifications'] = self.results(paper_id)
        result['read_only'] = True
        return result

    def link(self, paper_id, scan_id):
        with self.db.connect() as conn:
            conn.execute('INSERT OR IGNORE INTO paper_scans VALUES(?,?)', (paper_id,scan_id))
            conn.execute('UPDATE scans SET source_locked=1 WHERE id=?', (scan_id,))

    def compute(self, paper_id, model_id):
        paper = self.get(paper_id)
        response = self.findings.compute_text(paper['text'],paper['title'],'reviewbench:'+paper_id,model_id)
        self.link(paper_id,response['scan']['id'])
        return response
