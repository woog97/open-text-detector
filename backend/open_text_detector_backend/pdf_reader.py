"""Read-only local PDF catalogue for the reader prototype; no model execution."""
import hashlib
from pathlib import Path

from fastapi import HTTPException
from .service import public_scan
from .sqlite_runtime import sqlite3


class PDFReader:
    def __init__(self, service, root):
        self.service = service
        self.root = Path(root).resolve()
        self.files = {}
        for path in sorted(self.root.rglob('*.pdf')):
            resolved = path.resolve()
            if resolved.is_relative_to(self.root):
                key = hashlib.sha256(str(path.relative_to(self.root)).encode()).hexdigest()[:24]
                self.files[key] = path

    def path(self, key):
        path = self.files.get(key)
        if not path or not path.is_file() or not path.resolve().is_relative_to(self.root):
            raise HTTPException(404, 'Local PDF not found')
        return path

    def scans(self, source_id=None):
        fields = '*' if source_id else 'id,title,source'
        query = f"SELECT {fields} FROM scans WHERE status='completed' AND deleted_at IS NULL AND source LIKE 'reviewbench:%'"
        args = ()
        if source_id:
            query += " AND source IN (?,?,?)"
            args = tuple(f'reviewbench:{venue}:{source_id}' for venue in ('iclr', 'neurips', 'icml'))
        return self.service.db.all(query + ' ORDER BY created_at DESC', args)

    def list(self):
        scans = {}
        for scan in self.scans():
            scans.setdefault(scan['source'].split(':')[-1], scan)
        titles = {}
        catalogue = self.root / 'reviewbench' / 'catalogue.sqlite3'
        if catalogue.is_file():
            with sqlite3.connect(f'file:{catalogue}?mode=ro', uri=True) as conn:
                ids = [f'iclr:{path.stem}' for path in self.files.values()]
                for offset in range(0, len(ids), 400):
                    batch = ids[offset:offset + 400]
                    marks = ','.join('?' for _ in batch)
                    for row in conn.execute(f'SELECT id,title FROM papers WHERE id IN ({marks})', batch):
                        titles[row[0].split(':')[-1]] = row[1]
        items = []
        for key, path in self.files.items():
            scan = scans.get(path.stem)
            items.append({'id': key, 'title': scan['title'] if scan else titles.get(path.stem, path.stem.replace('_', ' ')),
                          'filename': path.name, 'collection': path.parent.name,
                          'bytes': path.stat().st_size, 'classified': bool(scan)})
        return {'items': sorted(items, key=lambda item: (not item['classified'], item['title']))}

    def detail(self, key):
        path = self.path(key)
        scans = [public_scan(scan, store=self.service.results) for scan in self.scans(path.stem)
                 if scan['source'].split(':')[-1] == path.stem]
        # Token-level arrays aren't used by this passage reader.
        for scan in scans:
            if scan.get('result'):
                scan['result'] = {k: v for k, v in scan['result'].items() if k != 'tokens'}
        stat = path.stat()
        return {'id': key, 'filename': path.name, 'bytes': stat.st_size,
                'version': f'{stat.st_size}-{stat.st_mtime_ns}',
                'reports': scans}
