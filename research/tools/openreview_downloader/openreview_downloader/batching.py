"""Download API attachment batches without trusting archive paths or ordering."""
import io
import zipfile
from pathlib import Path


def unpack_batch(data, notes):
    if len(notes) == 1 and data.startswith(b'%PDF-'):
        return {notes[0].id: data}
    numbers = {}
    for note in notes:
        number = str(note.number)
        if note.number is None or number in numbers:
            raise ValueError('Batch requires unique submission numbers from one venue')
        numbers[number] = note.id
    files = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            number = Path(info.filename).name.split('_', 1)[0]
            fid = numbers.get(number)
            if fid is None or fid in files:
                raise ValueError('Unexpected or duplicate PDF in batch: ' + info.filename)
            pdf = archive.read(info)
            if not pdf.startswith(b'%PDF-'):
                raise ValueError('Non-PDF content in batch: ' + info.filename)
            files[fid] = pdf
    if set(files) != {note.id for note in notes}:
        raise ValueError('Batch is incomplete; no files from this batch were saved')
    return files


def download_batch(client, items):
    """Items are the downloader's (note, category, path, match_info) tuples."""
    if not 1 <= len(items) <= 50:
        raise ValueError('Batch size must be between 1 and 50')
    notes = [item[0] for item in items]
    if len({n.id for n in notes}) != len(notes):
        raise ValueError('Duplicate paper IDs in batch')
    if len(notes) > 1:
        numbers = [n.number for n in notes]
        if None in numbers or len(set(map(str, numbers))) != len(numbers):
            raise ValueError('Batch requires unique submission numbers from one venue')
    kwargs = {'id': notes[0].id} if len(notes) == 1 else {'ids': [n.id for n in notes]}
    data = client.get_attachment(field_name='pdf', **kwargs)
    files = unpack_batch(data, notes)
    for note, _category, path, _match in items:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.part')
        tmp.write_bytes(files[note.id])
        tmp.replace(path)
    return len(files)
