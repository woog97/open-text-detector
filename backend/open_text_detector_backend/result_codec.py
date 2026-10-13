"""Versioned lossless columnar findings. No pickle or model-dependent reconstruction."""

import hashlib
import json
import struct
import warnings

import numpy as np
import zstandard as zstd

MAGIC = b"PGF1"
MAX_BYTES = 128 * 1024 * 1024


def json_bytes(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _column(values, key, rows):
    extra = {}
    if all(type(v) is str for v in values):
        labels = list(dict.fromkeys(values))
        indices = {v: i for i, v in enumerate(labels)}
        extra["labels"] = labels
        a = np.array([indices[v] for v in values], dtype="<i8")
    elif all(type(v) is int and -(2**62) < v < 2**62 for v in values):
        a = np.array(values, dtype="<i8")
        if key == "start":
            a = np.diff(a, prepend=0)
            extra["transform"] = "delta"
        elif key == "end" and all(type(r.get("start")) is int and 0 <= r["start"] < 2**62 for r in rows):
            a -= np.array([r["start"] for r in rows])
            extra["transform"] = "length"
    elif all(type(v) is float for v in values):
        a = np.array(values, dtype="<f8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            small = a.astype("<f4")
        if np.array_equal(a.view("<u8"), small.astype("<f8").view("<u8")):
            a = small
    else:
        raise ValueError("Unsupported column; retain in JSON metadata")
    if a.dtype.kind == "i":
        for dtype in ["<i1", "<i2", "<i4", "<i8"]:
            if np.array_equal(a, a.astype(dtype).astype("<i8")):
                a = a.astype(dtype)
                break
    b = a.view("u1").reshape(-1, a.dtype.itemsize).T.copy().tobytes()
    return {"key": key, "dtype": a.dtype.str, "bytes": len(b), **extra}, b


def encode(result, *, level=19):
    header = {"meta": dict(result), "tables": {}}
    chunks = []
    for name in ("tokens", "segments", "rectangles"):
        rows = result.get(name)
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            continue
        keys = list(rows[0])
        if not keys or not all(isinstance(row, dict) and list(row) == keys for row in rows):
            continue
        # Decode 'length' offsets only after the start column is available.
        if "end" in keys and "start" in keys and keys.index("end") < keys.index("start"):
            continue
        try:
            columns = [_column([row[k] for row in rows], k, rows) for k in keys]
        except (ValueError, OverflowError):
            continue
        del header["meta"][name]
        header["tables"][name] = {"count": len(rows), "columns": [c for c, _ in columns]}
        chunks.extend(b for _, b in columns)
    h = json_bytes(header)
    raw = struct.pack("<I", len(h)) + h + b"".join(chunks)
    if len(raw) > MAX_BYTES:
        raise ValueError("Result exceeds storage limit")
    return (
        MAGIC
        + struct.pack("<I", len(raw))
        + hashlib.sha256(raw).digest()
        + zstd.ZstdCompressor(level=level).compress(raw)
    )


def decode(blob):
    if len(blob) < 40 or blob[:4] != MAGIC:
        raise ValueError("Unsupported findings format")
    size = struct.unpack("<I", blob[4:8])[0]
    if not 0 < size <= MAX_BYTES:
        raise ValueError("Invalid findings size")
    if zstd.frame_content_size(blob[40:]) != size:
        raise ValueError("Invalid compressed size")
    raw = zstd.ZstdDecompressor().decompress(blob[40:], max_output_size=size)
    if len(raw) != size or hashlib.sha256(raw).digest() != blob[8:40]:
        raise ValueError("Findings checksum mismatch")
    n = struct.unpack("<I", raw[:4])[0]
    if n > size - 4:
        raise ValueError("Invalid findings header")
    header = json.loads(raw[4 : 4 + n])
    result = header["meta"]
    pos = 4 + n
    for name, table in header["tables"].items():
        values = {}
        for column in table["columns"]:
            dtype = column["dtype"]
            if dtype not in ("|i1", "<i2", "<i4", "<i8", "<f4", "<f8"):
                raise ValueError("Invalid column type")
            d = np.dtype(dtype)
            length = column["bytes"]
            if length != table["count"] * d.itemsize or length < 0 or pos + length > size:
                raise ValueError("Invalid column length")
            a = (
                np.frombuffer(raw[pos : pos + length], dtype="u1")
                .reshape(d.itemsize, -1)
                .T.copy()
                .reshape(-1)
                .view(d)
            )
            pos += length
            if column.get("transform") == "delta":
                a = np.cumsum(a, dtype="<i8")
            elif column.get("transform") == "length":
                a = a.astype("<i8") + np.array(values["start"])
            v = a.tolist()
            if "labels" in column:
                v = [column["labels"][i] for i in v]
            values[column["key"]] = v
        result[name] = [dict(zip(values, row)) for row in zip(*values.values())]
    if pos != size:
        raise ValueError("Unexpected findings payload")
    return result
