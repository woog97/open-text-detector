"""Immutable findings objects, backed by a local directory or a private S3/R2 bucket."""

import hashlib
import json
import os
import re
import tempfile
import threading
from collections import OrderedDict

from .result_codec import MAX_BYTES, canonical, decode, encode


class ResultStorageError(Exception):
    pass


class ResultStore:
    def __init__(self, settings, client=None):
        self.settings = settings
        self.root = settings.data_dir.resolve() / "findings"
        self.client = client
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.cache_size = 0
        self.prefix = settings.result_s3_prefix.strip("/")
        if not re.fullmatch(r"[A-Za-z0-9_/-]+", self.prefix):
            raise ValueError("Invalid result object prefix")
        self.store_id = hashlib.sha256(
            json.dumps([settings.result_s3_endpoint, settings.result_s3_bucket, self.prefix]).encode()
        ).hexdigest()
        if settings.result_storage == "s3" and not settings.result_s3_bucket:
            raise ValueError("OTD_RESULT_S3_BUCKET is required")

    def s3(self):
        with self.lock:
            if self.client is None:
                import boto3
                from botocore.config import Config

                s = self.settings
                self.client = boto3.client(
                    "s3",
                    endpoint_url=s.result_s3_endpoint,
                    region_name=s.result_s3_region,
                    aws_access_key_id=s.result_s3_access_key.get_secret_value()
                    if s.result_s3_access_key
                    else None,
                    aws_secret_access_key=s.result_s3_secret_key.get_secret_value()
                    if s.result_s3_secret_key
                    else None,
                    config=Config(
                        connect_timeout=5,
                        read_timeout=15,
                        retries={"max_attempts": 2, "mode": "standard"},
                        s3={"addressing_style": "path"},
                    ),
                )
            return self.client

    def _remember(self, digest, blob):
        if len(blob) > self.settings.result_cache_bytes:
            return
        with self.lock:
            old = self.cache.pop(digest, None)
            if old is not None:
                self.cache_size -= len(old)
            while self.cache and self.cache_size + len(blob) > self.settings.result_cache_bytes:
                _, old = self.cache.popitem(last=False)
                self.cache_size -= len(old)
            self.cache[digest] = blob
            self.cache_size += len(blob)

    def put(self, result):
        try:
            blob = encode(result)
            # Verify exact values before replacing any original database payload.
            if canonical(decode(blob)) != canonical(result):
                raise ValueError("Lossless roundtrip failed")
            digest = hashlib.sha256(blob).hexdigest()
            ref = {
                "format": "pgf1-zstd19",
                "backend": self.settings.result_storage,
                "sha256": digest,
                "bytes": len(blob),
                "json_bytes": len(json.dumps(result).encode()),
            }
            if ref["backend"] == "s3":
                ref.update(store_id=self.store_id, key=f"{self.prefix}/{digest[:2]}/{digest}.pgf")
                self.s3().put_object(
                    Bucket=self.settings.result_s3_bucket,
                    Key=ref["key"],
                    Body=blob,
                    ContentType="application/octet-stream",
                    Metadata={"sha256": digest, "format": "pgf1-zstd19"},
                )
            else:
                folder = self.root / digest[:2]
                folder.mkdir(parents=True, exist_ok=True, mode=0o700)
                fd, temporary = tempfile.mkstemp(dir=folder)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(blob)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, folder / (digest + ".pgf"))
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            # Read back the persisted object, not the cache, before a DB pointer is committed.
            if self._read(ref) != blob:
                raise ValueError("Object verification failed")
            self._remember(digest, blob)
            summary = {k: result[k] for k in ("score", "label", "score_type") if k in result}
            return json.dumps(summary), json.dumps(ref)
        except Exception as error:
            raise ResultStorageError("Could not save verified findings object") from error

    def _validate(self, ref):
        digest = ref.get("sha256", "")
        if not re.fullmatch("[0-9a-f]{64}", digest) or not 0 < ref.get("bytes", 0) <= MAX_BYTES:
            raise ValueError("Invalid object reference")
        if ref.get("format") != "pgf1-zstd19":
            raise ValueError("Unsupported object reference")
        if ref["backend"] == "s3":
            if (
                ref.get("store_id") != self.store_id
                or ref.get("key") != f"{self.prefix}/{digest[:2]}/{digest}.pgf"
            ):
                raise ValueError("Result bucket configuration does not match saved findings")
        elif ref["backend"] != "local":
            raise ValueError("Unsupported object backend")

    def _read(self, ref):
        self._validate(ref)
        digest = ref["sha256"]
        if ref["backend"] == "local":
            with (self.root / digest[:2] / (digest + ".pgf")).open("rb") as stream:
                blob = stream.read(ref["bytes"] + 1)
        else:
            response = self.s3().get_object(Bucket=self.settings.result_s3_bucket, Key=ref["key"])
            stream = response["Body"]
            try:
                if response["ContentLength"] != ref["bytes"]:
                    raise ValueError("Object size mismatch")
                blob = stream.read(ref["bytes"] + 1)
            finally:
                stream.close()
        if len(blob) != ref["bytes"] or hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError("Object integrity check failed")
        return blob

    def get(self, reference):
        try:
            ref = json.loads(reference)
            self._validate(ref)
            with self.lock:
                blob = self.cache.get(ref["sha256"])
                if blob is not None:
                    self.cache.move_to_end(ref["sha256"])
            if blob is None:
                blob = self._read(ref)
                self._remember(ref["sha256"], blob)
            # Always return fresh objects: report serialization/sharing mutates its copy.
            return decode(blob)
        except Exception as error:
            raise ResultStorageError("Saved findings are temporarily unavailable") from error

    def discard(self, reference):
        """Remove only an unpublished result produced by a cancelled worker attempt."""
        ref = json.loads(reference)
        self._validate(ref)
        digest = ref['sha256']
        if ref['backend'] == 'local':
            (self.root / digest[:2] / (digest + '.pgf')).unlink(missing_ok=True)
        else:
            self.s3().delete_object(Bucket=self.settings.result_s3_bucket, Key=ref['key'])
        with self.lock:
            old = self.cache.pop(digest, None)
            if old is not None:
                self.cache_size -= len(old)
