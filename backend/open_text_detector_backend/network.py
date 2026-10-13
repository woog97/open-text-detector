"""Bounded HTTP with DNS pinning. User URLs cannot reach private networks."""
import http.client
import ipaddress
import json
import socket
import ssl
from urllib.parse import urlsplit, urljoin


class NetworkError(ValueError):
    pass


def resolve_target(url, *, model_endpoint=False, allow_local=False):
    try:
        p = urlsplit(url)
        host = p.hostname
        port = p.port or (443 if p.scheme == "https" else 80)
    except ValueError as e:
        raise NetworkError("Invalid URL") from e
    if not host or p.username or p.password or p.scheme not in {"https", "http"} or p.fragment:
        raise NetworkError("Use an HTTP(S) URL without credentials or fragments")
    if not model_endpoint and port not in {80, 443}:
        raise NetworkError("Article URLs must use port 80 or 443")
    try:
        addresses = list(dict.fromkeys(r[4][0] for r in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)))
    except socket.gaierror as e:
        raise NetworkError("Host could not be resolved") from e
    if not addresses:
        raise NetworkError("Host could not be resolved")
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global and not (model_endpoint and allow_local and ip.is_loopback):
            raise NetworkError("Private, reserved, and link-local destinations are blocked")
        if model_endpoint and p.scheme != "https" and not (allow_local and ip.is_loopback):
            raise NetworkError("Model endpoints require HTTPS; loopback HTTP needs explicit server configuration")
    return p, addresses[0], port


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, ip, port, timeout):
        super().__init__(host, port=port, timeout=timeout, context=ssl.create_default_context())
        self.pinned_ip = ip

    def connect(self):
        sock = socket.create_connection((self.pinned_ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def request_bytes(url, *, payload=None, token=None, allow_local=False, model_endpoint=False,
                  max_bytes=5_000_000, timeout=30, redirects=3):
    for _ in range(redirects + 1):
        p, ip, port = resolve_target(url, model_endpoint=model_endpoint, allow_local=allow_local)
        conn = (PinnedHTTPSConnection(p.hostname, ip, port, timeout) if p.scheme == "https"
                else http.client.HTTPConnection(ip, port=port, timeout=timeout))
        headers = {"Host": p.netloc, "User-Agent": "OpenTextDetectorWorkbench/0.1", "Accept-Encoding": "identity"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if token:
            headers["Authorization"] = "Bearer " + token
        try:
            conn.request("POST" if payload is not None else "GET", (p.path or "/") + ("?" + p.query if p.query else ""),
                         body=json.dumps(payload).encode() if payload is not None else None, headers=headers)
            response = conn.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                if model_endpoint:
                    raise NetworkError("Model endpoint redirects are not permitted")
                location = response.getheader("Location")
                if not location:
                    raise NetworkError("Redirect is missing a destination")
                url = urljoin(url, location)
                continue
            if not 200 <= response.status < 300:
                raise NetworkError(f"Upstream returned HTTP {response.status}")
            data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise NetworkError("Upstream response exceeded the size limit")
            return data, response.getheader("Content-Type", ""), url
        except (OSError, http.client.HTTPException) as e:
            raise NetworkError("Upstream connection failed or timed out") from e
        finally:
            conn.close()
    raise NetworkError("Too many redirects")
