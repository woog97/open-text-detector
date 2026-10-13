import json
import time
import threading
from collections import defaultdict, deque
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.responses import JSONResponse
from .db import digest

bearer = HTTPBearer(auto_error=False)


class RateLimiter:
    def __init__(self):
        self.entries = defaultdict(deque)
        self.lock = threading.Lock()

    def admit(self, key, limit):
        stamp = time.monotonic()
        with self.lock:
            if len(self.entries) > 10000:
                self.entries = defaultdict(deque, {k: v for k, v in self.entries.items() if v and v[-1] > stamp-60})
            q = self.entries[key]
            while q and q[0] < stamp-60:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(stamp)
            return True


def require(scope):
    def dependency(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if credentials is None:
            raise HTTPException(401, "API key required", headers={"WWW-Authenticate": "Bearer"})
        row = request.app.state.db.one("SELECT * FROM api_keys WHERE token_hash=? AND revoked_at IS NULL", (digest(credentials.credentials),))
        if not row:
            raise HTTPException(401, "Invalid or revoked API key", headers={"WWW-Authenticate": "Bearer"})
        scopes = json.loads(row["scopes"])
        if "admin" not in scopes and scope not in scopes:
            raise HTTPException(403, "This API key does not have the required scope")
        if not request.app.state.limiter.admit(row["id"], request.app.state.settings.requests_per_minute):
            raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "60"})
        return row
    return dependency


class BodyLimitMiddleware:
    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > self.limit:
                return await JSONResponse({"detail": "Request is too large"}, status_code=413)(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(chunks)
        async def replay():
            try:
                return next(iterator)
            except StopIteration:
                return await receive()
        await self.app(scope, replay, send)
