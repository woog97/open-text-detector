"""Register the pinned local baseline and make localization the default."""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    token = (ROOT / "backend/.data/admin.key").read_text().strip()

    def api(path, method="GET", data=None):
        request = urllib.request.Request("http://127.0.0.1:8000/v1/" + path,
            data=json.dumps(data).encode() if data is not None else None, method=method,
            headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    model_id = "anon-review-meld-2026/meld"
    existing = next((m for m in api("models")["items"] if m["model_id"] == model_id), None)
    payload = {"name": "MELD v5 · Local evidence", "provider": "meld", "task": "text",
               "model_id": model_id, "enabled": True}
    model = api("models/" + existing["id"], "PUT", payload) if existing else api("models", "POST", payload)
    api("settings/default-model", "PUT", {"model_id": model["id"]})
    print(json.dumps({"default_model": model["name"], "id": model["id"]}))


if __name__ == "__main__":
    main()
