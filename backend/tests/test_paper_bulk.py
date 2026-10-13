import asyncio
import json
import threading
from .test_papers import seed
from .conftest import TEXT


def setup(configured, tmp_path):
    c, admin, app, provider, model, _ = configured
    seed(app, tmp_path)
    import sqlite3
    import hashlib
    import zlib

    with sqlite3.connect(app.state.papers.root / "catalogue.sqlite3") as conn:
        for pid, title in [("three", "A first paper"), ("four", "Z last paper")]:
            text = TEXT + " " + pid
            conn.execute(
                "INSERT INTO papers VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "iclr:" + pid,
                    title,
                    "iclr",
                    2023,
                    "Accept",
                    "[]",
                    "",
                    len(text.split()),
                    len(text),
                    hashlib.sha256(text.encode()).hexdigest(),
                    text[:260],
                    zlib.compress(text.encode()),
                    "test",
                ),
            )
    return c, admin, app, provider, model


def start(c, admin, model, **filters):
    response = c.post("/v1/paper-runs", headers=admin, json={"model_id": model["id"], **filters})
    assert response.status_code == 202, response.text
    return response.json()


def test_complements_and_ordered_bulk_skip_restart(configured, tmp_path):
    c, admin, app, provider, model = setup(configured, tmp_path)
    original = c.post("/v1/papers/iclr:one/compute", headers=admin, json={"model_id": model["id"]}).json()
    asyncio.run(app.state.service.process_one())
    for value in ["not:any", "not:" + model["id"]]:
        data = c.get("/v1/papers", headers=admin, params={"classified_by": value, "year": 2023}).json()
        assert [p["id"] for p in data["items"]] == ["iclr:three", "iclr:four"]
    assert c.get("/v1/papers", headers=admin, params={"classified_by": model["id"]}).json()["total"] == 1
    run = start(c, admin, model, year=2023)
    assert run["total"] == 2
    assert [p["title"] for p in run["upcoming"]] == ["A first paper", "Z last paper"]
    assert c.post("/v1/paper-runs", headers=admin, json={"model_id": model["id"]}).status_code == 409
    app.state.bulk.advance()
    current = app.state.bulk.get(run["id"])["current"]
    assert current["paper_id"] == "iclr:three"
    assert [p["paper_id"] for p in app.state.bulk.get(run["id"])["upcoming"]] == ["iclr:four"]
    asyncio.run(app.state.service.process_one())
    # Recreate the coordinator: persistence survives server/browser restarts.
    from open_text_detector_backend.paper_bulk import PaperBulk

    bulk = PaperBulk(app.state.service, app.state.papers)
    bulk.advance()
    assert bulk.get(run["id"])["current"]["paper_id"] == "iclr:four"
    asyncio.run(app.state.service.process_one())
    bulk.advance()
    done = bulk.get(run["id"])
    assert done["status"] == "completed" and done["counts"] == {"completed": 2}
    assert done["upcoming"] == []
    assert len(provider.calls) == 3
    assert c.get("/v1/scans/" + original["scan"]["id"], headers=admin).status_code == 200
    assert start(c, admin, model, year=2023)["total"] == 0


def test_stop_queued_does_not_touch_manual_jobs(configured, tmp_path):
    c, admin, app, _, model = setup(configured, tmp_path)
    manual = c.post("/v1/scans", headers=admin, json={"text": TEXT}).json()
    run = start(c, admin, model, year=2023)
    app.state.bulk.advance()
    stop = c.post("/v1/paper-runs/" + run["id"] + "/stop", headers=admin, json={"mode": "finish"})
    assert stop.status_code == 200 and stop.json()["status"] == "stopped"
    assert c.get("/v1/scans/" + manual["id"], headers=admin).json()["status"] == "queued"
    assert stop.json()["counts"] == {"cancelled": 3}


def test_stop_running_finish_or_discard(configured, tmp_path):
    c, admin, app, provider, model = setup(configured, tmp_path)
    normal = provider.classify
    for mode in ["finish", "discard"]:
        entered, release = threading.Event(), threading.Event()

        def slow(*args):
            entered.set()
            assert release.wait(10)
            return normal(*args)

        provider.classify = slow
        run = start(c, admin, model, year=2023)
        app.state.bulk.advance()
        worker = threading.Thread(target=lambda: asyncio.run(app.state.service.process_one()))
        worker.start()
        assert entered.wait(5)
        sid = app.state.bulk.get(run["id"])["current"]["scan_id"]
        try:
            stopped = c.post(
                "/v1/paper-runs/" + run["id"] + "/stop", headers=admin, json={"mode": mode}
            ).json()
            assert stopped["status"] == ("stopping" if mode == "finish" else "stopped")
        finally:
            release.set()
            worker.join(10)
        assert not worker.is_alive()
        app.state.bulk.advance()
        row = app.state.db.one("SELECT status,result_storage FROM scans WHERE id=?", (sid,))
        assert row["status"] == ("completed" if mode == "finish" else "cancelled")
        assert bool(row["result_storage"]) == (mode == "finish")
        assert app.state.bulk.get(run["id"])["status"] == "stopped"


def test_cancel_during_storage_removes_unpublished_object(configured, tmp_path, monkeypatch):
    c, admin, app, _, model = setup(configured, tmp_path)
    run = start(c, admin, model, year=2023)
    app.state.bulk.advance()
    put = app.state.service.results.put
    captured = []

    def cancel_on_write(result):
        summary, reference = put(result)
        captured.append(json.loads(reference))
        app.state.bulk.stop(run["id"], "discard")
        return summary, reference

    monkeypatch.setattr(app.state.service.results, "put", cancel_on_write)
    asyncio.run(app.state.service.process_one())
    ref = captured[0]
    assert not (app.state.service.results.root / ref["sha256"][:2] / (ref["sha256"] + ".pgf")).exists()


def test_invalid_paper_and_auth(configured, tmp_path):
    c, admin, app, _, model = setup(configured, tmp_path)
    assert c.get("/v1/paper-runs").status_code == 401
    assert c.post("/v1/paper-runs", json={"model_id": model["id"]}).status_code == 401
    run = start(c, admin, model, year=2026)
    app.state.bulk.advance()
    status = app.state.bulk.get(run["id"])
    assert status["status"] == "completed" and status["counts"]["failed"] == 1 and status["errors"]
