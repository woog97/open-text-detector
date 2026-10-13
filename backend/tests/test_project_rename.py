"""Guards for the rename from Pangram to Open Text Detector (October 11, 2026).

The backend package moved from pangram_backend to open_text_detector_backend and its settings from PANGRAM_* to
OTD_*. These tests check that nothing in the repository still imports the old package, that every import of the new
one names a module and attribute that exist, that old settings still work, and that "pangram" appears in the
renamed folders only where it names something that keeps that name (see ALLOWED).
"""
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from open_text_detector_backend.config import Settings

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
PACKAGE = "open_text_detector_backend"
PACKAGE_DIR = BACKEND / PACKAGE
THIS_FILE = Path(__file__).resolve()  # names the old package and patterns on purpose

# Every remaining "pangram" in the renamed folders must match one of these, with the reason it stays.
ALLOWED = {
    r"pangram-paper-atlas": "R2 bucket and the old Worker name, which now only redirects",
    r"\(as pangram-(iclr-bulk-upload|meld-run|meld-transfer)\)":
        "old name of a Worker deleted on Oct 13, 2026, in the note that says to redeploy it under the new name",
    r"/tmp/pangram-|/private/tmp/pangram-": "local scratch files written by other scripts",
    r"/workspace/(woog/)?pangram": "paths on the GPU hosts, which stay",
    r"pangram-baseline-2026-09-22-v1": "random seed string; changing it changes the baseline sample",
    r"Pangram Labs|Pangram['’]s|pangramlabs|pangram\.com|pangram/editlens|[Oo]pen[- ]Pangram|Pangram 4":
        "Pangram Labs, its product, models and reports",
    r"open-pangram-llama": "id of Pangram Labs' open model, stored in existing workspace databases",
    r"terracotta \"pangram\" branding": "dated changelog entry describing the old design",
    r"PANGRAM_API_KEY|\.env\.pangram|pangram_api_test|Pangram API|Pangram returned": "client for Pangram Labs' API",
    r"LEGACY_ENV_PREFIX = \"PANGRAM_\"|PANGRAM_\*": "old setting names, still read",
}
RENAMED = ["backend", "app", "scripts", "browser-extension", "deploy"]


def tracked(*paths):
    out = subprocess.run(["git", "ls-files", "-z", "--", *paths], cwd=REPO, capture_output=True, check=True)
    return [REPO / p for p in out.stdout.decode().split("\0") if p and (REPO / p).resolve() != THIS_FILE]


def test_old_package_is_gone():
    assert not (BACKEND / "pangram_backend").exists()
    stale = [str(p.relative_to(REPO)) for p in tracked(".") if p.is_file() and p.suffix not in {".lock"}
             and "pangram_backend" in p.read_text(errors="ignore")]
    assert stale == [], f"still refer to pangram_backend: {stale}"


def module_names(path: Path):
    """Top-level names a module defines or imports."""
    tree = ast.parse(path.read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in node.targets if isinstance(node, ast.Assign) else [node.target]:
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.If, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                    names.update((a.asname or a.name).split(".")[0] for a in sub.names)
                elif isinstance(sub, ast.Assign):
                    names.update(n.id for t in sub.targets for n in ast.walk(t) if isinstance(n, ast.Name))
    return names


def resolve(module: str):
    rel = Path(*module.split(".")[1:])
    for candidate in (PACKAGE_DIR / rel).with_suffix(".py"), PACKAGE_DIR / rel / "__init__.py":
        if candidate.exists():
            return candidate
    return None


def test_every_import_of_the_package_resolves():
    """Covers scripts and research code that only run on other machines, without importing torch etc."""
    problems, checked = [], 0
    for path in tracked("*.py"):
        if not path.is_file() or PACKAGE not in (source := path.read_text(errors="ignore")):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue  # files that do not parse here are not ours to judge
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 \
                    and node.module.split(".")[0] == PACKAGE:
                target = resolve(node.module)
                checked += 1
                if target is None:
                    problems.append(f"{path.relative_to(REPO)}: no module {node.module}")
                    continue
                defined = module_names(target)
                for alias in node.names:
                    if alias.name != "*" and alias.name not in defined and not resolve(f"{node.module}.{alias.name}"):
                        problems.append(f"{path.relative_to(REPO)}: {node.module} has no {alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == PACKAGE:
                        checked += 1
                        if resolve(alias.name) is None and alias.name != PACKAGE:
                            problems.append(f"{path.relative_to(REPO)}: no module {alias.name}")
    assert checked > 100, f"only {checked} imports found; the scan is not seeing the scripts"
    assert problems == []


def test_package_runs_as_a_module():
    out = subprocess.run([sys.executable, "-m", PACKAGE, "--help"], cwd=BACKEND, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "Open Text Detector" in out.stdout


def test_build_files_name_the_new_package():
    docker = (BACKEND / "Dockerfile").read_text()
    assert f"COPY {PACKAGE} ./{PACKAGE}" in docker and f'"-m", "{PACKAGE}"' in docker
    assert 'name = "open-text-detector-backend"' in (BACKEND / "pyproject.toml").read_text()
    assert 'name = "open-text-detector-backend"' in (BACKEND / "uv.lock").read_text()


@pytest.fixture
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("OTD_", "PANGRAM_")):
            monkeypatch.delenv(key)
    return monkeypatch


def test_settings_read_the_new_names(clean_env):
    clean_env.setenv("OTD_REQUESTS_PER_MINUTE", "7")
    assert Settings(_env_file=None).requests_per_minute == 7


def test_old_setting_names_still_work(clean_env):
    clean_env.setenv("PANGRAM_REQUESTS_PER_MINUTE", "9")
    clean_env.setenv("PANGRAM_WORKER_ENABLED", "false")
    s = Settings(_env_file=None)
    assert (s.requests_per_minute, s.worker_enabled) == (9, False)


def test_new_name_wins_over_old(clean_env):
    clean_env.setenv("PANGRAM_REQUESTS_PER_MINUTE", "9")
    clean_env.setenv("OTD_REQUESTS_PER_MINUTE", "7")
    assert Settings(_env_file=None).requests_per_minute == 7


def test_old_names_in_a_dotenv_file_still_work(clean_env, tmp_path):
    env = tmp_path / ".env"
    env.write_text("PANGRAM_REQUESTS_PER_MINUTE=11\nOTD_PROVIDER_TIMEOUT=33\n")
    s = Settings(_env_file=env)
    assert (s.requests_per_minute, s.provider_timeout) == (11, 33)


def test_disabling_the_dotenv_file_also_disables_old_names_in_it(clean_env, tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("PANGRAM_REQUESTS_PER_MINUTE=11\n")
    monkeypatch.chdir(tmp_path)
    assert Settings(_env_file=None).requests_per_minute == 120


def test_example_settings_use_the_new_names():
    keys = [line.split("=")[0].lstrip("# ") for line in (BACKEND / ".env.example").read_text().splitlines()
            if "=" in line]
    assert keys and all(k.startswith("OTD_") or k == "HF_TOKEN" for k in keys), keys


def test_pangram_only_remains_where_allowed():
    allowed = re.compile("|".join(f"(?:{p})" for p in ALLOWED))
    leftovers = []
    for path in tracked(*RENAMED):
        if not path.is_file() or path.suffix in {".lock", ".png", ".pdf", ".woff2", ".ico"} or "/dist/" in str(path):
            continue
        for n, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if re.search("pangram", allowed.sub("", line), re.I):
                leftovers.append(f"{path.relative_to(REPO)}:{n}: {line.strip()[:120]}")
    assert leftovers == [], "\n".join(leftovers)
