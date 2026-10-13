"""Start a token-protected notebook server without printing credentials."""

import os
from pathlib import Path
import subprocess
import sys


def main():
    token = os.environ.get("JUPYTER_TOKEN", "")
    if len(token) < 32:
        raise SystemExit("Set a strong JUPYTER_TOKEN Space Secret (at least 32 characters).")

    durable = Path("/data")
    root = Path(os.environ.get("WORKSPACE_DIR", str(
        durable / "workspace" if durable.is_dir() and os.access(durable, os.W_OK) else Path.home() / "workspace"
    )))
    root.mkdir(parents=True, exist_ok=True)
    if root.parent != durable:
        print("Temporary workspace: attach writable storage at /data to preserve work across restarts.")

    if os.environ.get("RUN_GPU_SMOKE_TEST") == "1":
        marker = root / "smoke-tests" / ".startup-test-passed"
        if not marker.exists():
            try:
                subprocess.run(
                    [sys.executable, str(Path(__file__).with_name("gpu_smoke_test.py")),
                     "--output-dir", str(root / "smoke-tests")],
                    check=True, timeout=90,
                )
                marker.write_text("Synthetic GPU training and checkpoint resume passed.\n")
            except (subprocess.SubprocessError, OSError) as exc:
                print(f"GPU_SMOKE_TEST_FAILED {type(exc).__name__}: {exc}", flush=True)
        else:
            print("GPU_SMOKE_TEST_ALREADY_PASSED", flush=True)

    from jupyter_server.serverapp import ServerApp
    from traitlets.config import Config

    # Token comes from the environment, never command-line arguments or log output.
    config = Config()
    config.ServerApp.jpserver_extensions = {"jupyterlab": True}
    app = ServerApp.instance(config=config)
    app.initialize([
        f"--ServerApp.ip={os.environ.get('BIND_HOST', '0.0.0.0')}",
        f"--ServerApp.port={os.environ.get('PORT', '7860')}",
        "--ServerApp.port_retries=0",
        "--ServerApp.open_browser=False",
        "--ServerApp.allow_remote_access=True",
        "--ServerApp.default_url=/lab",
        f"--ServerApp.root_dir={root}",
        "--ServerApp.log_level=WARN",
    ])
    app.identity_provider.token = token
    app.start()


if __name__ == "__main__":
    main()
