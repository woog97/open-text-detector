import argparse
from .config import Settings
from .db import Database


def main():
    parser = argparse.ArgumentParser(description="Run the private Open Text Detector classification backend")
    parser.add_argument("command", choices=["init", "serve"], default="serve", nargs="?")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    db = Database(Settings().data_dir)
    print(f"Owner API key: {db.root / 'admin.key'} (keep private)")
    if args.command == "serve":
        import uvicorn
        print(f"API documentation: http://{args.host}:{args.port}/docs")
        uvicorn.run("open_text_detector_backend.main:app", host=args.host, port=args.port, access_log=False,
                    limit_concurrency=32, timeout_keep_alive=5)


if __name__ == "__main__":
    main()
