"""Use the bundled SQLite runtime, including the Unix deadlock and WAL fixes."""
import pysqlite3 as sqlite3

if sqlite3.sqlite_version_info < (3, 51, 3):
    raise RuntimeError("SQLite 3.51.3 or newer is required; sync the backend dependencies.")
