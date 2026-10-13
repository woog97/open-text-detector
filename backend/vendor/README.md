# Bundled SQLite driver

pysqlite3 0.6.0 comes from its verified PyPI source distribution. Its MIT license is retained in the source directory.
SQLite 3.51.3 comes from https://www.sqlite.org/2026/sqlite-amalgamation-3510300.zip (public domain).
The official sqlite3.c SHA3-256 is 32d5424f97e0a7fc5ed2f6335afbb58be4e0298bd7117a34e39d345ff13d859e.

The upstream setup.py automatically compiles the included amalgamation. Local modifications: version suffix and inclusion of sqlite3.c/h in source distributions.
This avoids the SQLite 3.51.1 Unix connection deadlock and includes the 3.51.3 WAL reset fix.
See https://www.sqlite.org/releaselog/3_51_2.html and https://www.sqlite.org/releaselog/3_51_3.html.
Builds require a C compiler (Xcode command line tools on macOS, build-essential on Debian).
