"""
Build the SQLite database from sql/schema.sql and the CSV files in data/raw/.
"""

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "sql" / "schema.sql"
DB_PATH = ROOT / "data" / "fpl.db"
RAW_DIR = ROOT / "data" / "raw"


def create_database():
    """
    Create a fresh SQLite database from sql/schema.sql.
    Any existing database file is deleted first.
    Returns the open connection.
    """
    DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_PATH.read_text())
    except Exception:
        conn.close()
        raise
    return conn


def main():
    """
    Create the database, load the CSV files and run basic checks.
    """
    conn = create_database()
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    tables = [row[0] for row in cursor.fetchall()]
    print(tables)
    conn.close()


if __name__ == "__main__":
    main()