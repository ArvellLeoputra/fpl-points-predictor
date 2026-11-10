"""
Build the SQLite database from sql/schema.sql and the CSV files in data/raw/.
"""

import pandas as pd
import sqlite3

from src.config import ROOT, DB_PATH

SCHEMA_PATH = ROOT / "sql" / "schema.sql"
RAW_DIR = ROOT / "data" / "raw"

# parents before children (foreign keys)
TABLES = ["teams", "positions", "players", "gameweek_stats"]


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


def load_table(conn, table):
    """
    Load data/raw/<table>.csv into the table.
    """
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]

    df = pd.read_csv(RAW_DIR / f"{table}.csv")
    df = df[columns]

    df.to_sql(table, conn, if_exists="append", index=False)

    conn.commit()
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table}: {count} rows")


def main():
    """
    Create the database, load the CSV files and run basic checks.
    """
    conn = create_database()

    for table in TABLES:
        load_table(conn, table)

    problems = conn.execute("PRAGMA foreign_key_check").fetchall()
    print("Foreign key problems:", problems)

    conn.close()


if __name__ == "__main__":
    main()
