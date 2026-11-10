"""
Build the SQLite database from sql/schema.sql and the CSV files in data/raw/.
"""

import pandas as pd
import sqlite3

from src.config import CURRENT_SEASON, PAST_SEASONS, ROOT, DB_PATH

SCHEMA_PATH = ROOT / "sql" / "schema.sql"
RAW_DIR = ROOT / "data" / "raw"
HISTORY_DIR = RAW_DIR / "history"

# parents before children (foreign keys)
TABLES = ["teams", "positions", "players", "gameweek_stats"]

# file names of the past seasons (positions are only loaded for the current season)
HISTORY_FILES = {
    "teams": "teams.csv",
    "players": "players_raw.csv",
    "gameweek_stats": "merged_gw.csv",
}


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


def clean_rows(conn, table, df, season):
    """
    Remove rows that do not belong in the database: managers
    (element_type 5, only in 2024-25) with their matches, and duplicate matches.
    """
    if table == "players":
        df = df[df["element_type"].between(1, 4)]
    if table == "gameweek_stats":
        players = pd.read_sql(f"SELECT id FROM players WHERE season = '{season}'", conn)
        df = df[df["element"].isin(players["id"])]
        df = df.drop_duplicates(subset=["element", "fixture"])
    return df


def load_table(conn, table, path, season):
    """
    Load data/raw/<table>.csv into the table.
    """
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]

    df = pd.read_csv(path)
    df = clean_rows(conn, table, df, season)
    df["season"] = season
    df = df.reindex(columns=columns)

    df.to_sql(table, conn, if_exists="append", index=False)

    conn.commit()
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table}: {count} rows")


def main():
    """
    Create the database, load the current and the past seasons, and run basic checks.
    """
    conn = create_database()

    for table in TABLES:
        load_table(conn, table, RAW_DIR / f"{table}.csv", CURRENT_SEASON)
        if table in HISTORY_FILES:
            for season in PAST_SEASONS:
                path = HISTORY_DIR / season / HISTORY_FILES[table]
                load_table(conn, table, path, season)

    problems = conn.execute("PRAGMA foreign_key_check").fetchall()
    print("Foreign key problems:", problems)

    conn.close()


if __name__ == "__main__":
    main()
