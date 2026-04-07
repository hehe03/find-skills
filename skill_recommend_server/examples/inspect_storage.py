import argparse
import json
import pickle
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "skills.db"
DEFAULT_INDEX = PROJECT_ROOT / "data" / "faiss_index"


def inspect_db(db_path: Path, limit: int) -> None:
    print("== SQLite ==")
    print(f"Path: {db_path}")
    if not db_path.exists():
        print("Database file not found.\n")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        skills = conn.execute("SELECT * FROM skills ORDER BY id LIMIT ?", (limit,)).fetchall()
        feedback = conn.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        print(f"skills rows: {conn.execute('SELECT COUNT(*) FROM skills').fetchone()[0]}")
        print(f"feedback rows: {conn.execute('SELECT COUNT(*) FROM feedback').fetchone()[0]}")
        for row in skills:
            item = dict(row)
            for field in ("category", "capabilities", "input_schema", "output_schema", "dependencies", "usage_stats"):
                if field in item and item[field]:
                    try:
                        item[field] = json.loads(item[field])
                    except json.JSONDecodeError:
                        pass
            print(json.dumps(item, ensure_ascii=False, indent=2))
        if feedback:
            print("-- recent feedback --")
            for row in feedback:
                print(json.dumps(dict(row), ensure_ascii=False, indent=2))
    finally:
        conn.close()
    print()


def inspect_faiss(index_base: Path, limit: int) -> None:
    print("== FAISS ==")
    print(f"Base path: {index_base}")
    index_file = Path(f"{index_base}.index")
    meta_file = Path(f"{index_base}.meta")
    state_file = Path(f"{index_base}.state.json")

    print(f"index exists: {index_file.exists()}")
    print(f"meta exists: {meta_file.exists()}")
    print(f"state exists: {state_file.exists()}")

    if state_file.exists():
        print("-- vector state --")
        print(state_file.read_text(encoding='utf-8'))

    if meta_file.exists():
        print("-- vector metadata sample --")
        with open(meta_file, "rb") as file:
            metadata = pickle.load(file)
        print(f"metadata rows: {len(metadata)}")
        for item in metadata[:limit]:
            print(json.dumps(item, ensure_ascii=False, indent=2, default=str))

    if index_file.exists():
        try:
            import faiss

            index = faiss.read_index(str(index_file))
            print(f"index ntotal: {index.ntotal}")
            print(f"index dimension: {index.d}")
        except Exception as exc:
            print(f"Unable to inspect FAISS index binary directly: {exc}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect SQLite and FAISS storage contents.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to skills.db")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Base path to faiss_index")
    parser.add_argument("--limit", type=int, default=5, help="How many rows to print")
    args = parser.parse_args()

    inspect_db(args.db, args.limit)
    inspect_faiss(args.index, args.limit)


if __name__ == "__main__":
    main()
