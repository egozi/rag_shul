"""
chunker/preview.py — Preview the multi-table chunk output
==========================================================
Reads the chunks output file and prints 2 sample chunks per table.

Usage:
    python -m chunker.preview
    python -m chunker.preview --file data/chunks_siman.json
"""

import argparse
import json
from pathlib import Path

from .chunker import load_config

HERE = Path(__file__).parent.parent


def main():
    cfg = load_config()
    run_mode = cfg.get("run_mode", "full")
    cfg_paths = cfg.get("paths", {}).get(run_mode, cfg.get("paths", {}))
    default_file = str(HERE / cfg_paths.get("chunks_json", "data/chunks_siman.json"))

    parser = argparse.ArgumentParser(description="Preview chunk tables output")
    parser.add_argument("--file", default=default_file, help="path to chunks JSON file")
    parser.add_argument("--n", type=int, default=2, help="number of chunks to show per table")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as f:
        output = json.load(f)

    if not output:
        print("Empty file.")
        return

    # Multi-table format: list of {metadata, data}
    if isinstance(output[0], dict) and "metadata" in output[0]:
        for table in output:
            type_text = table["metadata"]["type_text"]
            data = table["data"]
            print(f"\n{'=' * 50}")
            print(f"Table: {type_text}  ({len(data)} chunks total)")
            print('=' * 50)
            for row in data[:args.n]:
                print(f"  siman={row['siman']}  seif={row.get('seif', '-')}  id={row['id']}")
                print(f"  text: {row['text'][:120]}...")
                print()
    else:
        # Flat single-table format (backward compat)
        print(f"Single table: {len(output)} chunks")
        for row in output[:args.n]:
            print(f"  siman={row['siman']}  seif={row.get('seif', '-')}  id={row['id']}")
            print(f"  text: {row['text'][:120]}...")
            print()


if __name__ == "__main__":
    main()
