"""
chunker/main.py — entry point for the chunker
==============================================
Loads the RAG JSON, builds the chunk tables, and saves to JSON.
All paths and settings are read from config/config.yaml.

Usage:
    python -m chunker.main
"""

import json
from pathlib import Path

from .chunker import load_schema, build_dataframe, build_tables, load_config

HERE = Path(__file__).parent.parent  # project root


def main():
    cfg        = load_config()
    run_mode   = cfg.get("run_mode", "full")
    cfg_paths  = cfg.get("paths", {}).get(run_mode, cfg.get("paths", {}))
    input_path  = HERE / cfg_paths.get("schema_json", "data/processed/shulchan_aruch_rag.json")
    output_path = HERE / cfg_paths.get("chunks_json",  "data/chunks.json")

    print(f"Loading: {input_path}")
    schema = load_schema(input_path)

    variants = cfg["chunker"].get("text_variants")
    if variants:
        output = build_tables(schema, variants)
        print(f"Tables:  {len(output)} variants")
        for table in output:
            print(f"  [{table['metadata']['type_text']}]  {len(table['data'])} chunks")
    else:
        df = build_dataframe(schema)
        output = [{"id": i, **row} for i, row in enumerate(df.to_dict(orient="records"))]
        print(f"Chunks:  {len(output)} chunks across {df['siman'].nunique()} simanim")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:   {output_path}")


if __name__ == "__main__":
    main()
