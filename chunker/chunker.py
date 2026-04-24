"""
chunker.py — Seif-level chunker for Shulchan Arukh RAG pipeline
================================================================
Reads the RAG JSON (produced by Member 1) and builds a flat DataFrame
where each row is one seif — the unit sent to the embedder.

Input JSON structure:
    {
      "title": "שולחן ערוך, אורח חיים",
      "source": "Torat Emet 363",
      "simanim": [
        {
          "siman": 1,
          "seifim": [
            {
              "seif": 1,
              "text": "יתגבר כארי...",
              "hagah": "ועל כל פנים...",   (Rema commentary, or null)
              "text_raw": "..."             (raw text with HTML, not used)
            }
          ]
        }
      ]
    }

Output DataFrame columns:
    siman      (int)  — chapter number
    seif       (int)  — sub-chapter number
    siman_seif (str)  — "סימן N, סעיף M"  (matches מקור column in eval CSV)
    breadcrumb (str)  — "<title>, סימן N, סעיף M"  (prepended to text at embed time)
    text       (str)  — text + hagah combined, sent to the embedder

Public API:
    load_schema(json_path)        → dict
    build_dataframe(schema, ...)  → DataFrame
    build_csv(json, csv, fields)  → writes CSV to disk
"""

import json
from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config_template.yaml"

# Default title used when the schema has no "title" key (defensive fallback).
DEFAULT_TITLE = "שולחן ערוך, אורח חיים"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema(json_path: str | Path) -> dict:
    """Load the RAG JSON from disk."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_dataframe(schema: dict, chunk_fields: list[str] | None = None) -> pd.DataFrame:
    """
    Convert the RAG JSON dict into a flat DataFrame (one row per seif).

    Args:
        schema: parsed JSON dict
        chunk_fields: ordered list of seif fields to join into the text column.
                      Defaults to the chunk_fields list in config_template.yaml.

    Returns:
        DataFrame with columns: siman, seif, siman_seif, breadcrumb, text
        Sorted by siman then seif, with a clean integer index.
    """
    if chunk_fields is None:
        chunk_fields = load_config()["chunker"]["chunk_fields"]

    title = schema.get("title") or DEFAULT_TITLE

    rows = []
    for siman_data in schema["simanim"]:
        siman_num = siman_data["siman"]
        for seif_data in siman_data["seifim"]:
            seif_num = seif_data["seif"]
            parts = [seif_data.get(f) for f in chunk_fields if seif_data.get(f)]
            text = " ".join(parts)
            siman_seif = f"סימן {siman_num}, סעיף {seif_num}"
            rows.append({
                "siman":      siman_num,
                "seif":       seif_num,
                "siman_seif": siman_seif,
                "breadcrumb": f"{title}, {siman_seif}",
                "text":       text,
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["siman", "seif"]).reset_index(drop=True)


def build_csv(
    json_path: str | Path,
    csv_path:  str | Path,
    chunk_fields: list[str] | None = None,
) -> Path:
    """
    Pipeline entry point: JSON → chunks CSV.

    Reads the RAG JSON, flattens it into a DataFrame (via build_dataframe),
    and writes it to `csv_path`. Creates parent directories if needed.

    Returns the path to the written CSV.
    """
    schema = load_schema(json_path)
    df = build_dataframe(schema, chunk_fields)

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path
