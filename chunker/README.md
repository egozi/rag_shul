# Chunker

Reads the Shulchan Arukh RAG JSON and produces a `chunks.json` file for the embedder.

---

## Input

JSON file with this structure (`data/processed/shulchan_aruch_rag.json`):
```json
{
  "simanim": [
    {
      "siman": 1,
      "seifim": [
        { "seif": 1, "text": "...", "hagah": "..." }
      ]
    }
  ]
}
```

---

## Output

Path is set by `paths.chunks_json` in `config/config.yaml`.

**Without `text_variants`** — a flat JSON array, one object per chunk:
```json
[
  { "id": 0, "siman": 1, "seif": 1, "siman_seif": "סימן 1, סעיף 1", "text": "..." },
  { "id": 1, "siman": 1, "seif": null, "siman_seif": "סימן 1", "text": "..." }
]
```

**With `text_variants`** — a JSON array of table objects, one per variant:
```json
[
  {
    "metadata": { "type_text": "text+hagah" },
    "data": [
      { "id": 0, "siman": 1, "seif": 1, "siman_seif": "סימן 1, סעיף 1", "text": "..." }
    ]
  }
]
```

`seif` is `null` for siman-level and sliding-window chunks.

---

## Options (config/config.yaml)

```yaml
chunker:
  mode: seif            # seif | siman | sliding_window
  chunk_size: 200       # words per chunk (sliding_window only)
  overlap: 50           # overlapping words between chunks (sliding_window only)
  chunk_fields:         # seif-level fields joined into the chunk text
    - text
    # - hagah           # uncomment to append Rema commentary
    # - siman_title     # uncomment to prepend the siman heading
  siman_fields:         # siman-level fields prepended to every chunk (all modes)
    # - hilchot_group   # uncomment to prepend the halachic category
    # - siman_sign      # uncomment to prepend the siman sign/marker
  text_variants:        # optional; if present, overrides single-mode output
    - type_text: text+hagah          # label for this table
      chunk_fields: [text, hagah]
      siman_fields: []
    - type_text: text_only
      chunk_fields: [text]
      siman_fields: []
    - type_text: text+hilchot_group
      chunk_fields: [text]
      siman_fields: [hilchot_group]
```

When `text_variants` is present, `build_tables` is called and the output contains one table per variant. The top-level `chunk_fields`, `siman_fields`, and `mode` are ignored for building text; each variant supplies its own fields.

| Mode | Description |
|---|---|
| `seif` | One chunk per seif (default) |
| `siman` | One chunk per siman (all seifim merged) |
| `sliding_window` | Fixed word-count windows across the full corpus |

---

## Run (CLI)

```bash
python3 -m chunker.main
```

All paths are read from `config/config.yaml` (`paths.schema_json`, `paths.chunks_json`).

---

## Use as API

```python
from chunker import build_chunks, build_tables, load_schema

# single-mode: returns list[dict] with id, siman, seif, siman_seif, text
chunks = build_chunks("data/processed/shulchan_aruch_rag.json")

# override mode or fields without touching config:
chunks = build_chunks("data/processed/shulchan_aruch_rag.json", mode="siman", chunk_fields=["text", "hagah"])

# multi-variant: returns list of {metadata, data} table objects
schema = load_schema("data/processed/shulchan_aruch_rag.json")
tables = build_tables(schema, variants=[
    {"type_text": "text+hagah", "chunk_fields": ["text", "hagah"], "siman_fields": []},
])
```
