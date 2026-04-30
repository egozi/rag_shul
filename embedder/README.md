# Embedder

Reads the chunker output (`chunks_siman.json`) and stores sentence embeddings in a ChromaDB collection.

---

## Input

`data/chunks_siman.json` — output of the chunker: a JSON array of table objects, one per variant.

```json
[
  {
    "metadata": { "type_text": "text+hagah" },
    "data": [
      { "id": 0, "siman": 1, "seif": 1, "siman_seif": "סימן 1, סעיף 1", "text": "..." },
      ...
    ]
  },
  {
    "metadata": { "type_text": "text_only" },
    "data": [ ... ]
  },
  {
    "metadata": { "type_text": "text+hilchot_group" },
    "data": [ ... ]
  }
]
```

Each table has:
- `metadata.type_text` — variant name (used as a label in ChromaDB)
- `data` — list of chunks with `siman`, `seif`, `siman_seif`, `text`

---

## Process

For each table:

1. Build encoding text per chunk:
   ```
   "passage: שולחן ערוך אורח חיים, סימן N, סעיף M: <text>"
   ```
2. Encode all texts into normalized 1024-dim vectors using `intfloat/multilingual-e5-large`
3. Store in ChromaDB

All tables are unified into a **single ChromaDB collection**.

---

## Output

`embedder/chroma_db/` — persistent ChromaDB collection named `shulchan_arukh_seifs`.

Each record contains:

| Field | Value | Example |
|-------|-------|---------|
| `id` | `{type_text}__siman_{N}_seif_{M}` | `text+hagah__siman_1_seif_1` |
| `document` | raw text | `"יתגבר כארי לעמוד..."` |
| `embedding` | 1024-dim float32 vector | `[0.0503, 0.0000, ...]` |
| `metadata.siman` | int | `1` |
| `metadata.seif` | int | `1` |
| `metadata.type_text` | str | `"text+hagah"` |

Total records: 3 variants × 4,168 chunks = **12,504 records**.

---

## Run (CLI)

```bash
python embedder/embed.py --chunks data/chunks_siman.json
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--chunks` | required | Path to `chunks_siman.json` |
| `--model` | `intfloat/multilingual-e5-large` | Embedding model |
| `--chroma-dir` | `embedder/chroma_db` | ChromaDB directory |
| `--collection` | `shulchan_arukh_seifs` | Collection name |
| `--batch-size` | `32` | Encoding batch size |

---

## Use as API

```python
from embedder.embed import encode_query

# encode a query at retrieval time
query_vec = encode_query("מה דין ציצית?")
# returns: normalized 1024-dim numpy array
```

---

## Query ChromaDB

```python
import chromadb

client = chromadb.PersistentClient("embedder/chroma_db")
col    = client.get_collection("shulchan_arukh_seifs")

# query a specific variant
results = col.query(
    query_embeddings=[query_vec.tolist()],
    n_results=10,
    where={"type_text": "text+hagah"},
)
```
