# rag_shul
Run from the exp_main.py
Files are created next to the exp_main.py file to examine the project flow.
config.yaml settings.

---

## Embedder

**File:** `embedder/embed.py`
**Entry point called by pipeline:** `build_embeddings(csv, npy, model, ...)`

### What it does

Receives the chunks JSON (output of the chunker), converts every chunk into a numeric vector (embedding), and saves the result as a `.npy` file. Optionally enriches each chunk with cached GPT data (summaries, questions) before encoding, and optionally stores the vectors in up to 4 vector databases.

---

### Input / Output

| | Description |
|---|---|
| **Input (primary)** | `chunks_v1.json` — produced by the chunker (fields: `siman`, `seif`, `text`) |
| **Input (legacy)** | `chunks_v1.csv` — accepted for backwards compatibility with existing experiments |
| **Input (optional)** | Cache JSON files from `data/` — `modern_summary`, `questions` (GPT-generated) |
| **Output (always)** | `embeddings_v1.npy` — matrix of shape `(num_chunks × 1024)`, one row per chunk |
| **Output (optional)** | ChromaDB / LanceDB / Qdrant / Milvus — enabled per DB in config |

---

### How it works (step by step)

```
1. Load chunks_v1.json  (or .csv for legacy experiments)
2. Merge enrichment caches (if enrich_fields is set)
       e.g. adds modern_summary and questions columns to each row
3. Build passage text per chunk:
       "passage: " + text  [+ " | " + modern_summary + " | " + questions]
4. Encode all texts with SentenceTransformer (batch_size=32, normalize=True)
5. Save to embeddings_v1.npy
6. Save to enabled vector DBs (Chroma / LanceDB / Qdrant / Milvus)
```

---

### Config fields (`config.yaml → embeddings:`)

```yaml
embeddings:
  model: "intfloat/multilingual-e5-large"  # embedding model to use
  batch_size: 32                            # chunks encoded per batch
  prefix_passage: "passage: "              # prepended to every chunk during encoding
  prefix_query:   "query: "               # prepended to every query during retrieval

  # Enrichment — append extra fields to each chunk's text before encoding
  # Leave empty [] for baseline (text only)
  # Add field names to include GPT-generated data
  enrich_fields: []                        # e.g. [modern_summary, questions]
  enrich_separator: " | "                  # separator between fields

  # Cache files for enrichment fields (relative to project root)
  # Only used if the matching field appears in enrich_fields above
  enrich_caches:
    modern_summary: "data/seif_modern_summary_cache.json"
    questions:      "data/seif_questions_gpt_cache.json"

  # Vector DB outputs — set enabled: true to activate
  outputs:
    npy: true          # always true — main retrieval path
    chroma:
      enabled: false
      dir: "embedder/chroma_db"
      collection: "shulchan_arukh_seifs"
    lancedb:
      enabled: false
      dir: "embedder/lancedb_db"
      table: "shulchan_arukh_seifs"
    qdrant:
      enabled: false
      path: "embedder/qdrant_db"
      collection: "shulchan_arukh_seifs"
    milvus:
      enabled: false
      path: "embedder/milvus.db"
      collection: "shulchan_arukh_seifs"
```

| Field | Type | Description |
|---|---|---|
| `model` | `str` | SentenceTransformer model name. Default: `intfloat/multilingual-e5-large` (Hebrew + 100 languages, 1024-dim vectors). |
| `batch_size` | `int` | Number of chunks encoded per batch. Lower = less memory. |
| `prefix_passage` | `str` | Prepended to each chunk text during encoding. Required by E5 model. |
| `prefix_query` | `str` | Prepended to each query at retrieval time. Required by E5 model. |
| `enrich_fields` | `list[str]` | Fields to append to chunk text before encoding. `[]` = baseline (text only). |
| `enrich_separator` | `str` | String placed between fields when building the passage text. |
| `enrich_caches` | `dict` | Path to cache JSON per enrichment field. Cache key format: `"siman_seif"` e.g. `"128_3"`. |
| `outputs.*` | `dict` | One block per vector DB. Set `enabled: true` to write to that DB. |

---

### Common configurations

**Baseline — text only (fastest):**
```yaml
enrich_fields: []
```

**With GPT enrichment (best recall — exp_028):**
```yaml
enrich_fields: [modern_summary, questions]
```

**Save to LanceDB in addition to NPY:**
```yaml
outputs:
  lancedb:
    enabled: true
    dir: "embedder/lancedb_db"
    table: "shulchan_arukh_seifs"
```

---

### Output file (`embeddings_v1.npy`)

NumPy matrix of shape `(N × 1024)` where N = number of chunks.
Row `i` is the L2-normalized embedding of chunk `i` from `chunks_v1.json`.
This file is the direct input to the retriever (`retrievers/retrieval_npy.py`).
