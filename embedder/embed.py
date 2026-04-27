"""
embed.py — Embedding layer for Shulchan Arukh RAG
===================================================
Output paths (controlled by config.yaml → embeddings → outputs):
  1. NPY matrix  (always)          — build_embeddings(...)
  2. ChromaDB    (opt-in)          — store_in_chroma(...)
  3. LanceDB     (opt-in, Shraga)  — store_in_lancedb(...)
  4. Qdrant      (opt-in, Shraga)  — store_in_qdrant(...)
  5. Milvus Lite (opt-in, Shraga)  — store_in_milvus(...)

All paths share the same passage-text formula (prefix + enrich_fields),
so a query embedded via encode_query() is compatible with any index.

Input: CSV (chunker output) or JSON (chunker/builders output):
    CSV columns:  siman, seif, text
    JSON fields:  id, siman, seif, siman_seif, text  (+ any enrich_fields)

Public API:
    build_embeddings(csv, npy, model, ...)     → writes NPY (+ any enabled DB)
    encode_query(query, model, ...)            → np.ndarray (1D, normalized)
    load_chunks(path)                          → list[dict]        (CSV or JSON)
    store_in_chroma(chunks, vectors, ...)      → writes ChromaDB
    store_in_lancedb(chunks, vectors, ...)     → writes LanceDB    (Shraga)
    store_in_qdrant(chunks, vectors, ...)      → writes Qdrant     (Shraga)
    store_in_milvus(chunks, vectors, ...)      → writes Milvus Lite (Shraga)
    embed(model, texts)                        → list[list[float]] (legacy helper)

CLI (legacy Chroma path):
    python embed.py --chunks path/to/chunks.csv
    python embed.py --chunks path/to/chunks.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
# chromadb is imported lazily inside store_in_chroma — it's only needed for the
# legacy Chroma CLI path, not for the NPY pipeline used by exp_main.py.

# ─── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "intfloat/multilingual-e5-large"
DEFAULT_CHROMA_DIR = Path(__file__).parent / "chroma_db"
DEFAULT_COLLECTION = "shulchan_arukh_seifs"
BATCH_SIZE         = 32


# ─── Shared helpers ────────────────────────────────────────────────────────────

def load_chunks(path: Path) -> list[dict]:
    """
    Load chunks from CSV (chunker output) or JSON (chunker/builders output).
    CSV requires at minimum: siman, seif, text
    JSON must be a list of dicts with at minimum: siman, seif, text
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
        required = {"siman", "seif", "text"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing columns in CSV: {missing}. "
                f"Expected at least {sorted(required)}."
            )
        records = df.to_dict("records")
    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r} — use .csv or .json")
    print(f"  {len(records)} chunks loaded from {path.name}")
    return records


def _merge_caches(
    chunks: list[dict],
    enrich_fields: list[str],
    enrich_caches: dict[str, str],
) -> list[dict]:
    """
    For each field in enrich_fields that has a matching entry in enrich_caches,
    load the cache JSON and add the field to each chunk.

    Cache key format: "siman_seif"  e.g. "1_1", "128_3"
    Cache files come from config.yaml → embeddings → enrich_caches.
    Missing keys get "" for string fields or [] for "questions".
    Modifies chunks in-place and returns them.
    """
    for field in enrich_fields:
        cache_path_str = enrich_caches.get(field)
        if not cache_path_str:
            continue
        cache_path = Path(cache_path_str)
        if not cache_path.exists():
            print(f"  [warn] cache not found: {cache_path} — skipping '{field}'")
            continue
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"  Loaded cache '{field}': {cache_path.name} ({len(cache)} entries)")
        for row in chunks:
            key = f"{int(row['siman'])}_{int(row['seif'])}"
            row[field] = cache.get(key, [] if field == "questions" else "")
    return chunks


def _build_passage_texts(
    chunks: list[dict],
    prefix_passage: str,
    enrich_fields: list[str] | None = None,
    enrich_separator: str = " | ",
) -> list[str]:
    """
    Build the passage text for each chunk.

    Without enrich_fields:  "<prefix_passage><text>"
    With enrich_fields:     "<prefix_passage><text> | <field1> | <field2> ..."

    enrich_fields comes from config.yaml → embeddings → enrich_fields.
    Fields that are lists (e.g. questions) are joined with spaces.
    Missing or empty fields are silently skipped.
    """
    texts = []
    for row in chunks:
        base = prefix_passage + row["text"]
        if not enrich_fields:
            texts.append(base)
            continue
        extras = []
        for field in enrich_fields:
            val = row.get(field, "")
            if not val:
                continue
            if isinstance(val, list):
                val = " ".join(str(v) for v in val if v)
            extras.append(str(val).strip())
        texts.append(base + enrich_separator + enrich_separator.join(extras) if extras else base)
    return texts


# ─── Public API — NPY path (used by exp_main orchestrator) ─────────────────────

def build_embeddings(
    csv:              str | Path,
    npy:              str | Path,
    model:            str = DEFAULT_MODEL,
    batch_size:       int = BATCH_SIZE,
    prefix_passage:   str = "passage: ",
    enrich_fields:    list[str] | None = None,
    enrich_separator: str = " | ",
    enrich_caches:    dict[str, str] | None = None,
    outputs_cfg:      dict | None = None,          # Shraga: multi-DB output config
) -> Path:
    """
    Pipeline entry point: chunks file (CSV or JSON) → embeddings NPY (+ optional vector DBs).

    Loads chunks, optionally merges enrichment caches (modern_summary, questions, etc.),
    builds passage texts, encodes, saves .npy, then stores in any additional DBs enabled
    in outputs_cfg (ChromaDB, LanceDB, Qdrant, Milvus Lite).
    Creates parent directories if needed. Returns the NPY path.

    enrich_fields:  fields to append to passage text, e.g. ["modern_summary", "questions"]
    enrich_caches:  {field: absolute_path_to_cache_json} — resolved by exp_main from config
    outputs_cfg:    config.yaml → embeddings → outputs (paths already absolute, resolved by exp_main)
    All come from config.yaml → embeddings. Empty enrich_fields = baseline (no enrichment).
    """
    print(f"  Loading chunks from {Path(csv).name}")
    chunks = load_chunks(Path(csv))

    if enrich_fields and enrich_caches:
        print(f"  Merging caches for: {enrich_fields}")
        chunks = _merge_caches(chunks, enrich_fields, enrich_caches)

    print(f"  Building passage texts (prefix={prefix_passage!r}, enrich={enrich_fields})")
    texts = _build_passage_texts(chunks, prefix_passage, enrich_fields, enrich_separator)

    print(f"  Loading model: {model}")
    m = SentenceTransformer(model)
    print(f"  Vector dim: {m.get_sentence_embedding_dimension()}")

    print(f"  Encoding {len(texts)} passages (batch_size={batch_size})...")
    t0 = time.time()
    vectors = m.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s — shape {vectors.shape}")

    npy = Path(npy)
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy), vectors)
    print(f"  Saved embeddings to {npy}")

    # Shraga: save to additional vector DBs if configured
    if outputs_cfg:
        chroma_cfg = outputs_cfg.get("chroma") or {}
        if chroma_cfg.get("enabled"):
            store_in_chroma(
                chunks, vectors.tolist(),
                chroma_dir=Path(chroma_cfg["dir"]),
                collection_name=chroma_cfg.get("collection", DEFAULT_COLLECTION),
            )

        lance_cfg = outputs_cfg.get("lancedb") or {}
        if lance_cfg.get("enabled"):
            store_in_lancedb(
                chunks, vectors,
                db_dir=Path(lance_cfg["dir"]),
                table_name=lance_cfg.get("table", DEFAULT_COLLECTION),
            )

        qdrant_cfg = outputs_cfg.get("qdrant") or {}
        if qdrant_cfg.get("enabled"):
            store_in_qdrant(
                chunks, vectors,
                path=Path(qdrant_cfg["path"]),
                collection_name=qdrant_cfg.get("collection", DEFAULT_COLLECTION),
            )

        milvus_cfg = outputs_cfg.get("milvus") or {}
        if milvus_cfg.get("enabled"):
            store_in_milvus(
                chunks, vectors,
                db_path=Path(milvus_cfg["path"]),
                collection_name=milvus_cfg.get("collection", DEFAULT_COLLECTION),
            )

    return npy


def encode_query(
    query:        str,
    model:        str | SentenceTransformer = DEFAULT_MODEL,
    prefix_query: str = "query: ",
) -> np.ndarray:
    """
    Encode a single query into a normalized 1D vector.
    Accepts either a model name (loads it) or an already-loaded SentenceTransformer.
    """
    m = model if isinstance(model, SentenceTransformer) else SentenceTransformer(model)
    text = prefix_query + query
    return m.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


# ─── Legacy Chroma path (kept for backwards compatibility) ─────────────────────

def embed(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Encode all texts into normalized float32 vectors."""
    print(f"  Encoding {len(texts)} seifs (batch_size={BATCH_SIZE})...")
    t0 = time.time()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s")
    return vectors.tolist()


def store_in_chroma(
    chunks: list[dict],
    vectors: list[list[float]],
    chroma_dir: Path,
    collection_name: str,
) -> None:
    """Store embeddings + metadata in ChromaDB."""
    import chromadb  # lazy — only needed for this legacy path

    client = chromadb.PersistentClient(path=str(chroma_dir))

    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"  Deleted existing collection: {collection_name}")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[f"siman_{row['siman']}_seif_{row['seif']}" for row in chunks],
        embeddings=vectors,
        documents=[row["text"] for row in chunks],
        metadatas=[
            {
                "siman":      int(row["siman"]),
                "seif":       int(row["seif"]),
                "siman_seif": f"{int(row['siman'])}:{int(row['seif'])}",  # built on-the-fly
            }
            for row in chunks
        ],
    )
    print(f"  Stored {collection.count()} seifs in collection '{collection_name}'")
    print(f"  ChromaDB path: {chroma_dir}")


# ─── Shraga: additional vector-DB store functions ──────────────────────────────

def store_in_lancedb(
    chunks: list[dict],
    vectors: np.ndarray,
    db_dir: str | Path,
    table_name: str,
) -> None:
    """Store embeddings + metadata in LanceDB (local directory)."""
    import lancedb  # lazy — only needed when lancedb output is enabled

    Path(db_dir).mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_dir))
    data = [
        {
            "id":     f"siman_{row['siman']}_seif_{row['seif']}",
            "siman":  int(row["siman"]),
            "seif":   int(row["seif"]),
            "text":   row["text"],
            "vector": vectors[i].tolist(),
        }
        for i, row in enumerate(chunks)
    ]
    db.create_table(table_name, data=data, mode="overwrite")
    print(f"  LanceDB: stored {len(data)} rows in table '{table_name}' → {db_dir}")


def store_in_qdrant(
    chunks: list[dict],
    vectors: np.ndarray,
    path: str | Path,
    collection_name: str,
) -> None:
    """Store embeddings + metadata in Qdrant (local on-disk mode)."""
    from qdrant_client import QdrantClient           # lazy — only needed when qdrant output is enabled
    from qdrant_client.models import Distance, VectorParams, PointStruct

    Path(path).mkdir(parents=True, exist_ok=True)
    dim    = vectors.shape[1]
    client = QdrantClient(path=str(path))
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "siman": int(row["siman"]),
                "seif":  int(row["seif"]),
                "text":  row["text"],
            },
        )
        for i, row in enumerate(chunks)
    ]
    client.upsert(collection_name=collection_name, points=points)
    print(f"  Qdrant: stored {len(points)} points in '{collection_name}' → {path}")


def store_in_milvus(
    chunks: list[dict],
    vectors: np.ndarray,
    db_path: str | Path,
    collection_name: str,
) -> None:
    """Store embeddings + metadata in Milvus Lite (local .db file via pymilvus MilvusClient)."""
    from pymilvus import MilvusClient  # lazy — only needed when milvus output is enabled

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    dim    = vectors.shape[1]
    client = MilvusClient(str(db_path))
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(collection_name, dimension=dim)
    data = [
        {
            "id":     i,
            "vector": vectors[i].tolist(),
            "siman":  int(row["siman"]),
            "seif":   int(row["seif"]),
            "text":   row["text"],
        }
        for i, row in enumerate(chunks)
    ]
    client.insert(collection_name, data)
    print(f"  Milvus Lite: stored {len(data)} rows in '{collection_name}' → {db_path}")


def main():
    parser = argparse.ArgumentParser(description="Embed Shulchan Arukh seifs into ChromaDB")
    parser.add_argument("--chunks",     required=True,                   help="Path to chunks.csv (chunker output)")
    parser.add_argument("--model",      default=DEFAULT_MODEL,           help=f"Embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR), help="ChromaDB directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,      help="ChromaDB collection name")
    parser.add_argument("--prefix-passage", default="passage: ",         help="Prefix used for passage encoding")
    args = parser.parse_args()

    csv_path   = Path(args.chunks)
    chroma_dir = Path(args.chroma_dir)

    print(f"\n1. Loading chunks...")
    chunks = load_chunks(csv_path)

    print(f"\n2. Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    print(f"   Vector dim: {model.get_sentence_embedding_dimension()}")

    print(f"\n3. Building encoding texts...")
    texts = _build_passage_texts(chunks, args.prefix_passage)

    print(f"\n4. Embedding...")
    vectors = embed(model, texts)

    print(f"\n5. Storing in ChromaDB...")
    store_in_chroma(chunks, vectors, chroma_dir, args.collection)

    print(f"\nDone. To query:\n"
          f"  client = chromadb.PersistentClient('{chroma_dir}')\n"
          f"  col = client.get_collection('{args.collection}')\n"
          f"  col.query(query_embeddings=[...], n_results=10)")


if __name__ == "__main__":
    main()
