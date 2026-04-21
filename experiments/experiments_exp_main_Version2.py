"""
exp_main.py — RAG experiment pipeline (YAML config)
====================================================
טוען הגדרות מ-exp_config.yaml ומריץ את כל שלבי הצינור:
  1. chunking  — בניית DataFrame מה-JSON
  2. embedding — קידוד הצ'אנקים עם multilingual-e5-small
  3. retrieval — שליפת top-k לכל שאלה
  4. evaluation — Recall@k לפי השוואת סימן/סעיף

Usage:
    python exp_main.py
    python exp_main.py --chunks chunks_v1.csv --topk 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chunker import load_schema, build_dataframe

# ─── Load config ───────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
CONFIG_PATH = HERE / "experiments_exp_config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Paths
JSON_FILE = (HERE / cfg["paths"]["json_file"]).resolve()
CSV_PATH  = (HERE / cfg["paths"]["csv_path"]).resolve()

# Param dicts
chunk_params     = cfg["chunker"]
embed_params     = cfg["embeddings"]
retrieval_params = cfg["retrieval"]
eval_params      = cfg["evaluation"]


# ─── Pipeline helpers ──────────────────────────────────────────────────────────

def load_queries(csv_path: Path) -> pd.DataFrame:
    """
    טוען את ה-CSV עם השאלות וה-ground truth (סימן, סעיף).

    מנרמל שמות עמודות לאנגלית:
        שאלה / question / query  →  question
        סימן  / siman            →  siman
        סעיף  / seif             →  seif
    """
    df = pd.read_csv(csv_path)
    col_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc in ("שאלה", "question", "query"):
            col_map[col] = "question"
        elif lc in ("סימן", "siman"):
            col_map[col] = "siman"
        elif lc in ("סעיף", "seif"):
            col_map[col] = "seif"
    df = df.rename(columns=col_map)
    required = {"question", "siman", "seif"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"חסרות עמודות ב-CSV: {missing}")
    return df


def build_embeddings(chunks: pd.DataFrame, model: SentenceTransformer,
                     prefix: str, batch_size: int) -> np.ndarray:
    """מקודד את כל הצ'אנקים ומחזיר מטריצה מנורמלת (N, dim)."""
    texts = [prefix + row["text"] for _, row in chunks.iterrows()]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors  # shape: (N, dim)


def retrieve(query: str, chunks: pd.DataFrame, embeddings: np.ndarray,
             model: SentenceTransformer, top_k: int,
             prefix_query: str = "query: ", **kwargs) -> list[dict]:
    """שולף top_k צ'אנקים לשאילתה לפי cosine similarity."""
    query_vec = model.encode(
        prefix_query + query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    scores = embeddings @ query_vec                          # (N,)
    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]   # ממוין יורד

    results = []
    for idx in top_idx:
        row = chunks.iloc[idx]
        results.append({
            "siman":      int(row["siman"]),
            "seif":       int(row["seif"]),
            "siman_seif": row["siman_seif"],
            "text":       row["text"],
            "score":      float(scores[idx]),
        })
    return results


def retrieve_evaluate(results: list[dict], gt_siman: int, gt_seif: int) -> float:
    """
    Seif-level Recall@k:
    מחזיר 1.0 אם ה-ground truth (סימן, סעיף) נמצא בתוצאות, אחרת 0.0.
    """
    for r in results:
        if r["siman"] == gt_siman and r["seif"] == gt_seif:
            return 1.0
    return 0.0


# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG experiment — YAML config")
    parser.add_argument("--chunks", default=str(HERE / "chunks_v1.csv"),
                        help="נתיב לקובץ CSV של הצ'אנקים")
    parser.add_argument("--topk", type=int, default=retrieval_params["top_k"],
                        help=f"כמות תוצאות להחזיר (ברירת מחדל: {retrieval_params['top_k']})")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)

    print(f"Config:    {CONFIG_PATH}")
    print(f"JSON file: {JSON_FILE}")
    print(f"CSV path:  {CSV_PATH}")
    print(f"Chunks:    {chunks_path}")
    print(f"Model:     {embed_params['model']}")
    print(f"Chunker:   {chunk_params}\n")

    # ── 1. Chunks ──────────────────────────────────────────────────────────────
    if not chunks_path.exists():
        print(f"קובץ צ'אנקים לא נמצא. בונה מ-{JSON_FILE.name}...")
        schema = load_schema(JSON_FILE)
        chunks = build_dataframe(schema, chunk_fields=chunk_params.get("chunk_fields"))
        chunks.to_csv(chunks_path, index=False, encoding="utf-8")
        print(f"  נשמרו {len(chunks)} צ'אנקים -> {chunks_path}")
    else:
        print(f"קובץ צ'אנקים נמצא. טוען מ-{chunks_path.name}...")
        chunks = pd.read_csv(chunks_path)
    print(f"  {len(chunks)} צ'אנקים נטענו.\n")

    # ── 2. Embeddings ──────────────────────────────────────────────────────────
    embeddings_path = chunks_path.with_suffix(".embeddings.npy")

    print(f"טוען מודל: {embed_params['model']}")
    embedding_model = SentenceTransformer(embed_params["model"])

    if not embeddings_path.exists():
        print(f"הטמעות לא נמצאו. מחשב עבור {len(chunks)} צ'אנקים...")
        embeddings = build_embeddings(
            chunks, embedding_model,
            prefix=embed_params["prefix_passage"],
            batch_size=embed_params["batch_size"],
        )
        np.save(str(embeddings_path), embeddings)
        print(f"  נשמרו הטמעות -> {embeddings_path}")
    else:
        print(f"הטמעות נמצאו. טוען מ-{embeddings_path.name}...")
        embeddings = np.load(str(embeddings_path))
    print(f"  צורת מטריצת הטמעות: {embeddings.shape}\n")

    # ── 3. Queries ─────────────────────────────────────────────────────────────
    print(f"טוען שאלות מ-{CSV_PATH.name}...")
    queries_df = load_queries(CSV_PATH)
    print(f"  {len(queries_df)} שאלות נטענו.\n")

    # ── 4. Retrieval + Evaluation ──────────────────────────────────────────────
    retrieve_k = retrieval_params.get("top_k_retrieve", args.topk)
    scores_list = []
    report_lines: list[str] = []
    for q_idx, (_, row) in enumerate(queries_df.iterrows(), start=1):
        query    = str(row["question"])
        gt_siman = int(row["siman"])
        gt_seif  = int(row["seif"])

        print(f"\nשאלה: {query}")
        print(f"  Ground truth: סימן {gt_siman}, סעיף {gt_seif}")

        results = retrieve(
            query, chunks, embeddings, embedding_model,
            top_k=retrieve_k,
            prefix_query=embed_params["prefix_query"],
            **{k: v for k, v in retrieval_params.items()
               if k not in ("top_k", "top_k_retrieve")},
        )

        for i, r in enumerate(results[:args.topk], 1):
            print(f"  תוצאה {i}: {r['siman_seif']} (score: {r['score']:.4f})")

        score = retrieve_evaluate(results[:args.topk], gt_siman, gt_seif)
        scores_list.append(score)
        print(f"  Hit: {'OK' if score == 1.0 else 'MISS'}")

        # rank לפי סימן בלבד (תואם לפורמט הדוח)
        rank = None
        for i, r in enumerate(results, 1):
            if r["siman"] == gt_siman:
                rank = i
                break

        unique_simanim: list[int] = []
        seen: set[int] = set()
        for r in results:
            if r["siman"] not in seen:
                seen.add(r["siman"])
                unique_simanim.append(r["siman"])

        hit_icon = "✅" if rank is not None else "❌"
        top = results[0]
        text_preview = " ".join(str(top["text"]).split())
        if len(text_preview) > 200:
            text_preview = text_preview[:200] + "…"
        retrieval_mode = retrieval_params.get("mode", "flat")

        report_lines.append(
            f"--- Q{q_idx} {hit_icon} rank={rank if rank is not None else '-'} ---"
        )
        report_lines.append(f"שאלה: {query}")
        report_lines.append(f"מקור: סימן {gt_siman}, סעיף {gt_seif}")
        report_lines.append(
            f"שליפה: סימן {top['siman']} [{retrieval_mode}] (score={top['score']:.4f})"
        )
        report_lines.append(
            f"unique_simanim ({len(unique_simanim)}): "
            + ",".join(str(s) for s in unique_simanim)
        )
        report_lines.append(f"טקסט: {text_preview}")
        report_lines.append("")

    # ── 5. Summary ───────────────────────────────────────────────────────────���─
    recall = sum(scores_list) / len(scores_list) if scores_list else 0.0
    print(f"\n{'=' * 50}")
    print(f"Recall@{args.topk}: {recall:.4f}  "
          f"({int(sum(scores_list))}/{len(scores_list)} שאלות)")

    # שמירת תוצאות
    results_path = HERE / "exp_results.json"
    summary = {
        "config":     str(CONFIG_PATH),
        "model":      embed_params["model"],
        "top_k":      args.topk,
        "recall":     recall,
        "n_queries":  len(scores_list),
        "n_hits":     int(sum(scores_list)),
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"תוצאות נשמרו -> {results_path}")

    # שמירת דוח טקסטואלי קריא
    report_path = HERE / "exp_results.txt"
    header = [
        f"Model: {embed_params['model']}",
        f"top_k (eval): {args.topk} | top_k_retrieve: {retrieve_k}",
        f"Queries: {len(scores_list)} | Hits: {int(sum(scores_list))} | "
        f"Recall@{args.topk}: {recall:.4f}",
        "=" * 60,
        "",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header + report_lines))
    print(f"דוח טקסטואלי נשמר -> {report_path}")


if __name__ == "__main__":
    main()