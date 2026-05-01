import csv
import json
from pathlib import Path

EVAL_CSV = Path(__file__).resolve().parents[2] / "data" / "eval" / "sa_eval.csv"
_cache = None


def _load():
    global _cache
    if _cache is not None:
        return _cache
    rows = []
    with open(EVAL_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append({
                "id":       row.get("#", ""),
                "question": row.get("שאלה", ""),
                "answer":   row.get("תשובה", ""),
                "siman":    row.get("סימן", ""),
                "seif":     row.get("סעיף", ""),
            })
    _cache = rows
    return _cache


class handler:
    def do_GET(self):
        data = _load()
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
