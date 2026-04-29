"""
build_source_from_sefaria.py
============================
מוריד את השולחן ערוך אורח חיים מ-Sefaria API ובונה קובץ טקסט
עם חלוקה נכונה לסעיפים.

שימוש:
  python build_source_from_sefaria.py              # כל 697 סימנים
  python build_source_from_sefaria.py --test 5     # 5 סימנים ראשונים בלבד
  python build_source_from_sefaria.py --resume 100 # המשך מסימן 100
"""
import requests, time, json, sys, io, argparse, datetime
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def make_output_path() -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).parent / "data" / f"Shulchan_Arukh_OC_sefaria_{ts}.txt"
CACHE    = Path(__file__).parent / "data" / "sefaria_cache.json"
BASE_URL = "https://www.sefaria.org/api/texts/Shulchan_Arukh,_Orach_Chayim.{n}?lang=he"
TOTAL_SIMANIM = 697
SLEEP = 0.5  # שניות בין קריאות API


def fetch_siman(n: int, cache: dict) -> list[str]:
    """מחזיר list של טקסטים לכל סעיף בסימן n."""
    key = str(n)
    if key in cache:
        return cache[key]

    url = BASE_URL.format(n=n)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    seifs = data.get("he", [])
    # סנן סעיפים ריקים
    seifs = [s.strip() for s in seifs if isinstance(s, str) and s.strip()]
    cache[key] = seifs
    time.sleep(SLEEP)
    return seifs


def build_file(simanim_range: range, cache: dict, output: Path) -> None:
    lines = []
    lines.append("Shulchan Arukh, Orach Chayim")
    lines.append("Source: Sefaria (sefaria.org)")
    lines.append("")

    total_seifs = 0
    empty_simanim = []

    for n in simanim_range:
        seifs = fetch_siman(n, cache)
        if not seifs:
            print(f"  [אזהרה] סימן {n}: 0 סעיפים")
            empty_simanim.append(n)
            continue

        lines.append(f"Siman {n}")
        lines.append("")
        for seif_text in seifs:
            lines.append(seif_text)
            lines.append("")

        total_seifs += len(seifs)
        print(f"  סימן {n}: {len(seifs)} סעיפים")

    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nנשמר: {output}")
    print(f"סה\"כ סימנים: {len(simanim_range) - len(empty_simanim)}")
    print(f"סה\"כ סעיפים: {total_seifs:,}")
    if empty_simanim:
        print(f"סימנים ריקים: {empty_simanim}")


def main():
    parser = argparse.ArgumentParser(
        description="מוריד שולחן ערוך אורח חיים מ-Sefaria API"
    )
    parser.add_argument("--test",   type=int, default=None,
                        help="הורד N סימנים ראשונים בלבד (לבדיקה)")
    parser.add_argument("--resume", type=int, default=1,
                        help="התחל מסימן N (ברירת מחדל: 1)")
    args = parser.parse_args()

    # טען cache אם קיים
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    print(f"Cache קיים: {len(cache)} סימנים שמורים")

    end = (args.resume + args.test - 1) if args.test else TOTAL_SIMANIM
    r = range(args.resume, end + 1)

    print(f"מוריד סימנים {r.start}–{r.stop - 1}...")
    try:
        output = make_output_path()
        build_file(r, cache, output)
    finally:
        # שמור cache גם אם נקטע באמצע
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Cache נשמר: {len(cache)} סימנים")


if __name__ == "__main__":
    main()
