"""
fix_seifs_with_sefaria.py
=========================
מתקן את חלוקת הסעיפים בקובץ המקור המקומי לפי Sefaria API.

לכל סימן שספירת הסעיפים שלו שונה מ-Sefaria:
  - מאחד את כל השורות של הסימן לטקסט אחד
  - מוצא את גבולות הסעיפים לפי הסתיים של כל סעיף ב-Sefaria
  - מפצל את הטקסט המקומי בהתאם

הקובץ המקורי נשמר כ-<שם>_backup.txt, ולאחר מכן נדרס.
"""
import requests, time, json, re, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))
from config import TEXT_FILE

CACHE    = Path(__file__).parent / "data" / "sefaria_cache.json"
BASE_URL = "https://www.sefaria.org/api/texts/Shulchan_Arukh,_Orach_Chayim.{n}?lang=he"
SLEEP    = 0.5


# ─── Sefaria ──────────────────────────────────────────────────────────────────

def strip_html(text: str) -> str:
    """מסיר תגי HTML ומנרמל רווחים."""
    text = re.sub(r'<[^>]+>', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_content(text: str) -> str:
    """
    מטפל בסעיפים שמכילים כותרת פרק + תוכן:
    'דין TITLE. ובו X סעיפים:CONTENT' → מחזיר 'CONTENT' בלבד.
    אם הכותרת ריקה מתוכן — מחזיר '' (יסונן).
    אם אין כותרת — מחזיר את הטקסט כמו שהוא.
    """
    m = re.search(r'ובו\s+[א-תa-z\d]+\s+סעיפים[:\s]*(.*)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text

def fetch_siman(n: int, cache: dict) -> list[str]:
    key = str(n)
    if key in cache:
        return cache[key]
    resp = requests.get(BASE_URL.format(n=n), timeout=10)
    resp.raise_for_status()
    raw_seifs = resp.json().get("he", [])
    seifs = []
    for s in raw_seifs:
        if not isinstance(s, str):
            continue
        clean = strip_html(s)
        if not clean:
            continue
        clean = extract_content(clean)   # הסר כותרת אם קיימת
        if clean:
            seifs.append(clean)
    cache[key] = seifs
    time.sleep(SLEEP)
    return seifs


# ─── Normalization ────────────────────────────────────────────────────────────

def norm(text: str) -> str:
    """מסיר ניקוד, פיסוק ותווים שאינם עבריים/ספרות — לצורך השוואת טקסטים."""
    text = re.sub(r'[\u0591-\u05C7]', '', text)          # strip niqqud
    text = re.sub(r'[^\u05D0-\u05EA0-9\s]', ' ', text)   # keep Hebrew + digits
    return re.sub(r'\s+', ' ', text).strip()

def words_match(w1: str, w2: str) -> bool:
    """
    בודק אם שתי מילים הן אותה מילה בכתיב שונה (כתיב חסר/מלא).
    קריטריון: הפרש אורך ≤ 2, תו ראשון זהה, 2 תווים אחרונים זהים.
    מכסה: 'תפילין'↔'תפלין', 'תיקון'↔'תקון', 'שהוחזקו'↔'שהחזקו'.
    """
    if w1 == w2:
        return True
    if abs(len(w1) - len(w2)) > 2:
        return False
    return (len(w1) >= 3 and len(w2) >= 3
            and w1[0] == w2[0]
            and w1[-2:] == w2[-2:])


# ─── Seif splitting ───────────────────────────────────────────────────────────

def split_by_sefaria(local_lines: list[str], sefaria_seifs: list[str]) -> list[str]:
    """
    מחלק את local_lines ל-len(sefaria_seifs) סעיפים לפי גבולות Sefaria.

    שיטה: חלוקה לפי ספירת מילים בלבד.
    אם הטקסט המקומי ו-Sefaria מכילים אותו תוכן (מהדורות שונות),
    מספר המילים לכל סעיף זהה — אין צורך בהשוואה.
    """
    local_words = ' '.join(local_lines).split()

    result  = []
    loc_pos = 0

    for seif in sefaria_seifs[:-1]:
        count   = len(norm(seif).split())
        end     = min(loc_pos + count, len(local_words))
        result.append(' '.join(local_words[loc_pos:end]))
        loc_pos = end

    result.append(' '.join(local_words[loc_pos:]))
    return [r for r in result if r.strip()]


# ─── Parse local file ─────────────────────────────────────────────────────────

def parse_local_file(path: Path) -> tuple[list[str], dict[int, list[str]]]:
    """
    מחזיר (header_lines, simanim_dict).
    header_lines — שורות לפני הסימן הראשון.
    simanim_dict — {siman_num: [שורת_סעיף, ...]}  (שורות תוכן בלבד, ללא ריקות).
    """
    lines   = path.read_text(encoding='utf-8').splitlines()
    header  = []
    simanim: dict[int, list[str]] = {}
    current = None

    for line in lines:
        stripped = line.strip()
        if re.match(r'^Siman \d+\s*$', stripped):
            current = int(stripped.split()[1])
            simanim[current] = []
        elif current is None:
            header.append(line)
        elif stripped and norm(stripped):   # מסנן שורות HTML-בלבד (ללא תוכן עברי)
            simanim[current].append(stripped)

    return header, simanim


# ─── Reconstruct file ─────────────────────────────────────────────────────────

def reconstruct(header: list[str], simanim: dict[int, list[str]]) -> str:
    parts = list(header)
    for siman_num in sorted(simanim):
        parts.append('')
        parts.append(f'Siman {siman_num}')
        parts.append('')
        for seif in simanim[siman_num]:
            parts.append(seif)
            parts.append('')
    return '\n'.join(parts)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cache = json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}

    print(f"קורא: {TEXT_FILE.name}")
    header, simanim = parse_local_file(TEXT_FILE)
    print(f"  סימנים: {len(simanim)},  שורות header: {len(header)}")
    print(f"  סעיפים לפני תיקון: {sum(len(v) for v in simanim.values()):,}")

    fixed_count   = 0
    missing_count = 0

    for siman_num in sorted(simanim):
        local_seifs   = simanim[siman_num]
        sefaria_seifs = fetch_siman(siman_num, cache)

        if not sefaria_seifs:
            print(f"  [דלג] סימן {siman_num}: אין ב-Sefaria")
            missing_count += 1
            continue

        if len(local_seifs) == len(sefaria_seifs):
            continue  # תקין — אין שינוי

        print(f"  סימן {siman_num}: מקומי={len(local_seifs)} | Sefaria={len(sefaria_seifs)} → מתקן")
        simanim[siman_num] = split_by_sefaria(local_seifs, sefaria_seifs)
        fixed_count += 1

    # שמירת cache
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\nCache: {len(cache)} סימנים")

    # גיבוי + דריסה
    backup = TEXT_FILE.parent / (TEXT_FILE.stem + '_backup' + TEXT_FILE.suffix)
    backup.write_bytes(TEXT_FILE.read_bytes())
    print(f"גיבוי: {backup.name}")

    TEXT_FILE.write_text(reconstruct(header, simanim), encoding='utf-8')
    print(f"נשמר:  {TEXT_FILE.name}")

    # סטטיסטיקות
    total_seifs = sum(len(v) for v in simanim.values())
    print(f"\nסימנים שתוקנו:       {fixed_count}")
    if missing_count:
        print(f"סימנים חסרים ב-Sefaria: {missing_count}")
    print(f"סה\"כ סעיפים אחרי תיקון: {total_seifs:,}")


if __name__ == '__main__':
    main()
