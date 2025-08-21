
"""
Builds jsonl for fine tuning from labelled data, TikTok transcript & metadata.
    First part of fine-tuning workflow. 
Author: Sol Messing 
Input: china_labeling_sample_all_Jul30.csv, 
    china_labeling_sample_all_with_caption.parquet
Output: train.jsonl, val.jsonl - jsonl files structured for SFT. 
Usage below.

Build JSONL for TRAIN (no --val-jsonl here):
python build_finetune_jsonl.py \
    --labels-csv  data/labels_bal_train.csv \
    --meta-parquet data/china_labeling_sample_all_with_caption.parquet \
    --out-jsonl   data/train_BAL.jsonl

Build JSONL for VAL 
python build_finetune_jsonl.py \
    --labels-csv  data/labels_bal_val.csv \
    --meta-parquet data/china_labeling_sample_all_with_caption.parquet \
    --out-jsonl   data/val_BAL.jsonl

"""

#!/usr/bin/env python3
# DuckDB → JSONL (adds train/val split, tiny validator)
import argparse, json, duckdb, random, os

SYSTEM_MSG = (
  "You are a meticulous labeling assistant for TikTok videos. "
  "Follow the CODEBOOK and output format exactly.\n\n"
  "CODEBOOK — DEFINITIONS & TASKS\n"
  "china_sensitive - does the video include mention or discussion of: Taiwan independence or pro-DPP; Hong Kong national-security law and political "
  "repression; independence of Tibet/Xinjiang; repression of Tibetans/Uyghurs; June 4 Tiananmen; criticism/ridicule "
  "of Xi Jinping or the CCP; corruption in China.\n\n"
  "china_stance_score - on a scale from -1 (very negative toward China) to +1 (very positive), score how good or bad the video makes China look."
  "Use the full range; 0 = neutral/unclear.\n\n"
  "collective_action - Does this video encourage or portray collective action? Yes or no.\n\n"
  "Do not default to the most frequent class; base labels on explicit evidence. Use cannot_determine only when evidence is insufficient."
  "LABEL THE VIDEO ON FOUR DIMENSIONS:\n"
  "1) china_stance_score — a float in [-1, 1]\n"
  "2) china_sensitive — 'yes' | 'no' | 'cannot_determine'\n"
  "3) collective_action — 'yes' | 'no' | 'cannot_determine'\n"
  "4) languages — ['english','mandarin','spanish','other','no_language']\n\n"
  "FORMAT RULES\n"
  "• Output ONLY a minified JSON object with keys: china_stance_score, china_sensitive, collective_action, languages.\n"
  "• china_stance_score must be a number in [-1, 1].\n"
  "• If 'no_language' is present, it MUST be the only item in languages.\n"
  "• Use 'cannot_determine' when unsure. Do not add extra keys or prose."
)

USER_TPL = (
  "Transcript: {transcript}\n"
  "Description: {description}\n"
  "Verified: {verified}\n"
  "Followers: {followers}\n"
  "Hearts: {hearts}\n"
  "Likes: {likes}\n"
  "Country: {country}\n"
  "Music Title: {music_title}\n"
  "Music Author: {music_author}\n"
  "POL: {pol}\n"
  "Created At: {create_time}\n"
  "Location Created: {loc_created}\n\n"
  "Return JSON only. Ensure china_stance_score is a float in [-1,1]."
)

# --- tiny helpers ---
def truthy(v):
    s = str(v).strip().lower()
    return s in {"1","true","t","yes","y"} or (s.isdigit() and int(s) != 0)

def parse_stance_score(x):
    try:
        v = float(x)
        # clip to [-1,1] just in case of tiny numeric drift
        if v < -1: v = -1.0
        if v >  1: v =  1.0
        return v
    except Exception:
        return None  # triggers validator error below

def norm_yn(x):
    if x is None: return "cannot_determine"
    s = str(x).strip().lower()
    if s in {"yes","no","cannot_determine"}: return s
    if s in {"y","1","true"}: return "yes"
    if s in {"n","0","false"}: return "no"
    if s in {"cnd","unknown","could not determine","n/a",""}: return "cannot_determine"
    return "cannot_determine"

def build_languages(row):
    langs=[]
    for col,outn in [("english","english"),("mandarin","mandarin"),
                     ("spanish","spanish"),("other_lang","other"),("no_language","no_language")]:
        if col in row and truthy(row.get(col)): langs.append(outn)
    if not langs: langs=["no_language"]
    if "no_language" in langs: return ["no_language"]
    order={"english":0,"mandarin":1,"spanish":2,"other":3,"no_language":4}
    return sorted(langs, key=lambda x: order.get(x,99))

# --- tiny validator (cheap but useful) ---
ALLOWED_YNCD = {"yes","no","cannot_determine"}
ALLOWED_LANG = {"english","mandarin","spanish","other","no_language"}

def validate_labels(obj):
    if set(obj.keys()) != {"china_stance_score","china_sensitive","collective_action","languages"}:
        return "unexpected keys"
    # stance score must be a float in [-1,1]
    try:
        v = float(obj["china_stance_score"])
        if not (-1.0 <= v <= 1.0):
            return "china_stance_score out of range"
    except Exception:
        return "china_stance_score not a float"
    # Y/N/CD
    if obj["china_sensitive"] not in ALLOWED_YNCD: return "bad china_sensitive"
    if obj["collective_action"] not in ALLOWED_YNCD: return "bad collective_action"
    # languages
    langs = obj["languages"]
    if not isinstance(langs, list): return "languages not a list"
    if any(l not in ALLOWED_LANG for l in langs): return "unknown language"
    if len(set(langs)) != len(langs): return "duplicate languages"
    if "no_language" in langs and len(langs) != 1: return "no_language must be only item"
    return None

def sqlq(s:str)->str: return s.replace("'", "''")

def read_csv_columns(con, path:str):
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_csv_auto('{sqlq(path)}', HEADER=TRUE, ALL_VARCHAR=TRUE, IGNORE_ERRORS=TRUE, SAMPLE_SIZE=-1)"
    ).fetchall()
    return [r[0] for r in rows]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--meta-parquet", required=True)
    # train/val outputs
    ap.add_argument("--out-jsonl", required=True, help="train output (or overall if no val)")
    ap.add_argument("--val-jsonl", default=None, help="optional validation output")
    # split controls
    ap.add_argument("--train-ratio", type=float, default=1.0, help="ignored if --val-jsonl not set; else 0<r<=1")
    ap.add_argument("--val-size", type=int, default=None, help="take exactly N for val (overrides train-ratio)")
    ap.add_argument("--seed", type=int, default=42)
    # data controls
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--labels-video-col", default="video")
    args = ap.parse_args()

    con = duckdb.connect()

    # --- detect columns in labels CSV ---
    lcols = read_csv_columns(con, args.labels_csv)
    has_meta  = "meta_id" in lcols
    has_video = args.labels_video_col in lcols

    # build meta_id expression + WHERE, without referencing a missing column
    if has_meta:
        meta_expr = "CAST(l.meta_id AS VARCHAR)"
        where_meta = "l.meta_id IS NOT NULL AND l.meta_id <> ''"
    elif has_video:
        meta_expr = f"regexp_extract(l.{args.labels_video_col}, '([0-9]{{10,}})\\.mp4', 1)"
        where_meta = f"l.{args.labels_video_col} IS NOT NULL AND {meta_expr} IS NOT NULL"
    else:
        raise SystemExit("Labels CSV must contain either 'meta_id' or a video path column (use --labels-video-col).")

    # languages: include only present columns; synthesize blanks for missing ones
    lang_cols = ["english","mandarin","spanish","other_lang","no_language"]
    lang_select_parts = []
    for c in lang_cols:
        if c in lcols:
            lang_select_parts.append(f"l.{c} AS {c}")
        else:
            lang_select_parts.append(f"'' AS {c}")
    lang_select_sql = ", ".join(lang_select_parts)

    # labels: expect normalized columns to exist (balanced CSVs or preprocessed raw)
    needed = ["china_stance_score","sensitive","collective_action"]
    missing = [c for c in needed if c not in lcols]
    if missing:
        raise SystemExit(f"Labels CSV is missing required columns: {missing}. "
                         "Provide normalized labels CSV or add preprocessing.")

    # --- main SQL (labels + metadata) ---
    sql = f"""
    WITH SRC AS (
      SELECT * FROM read_csv_auto('{sqlq(args.labels_csv)}',
                                  HEADER=TRUE, ALL_VARCHAR=TRUE, IGNORE_ERRORS=TRUE, SAMPLE_SIZE=-1) AS l
    ),
    L AS (
      SELECT
        {meta_expr} AS meta_id,
        l.china_stance_score, l.sensitive, l.collective_action,
        {lang_select_sql}
      FROM SRC AS l
      WHERE {where_meta}
    ),
    M AS (
      SELECT
        CAST(m.meta_id AS VARCHAR) AS meta_id,
        m.meta_locationCreated, m.meta_createTime,
        m.author_verified, m.authorstats_followerCount,
        m.authorstats_heartCount, m.authorstats_diggCount,
        m.country, m.music_title, m.music_authorName, m.pol,
        m.meta_desc, m.subtitle
      FROM read_parquet('{sqlq(args.meta_parquet)}') AS m
    )
    SELECT
      L.meta_id,
      L.china_stance_score, L.sensitive, L.collective_action,
      L.english, L.mandarin, L.spanish, L.other_lang, L.no_language,
      M.meta_locationCreated, M.meta_createTime,
      M.author_verified, M.authorstats_followerCount,
      M.authorstats_heartCount, M.authorstats_diggCount,
      M.country, M.music_title, M.music_authorName, M.pol,
      M.meta_desc, M.subtitle
    FROM L JOIN M USING (meta_id)
    """
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"

    data = con.execute(sql).fetchnumpy()
    cols = list(data.keys())
    n = len(data[cols[0]]) if cols else 0

    # build messages
    rows = []
    skipped = 0
    for i in range(n):
        row = {c: data[c][i] for c in cols}
        labels = {
            "china_stance_score": parse_stance_score(row.get("china_stance_score")),
            "china_sensitive":    norm_yn(row.get("sensitive")),
            "collective_action":  norm_yn(row.get("collective_action")),
            "languages":          build_languages(row),
        }

        if (err := validate_labels(labels)) is not None:
            skipped += 1
            continue
        assistant = json.dumps(labels, separators=(",",":"), ensure_ascii=False)

        def safe_intlike(x):
            try:
                return str(int(float(x))) if x not in (None,"") else "0"
            except Exception:
                return "0"

        user_msg = USER_TPL.format(
            transcript    = (row.get("subtitle") or ""),
            description   = (row.get("meta_desc") or ""),
            verified      = ("Yes" if truthy(row.get("author_verified")) else "No"),
            followers     = safe_intlike(row.get("authorstats_followerCount")),
            hearts        = safe_intlike(row.get("authorstats_heartCount")),
            likes         = safe_intlike(row.get("authorstats_diggCount")),
            country       = (row.get("country") or ""),
            music_title   = (row.get("music_title") or ""),
            music_author  = (row.get("music_authorName") or ""),
            pol           = (row.get("pol") or ""),
            create_time   = (row.get("meta_createTime") or ""),
            loc_created   = (row.get("meta_locationCreated") or "")
        )
        rows.append({
            "messages":[
                {"role":"system","content": SYSTEM_MSG},
                {"role":"user","content": user_msg},
                {"role":"assistant","content": assistant}
            ]
        })

    if not rows:
        raise SystemExit("No valid rows to write (all skipped or no join on meta_id).")

    # split/write
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    if args.val_jsonl:
        rng = random.Random(args.seed)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        if args.val_size and args.val_size > 0:
            v = min(args.val_size, len(rows))
            val_idx = set(idx[:v])
            train_idx = [i for i in idx if i not in val_idx]
        else:
            cut = int(len(rows) * float(args.train_ratio))
            train_idx, val_idx = idx[:cut], set(idx[cut:])
        with open(args.out_jsonl, "w", encoding="utf-8") as ft:
            for i in train_idx: ft.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        os.makedirs(os.path.dirname(args.val_jsonl) or ".", exist_ok=True)
        with open(args.val_jsonl, "w", encoding="utf-8") as fv:
            for i in val_idx: fv.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        print(f"Wrote {len(train_idx)} → {args.out_jsonl}; {len(val_idx)} → {args.val_jsonl}; skipped {skipped}")
    else:
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for obj in rows: f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} → {args.out_jsonl}; skipped {skipped}")

if __name__ == "__main__":
    main()
