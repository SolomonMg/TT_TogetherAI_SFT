#!/usr/bin/env python3
"""
Deterministic TRAIN/VAL split with per-cell caps (balancing buckets),
while preserving raw numeric labels in outputs.

- Reads labels CSV (DuckDB).
- Extracts meta_id from file path (.../<digits>.mp4 → <digits>).
- Keeps numeric columns AS-IS in outputs:
    china_stance_score, sensitive, collective_action,
    english, mandarin, spanish, other_lang, no_language
- Builds TEMPORARY buckets ONLY for balancing/caps:
    stance_bucket: 3 bins via --stance-cut1/--stance-cut2 (default -0.33, +0.33)
    sensitive_bucket / collective_bucket: numeric >= --yn-cap-thresh → 'yes' else 'no'
- Ranks deterministically within each (stance_bucket, sensitive_bucket, collective_bucket) cell.
- Applies per-cell caps for TRAIN and VAL with a train/val ratio.
- Guarantees: if cell count c ≥ 2 → both splits get ≥1 item (subject to VAL cap).
- Writes two CSVs with RAW numeric labels unchanged.

Usage:
python sample_labels_duckdb.py \
  --labels-csv data/china_labeling_sample_all_Jul30.csv \
  --out-train  data/labels_train_new.csv \
  --out-val    data/labels_val_new.csv \
  --labels-video-col video \
  --stance-col pol \
  --train-frac 0.7 \
  --train-cap-per-cell 100 \
  --val-cap-per-cell 50 \
  --yn-cap-thresh 0.5 \
  --stance-cut1 -0.33 \
  --stance-cut2  0.33 \
  --seed 7
"""
import csv
import argparse
import os
import duckdb

OUT_COLS = [
    "meta_id", "china_stance_score", "sensitive", "collective_action",
    "english", "mandarin", "spanish", "other_lang", "no_language"
]

def sql_q(s: str) -> str:
    return s.replace("'", "''")

def connect_db():
    return duckdb.connect()

def build_raw(con, labels_csv: str):
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE RAW AS
      SELECT *
      FROM read_csv_auto('{sql_q(labels_csv)}', HEADER=TRUE, SAMPLE_SIZE=-1);
    """)

def _sniff_delim(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            sample = fh.read(2048)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except Exception:
        return ","

def build_L0(con, labels_csv: str, labels_video_col: str, stance_col: str | None = None):
    """
    Raw parse; keep only rows with meta_id. Inputs are NUMERIC coder means already.
    Expected columns in CSV:
      china_stance_score (or pol), sensitive, collective_action,
      english, mandarin, spanish, other_lang, no_language, and the video path column.
    """
    # First attempt: auto-detect
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE SRC AS
        SELECT * FROM read_csv_auto('{sql_q(labels_csv)}', HEADER=TRUE, SAMPLE_SIZE=-1);
    """)
    cols = [r[1] for r in con.execute("PRAGMA table_info('SRC')").fetchall()]

    # If header failed (numeric col names) or required columns missing, re-read with forced delimiter
    header_broken = cols and all(c.isdigit() for c in cols)
    need_any_stance = ("china_stance_score" in cols) or ("pol" in cols) or (stance_col and stance_col in cols)
    if header_broken or (labels_video_col not in cols) or (not need_any_stance):
        delim = _sniff_delim(labels_csv)
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE SRC AS
            SELECT * FROM read_csv_auto('{sql_q(labels_csv)}', HEADER=TRUE, DELIM='{delim}', SAMPLE_SIZE=-1);
        """)
        cols = [r[1] for r in con.execute("PRAGMA table_info('SRC')").fetchall()]

    # Choose the stance column: explicit → china_stance_score → pol
    stance_source = None
    if stance_col and stance_col in cols:
        stance_source = stance_col
    elif "china_stance_score" in cols:
        stance_source = "china_stance_score"
    elif "pol" in cols:
        stance_source = "pol"

    required = [labels_video_col, "sensitive", "collective_action",
                "english", "mandarin", "spanish", "other_lang", "no_language"]
    missing = [c for c in required if c not in cols]
    if missing or stance_source is None:
        raise SystemExit(
            f"[error] Missing columns: {missing + ([] if stance_source else ['china_stance_score/pol/<--stance-col>'])}. "
            f"Found columns: {cols}"
        )

    print(f"[info] Using stance column: {stance_source}")

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE L0 AS
    SELECT
      regexp_extract({labels_video_col}, '([0-9]{{10,}})\\.mp4', 1)              AS meta_id,
      CAST({stance_source}     AS DOUBLE) AS china_stance_score,
      CAST(sensitive           AS DOUBLE) AS sensitive,
      CAST(collective_action   AS DOUBLE) AS collective_action,
      CAST(english             AS DOUBLE) AS english,
      CAST(mandarin            AS DOUBLE) AS mandarin,
      CAST(spanish             AS DOUBLE) AS spanish,
      CAST(other_lang          AS DOUBLE) AS other_lang,
      CAST(no_language         AS DOUBLE) AS no_language
    FROM SRC
    WHERE {labels_video_col} IS NOT NULL
      AND regexp_extract({labels_video_col}, '([0-9]{{10,}})\\.mp4', 1) IS NOT NULL
      AND {stance_source} IS NOT NULL;
    """)

def build_LN(con):
    # passthrough (keep raw numeric labels as-is)
    con.execute("""
      CREATE OR REPLACE TEMP TABLE LN AS
      SELECT *
      FROM L0
      WHERE meta_id IS NOT NULL;
    """)

def build_BUCKETS(con, yn_cap_thresh: float, cut1: float, cut2: float):
    """
    Temporary buckets for balancing only:
      stance_bucket ∈ {'neg','neu','pos'}
      sensitive_bucket, collective_bucket ∈ {'no','yes'} via yn_cap_thresh
    """
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE BUCKETS AS
    SELECT
      *,
      CASE
        WHEN china_stance_score < {float(cut1)} THEN 'neg'
        WHEN china_stance_score > {float(cut2)} THEN 'pos'
        ELSE 'neu'
      END AS stance_bucket,
      CASE WHEN sensitive         >= {float(yn_cap_thresh)} THEN 'yes' ELSE 'no' END AS sensitive_bucket,
      CASE WHEN collective_action >= {float(yn_cap_thresh)} THEN 'yes' ELSE 'no' END AS collective_bucket
    FROM LN;
    """)

def rank_cells(con, seed: int):
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE RANKED AS
    SELECT
      *,
      row_number() OVER (
        PARTITION BY stance_bucket, sensitive_bucket, collective_bucket
        ORDER BY hash(meta_id || '-SPLIT-{int(seed)}')
      ) AS rn,
      count(*) OVER (
        PARTITION BY stance_bucket, sensitive_bucket, collective_bucket
      ) AS c
    FROM BUCKETS;
    """)

def allocate_counts(con, train_cap_per_cell: int, val_cap_per_cell: int, train_frac: float):
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE ALLOC AS
    SELECT
      *,
      LEAST(
        {int(train_cap_per_cell)},
        CASE
          WHEN c = 1 THEN 1
          WHEN c = 2 THEN 1
          ELSE GREATEST(1, CAST(ROUND(c * {float(train_frac)}) AS INTEGER))
        END,
        CASE WHEN c >= 2 THEN c - 1 ELSE 1 END
      ) AS train_n
    FROM RANKED;
    """)
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE ALLOC2 AS
    SELECT
      *,
      LEAST({int(val_cap_per_cell)}, GREATEST(0, c - train_n)) AS val_n
    FROM ALLOC;
    """)

def slice_splits(con):
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE TRAIN AS
    SELECT {", ".join(OUT_COLS)}
    FROM ALLOC2
    WHERE rn <= train_n;
    """)
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE VAL AS
    SELECT {", ".join(OUT_COLS)}
    FROM ALLOC2
    WHERE rn > train_n AND rn <= train_n + val_n;
    """)

def write_csv(con, table: str, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    con.execute(f"COPY (SELECT {', '.join(OUT_COLS)} FROM {table}) TO '{sql_q(out_path)}' (FORMAT CSV, HEADER);")

def summarize(con, table: str, title: str):
    print(f"\n== {title} ==")
    total = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    stance_rng = con.execute(f"SELECT MIN(china_stance_score), MAX(china_stance_score) FROM {table}").fetchone()
    means = con.execute(f"""
      SELECT
        avg(sensitive), avg(collective_action),
        avg(english), avg(mandarin), avg(spanish), avg(other_lang), avg(no_language)
      FROM {table}
    """).fetchone()
    print(f"rows={total}  stance_range={stance_rng}")
    print("means: sensitive=%.3f collective=%.3f | lang: en=%.3f zh=%.3f es=%.3f other=%.3f none=%.3f" % means)

def audit_starved(con):
    res = con.execute("""
      WITH C AS (
        SELECT stance_bucket, sensitive_bucket, collective_bucket, COUNT(*) AS c
        FROM BUCKETS GROUP BY 1,2,3
      ),
      V AS (
        SELECT stance_bucket, sensitive_bucket, collective_bucket, COUNT(*) AS v
        FROM ALLOC2
        WHERE rn > train_n AND rn <= train_n + val_n
        GROUP BY 1,2,3
      )
      SELECT C.stance_bucket, C.sensitive_bucket, C.collective_bucket, C.c, COALESCE(V.v,0) AS val_got
      FROM C LEFT JOIN V USING (stance_bucket, sensitive_bucket, collective_bucket)
      WHERE C.c >= 2 AND COALESCE(V.v,0) = 0
      ORDER BY C.c DESC;
    """).fetchall()
    if res:
        print("\nCells with c>=2 but VAL received 0 (lower --train-frac or raise --val-cap-per-cell):")
        for r in res:
            print(f"{r[0]:>3} | {r[1]:>3} | {r[2]:>3}  c={r[3]}  val=0")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--labels-video-col", default="video",
                    help="Column with file path containing .../<digits>.mp4")
    ap.add_argument("--train-frac", type=float, default=0.7, help="fraction per cell to TRAIN (0..1)")
    ap.add_argument("--train-cap-per-cell", type=int, default=100)
    ap.add_argument("--val-cap-per-cell", type=int, default=50)
    ap.add_argument("--yn-cap-thresh", type=float, default=0.5,
                    help="TEMPORARY threshold for sensitive/collective *bucketing* (outputs remain raw)")
    ap.add_argument("--stance-cut1", type=float, default=-0.33,
                    help="TEMPORARY lower cut for stance bucket (neg if < cut1)")
    ap.add_argument("--stance-cut2", type=float, default=0.33,
                    help="TEMPORARY upper cut for stance bucket (pos if > cut2)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stance-col", default=None,
                help="Explicit stance column name in the CSV (e.g., 'pol'). "
                     "If not provided, tries 'china_stance_score' then 'pol'.")

    return ap.parse_args()

def main():
    args = parse_args()
    con = connect_db()
    build_raw(con, args.labels_csv)
    build_L0(con, args.labels_csv, args.labels_video_col, stance_col=args.stance_col)
    build_LN(con)
    build_BUCKETS(con, args.yn_cap_thres, args.stance_cut1, args.stance_cut2) if False else None  # keeps linter happy

    # call with real args
    build_BUCKETS(con, args.yn_cap_thresh, args.stance_cut1, args.stance_cut2)
    rank_cells(con, args.seed)
    allocate_counts(con, args.train_cap_per_cell, args.val_cap_per_cell, args.train_frac)
    slice_splits(con)
    write_csv(con, "TRAIN", args.out_train)
    write_csv(con, "VAL",   args.out_val)
    summarize(con, "TRAIN", "TRAIN (raw labels, caps by buckets)")
    summarize(con, "VAL",   "VAL   (raw labels, caps by buckets)")
    audit_starved(con)

if __name__ == "__main__":
    main()
