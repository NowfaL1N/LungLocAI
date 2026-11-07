"""Make a compact retraining table from the LungLocAI-3D dataset (IEEE DataPort, DOI 10.21227/0x9w-1y93).

    python tools/make_base_table.py --metadata metadata.csv --features features.csv --out base_table.csv

Keeps every nodule and a random sample of non-nodules from the TRAIN and VAL splits (the test split is left out so
it stays a clean benchmark), rounds values to 2 decimals and stops below --max-mb (default 60 MB) so the file fits in
the private GitHub storage repo. Upload the result on the website's /admin page ("Base training table").
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="base_table.csv")
    ap.add_argument("--max-mb", type=float, default=60)
    ap.add_argument("--neg-per-pos", type=float, default=8.0)
    a = ap.parse_args()

    meta = pd.read_csv(a.metadata)
    meta = meta[meta["split"].isin(["train", "val"])]
    pos = meta[meta.label == 1]
    neg = meta[meta.label == 0].sample(frac=1, random_state=42)
    n_neg = int(len(pos) * a.neg_per_pos)
    while True:
        keep = pd.concat([pos, neg.head(n_neg)])
        ids = set(keep.candidate_id)
        feats = pd.concat(c[c.candidate_id.isin(ids)] for c in pd.read_csv(a.features, chunksize=5000))
        df = keep[["candidate_id", "seriesuid", "label", "split"]].rename(columns={"label": "is_nodule"}) \
            .merge(feats, on="candidate_id")
        df.to_csv(a.out, index=False, float_format="%.2f")
        mb = Path(a.out).stat().st_size / 1e6
        print(f"{len(df)} rows ({int(df.is_nodule.sum())} nodules) -> {a.out}: {mb:.1f} MB")
        if mb <= a.max_mb or n_neg < len(pos):
            break
        n_neg = int(n_neg * a.max_mb / mb * 0.95)


if __name__ == "__main__":
    main()
