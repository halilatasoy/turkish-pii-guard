#!/usr/bin/env python
"""Compare two eval result CSVs (from scripts/eval.py) row by row and print a diff summary.
  python scripts/error_report.py results/baseline_270m.csv results/v1.csv
"""
import sys, pandas as pd
a, b = (pd.read_csv(p) for p in sys.argv[1:3])
m = a[["id", "kategori", "ozellikler", "dogru", "pred"]].merge(b[["id", "dogru", "pred", "expected_output", "instruction", "input"]], on="id", suffixes=("_a", "_b"))
print(f"A exact: {m.dogru_a.mean():.3f}   B exact: {m.dogru_b.mean():.3f}")
print(f"fixed by B: {(~m.dogru_a & m.dogru_b).sum()}   broken by B: {(m.dogru_a & ~m.dogru_b).sum()}   still wrong: {(~m.dogru_a & ~m.dogru_b).sum()}")
print("\nper kategori (A -> B):")
print(m.groupby("kategori")[["dogru_a", "dogru_b"]].mean().round(3))
print("\nBROKEN BY B (sample):")
for r in m[m.dogru_a & ~m.dogru_b].head(10).itertuples():
    print(f"  [{r.kategori}|{r.ozellikler}] {r.instruction[:70]}\n    EXP: {r.expected_output[:140]}\n    GOT: {r.pred_b[:140]}")
print("\nSTILL WRONG (sample):")
for r in m[~m.dogru_a & ~m.dogru_b].sample(min(15, int((~m.dogru_a & ~m.dogru_b).sum())), random_state=0).itertuples():
    print(f"  [{r.kategori}|{r.ozellikler}] {r.instruction[:70]}\n    EXP: {r.expected_output[:140]}\n    GOT: {r.pred_b[:140]}")
