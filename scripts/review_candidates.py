#!/usr/bin/env python
# Genera un HTML sencillo para escuchar candidatos y decidir cuáles aprobar.

from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="candidates_review.html")
    ap.add_argument("--approved-only", action="store_true")
    ap.add_argument("--sample-per-speaker", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    df = pd.read_csv(args.candidates)
    if args.approved_only and "approved" in df:
        approved = pd.to_numeric(df["approved"], errors="coerce").fillna(0)
        df = df.loc[approved.eq(1)].copy()
    risk_columns = [
        column for column in ("syncnet_confidence", "face_margin", "snr_db") if column in df
    ]
    if risk_columns:
        df = df.sort_values(risk_columns, ascending=True)
    if args.sample_per_speaker > 0:
        samples = []
        for _, group in df.groupby("speaker_id", sort=True):
            boundary_count = min(len(group), max(1, args.sample_per_speaker // 2))
            boundary = group.head(boundary_count)
            remainder = group.drop(boundary.index)
            random_count = min(len(remainder), args.sample_per_speaker - len(boundary))
            random = remainder.sample(random_count, random_state=args.seed) if random_count else remainder.head(0)
            samples.append(pd.concat([boundary, random]))
        df = pd.concat(samples, ignore_index=True) if samples else df.head(0)
    rows = []
    for _, r in df.iterrows():
        path = html.escape(str(r["candidate_path"]), quote=True)
        rows.append(f"""
        <tr>
          <td>{r.get('speaker_id','')}</td>
          <td>{r.get('speaker_name','')}</td>
          <td>{r.get('video_id','')}</td>
          <td>{r.get('candidate_idx','')}</td>
          <td>{float(r.get('start',0)):.2f}-{float(r.get('end',0)):.2f}</td>
          <td>{float(r.get('rank_score',0)):.3f}</td>
          <td>{float(r.get('snr_db',0)):.1f}</td>
          <td>{float(r.get('vad_ratio',0)):.2f}</td>
          <td>{float(r.get('face_margin',-1)):.3f}</td>
          <td>{float(r.get('syncnet_confidence',-1)):.3f}</td>
          <td><code>{html.escape(str(r.get('sha1','')))}</code></td>
          <td><audio controls src="{path}"></audio></td>
        </tr>
        """)
    html = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Review candidates</title>
<style>body{{font-family:Arial}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ddd;padding:4px;font-size:12px}} tr:nth-child(even){{background:#f7f7f7}}</style>
</head><body>
<h1>VoxCeleb-ESP candidates</h1>
<p>Audita identidad y ausencia de voces solapadas. Para rechazar, copia el SHA-1 a <code>manual_rejections.csv</code>; no edites <code>approved</code>, porque la aprobación se recalcula.</p>
<table><tr><th>speaker</th><th>name</th><th>video</th><th>idx</th><th>time</th><th>rank</th><th>SNR</th><th>VAD</th><th>face margin</th><th>SyncNet</th><th>SHA-1</th><th>audio</th></tr>
{''.join(rows)}
</table></body></html>"""
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"✅ HTML creado: {args.out}")


if __name__ == "__main__":
    main()
