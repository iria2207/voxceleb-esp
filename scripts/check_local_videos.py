#!/usr/bin/env python3
#Comprueba que los youtube_id de videos.csv existen como <youtube_id>.mp4.

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', required=True)
    ap.add_argument('--video-dir', required=True)
    ap.add_argument('--fail-if-missing', action='store_true')
    ap.add_argument('--out-missing', default='missing_videos.txt')
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        print(f"video-dir no existe: {video_dir}")
        if args.fail_if_missing:
            raise SystemExit(2)

    rows = []
    with open(args.videos, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            use = str(r.get('use','1')).strip().lower()
            if use in {'0','false','no'}:
                continue
            yid = (r.get('youtube_id') or '').strip()
            if not yid:
                continue
            rows.append(r)

    missing = []
    found = 0
    for r in rows:
        yid = r['youtube_id'].strip()
        # Formato principal esperado por el pipeline
        direct = video_dir / f"{yid}.mp4"
        if direct.exists():
            found += 1
            continue
        # Fallback por si rclone dejó extensiones o nombres raros; no cambia el pipeline, solo avisa
        variants = list(video_dir.glob(f"{yid}.*"))
        if variants:
            print(f"Existe {variants[0].name}, pero el pipeline espera {yid}.mp4")
        missing.append(yid)

    total = len(rows)
    print(f"Vídeos esperados: {total}")
    print(f"Encontrados como <youtube_id>.mp4: {found}")
    print(f"Faltan: {len(missing)}")

    out = Path(args.out_missing)
    if missing:
        out.write_text('\n'.join(missing) + '\n', encoding='utf-8')
        print(f"Lista de faltantes guardada en: {out}")
        print("Primeros faltantes:")
        for yid in missing[:20]:
            print(f"  - {yid}")
    else:
        if out.exists():
            out.unlink()

    if missing and args.fail_if_missing:
        raise SystemExit(3)


if __name__ == '__main__':
    main()
