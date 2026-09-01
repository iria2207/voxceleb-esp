#!/usr/bin/env python
#Genera un YAML de configuración desde speakers.csv y videos.csv.

from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import yaml


def norm_name(s: str) -> str:
    s = str(s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.split())


def read_exclusions(path: str | None) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    return {norm_name(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def split_refs(value: str) -> List[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def truthy(x) -> bool:
    if pd.isna(x):
        return True
    return str(x).strip().lower() not in {"0", "false", "no", "n"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--speakers", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-videos-per-speaker", type=int, default=2)
    ap.add_argument("--limit-speakers", type=int, default=None)
    ap.add_argument("--exclude-speakers-file", default=None)
    args = ap.parse_args()

    base_path = Path(args.base_config)
    with open(base_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    speakers = pd.read_csv(args.speakers).fillna("")
    videos = pd.read_csv(args.videos).fillna("")
    exclusions = read_exclusions(args.exclude_speakers_file)

    required_s = {"speaker_id", "name", "reference_images"}
    required_v = {"speaker_id", "youtube_id"}
    miss_s = required_s - set(speakers.columns)
    miss_v = required_v - set(videos.columns)
    if miss_s:
        raise ValueError(f"Faltan columnas en speakers.csv: {sorted(miss_s)}")
    if miss_v:
        raise ValueError(f"Faltan columnas en videos.csv: {sorted(miss_v)}")

    if "use" in speakers.columns:
        speakers = speakers[speakers["use"].apply(truthy)]
    if "exclude_from_train" in speakers.columns:
        speakers = speakers[~speakers["exclude_from_train"].apply(lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"})]
    if exclusions:
        speakers = speakers[~speakers["name"].apply(lambda x: norm_name(x) in exclusions)]

    if "use" in videos.columns:
        videos = videos[videos["use"].apply(truthy)]

    celebrities = []
    for _, spk in speakers.iterrows():
        sid = str(spk["speaker_id"]).strip()
        name = str(spk["name"]).strip()
        if not sid or not name:
            continue
        vdf = videos[videos["speaker_id"].astype(str).str.strip() == sid].copy()
        if "priority" in vdf.columns:
            vdf["priority"] = pd.to_numeric(vdf["priority"], errors="coerce").fillna(1).astype(int)
            vdf = vdf.sort_values("priority")
        if len(vdf) < args.min_videos_per_speaker:
            continue
        refs = split_refs(spk.get("reference_images", ""))
        if not refs:
            continue
        celeb = {
            "id": sid,
            "name": name,
            "gender": str(spk.get("gender", "unknown") or "unknown"),
            "category": str(spk.get("category", "unknown") or "unknown"),
            "region": str(spk.get("region", "unknown") or "unknown"),
            "birth_year": int(spk["birth_year"]) if str(spk.get("birth_year", "")).strip().isdigit() else None,
            "reference_images": refs,
            "videos": [],
            "notes": str(spk.get("notes", "")),
        }
        for _, v in vdf.iterrows():
            yid = str(v["youtube_id"]).strip()
            if not yid:
                continue
            celeb["videos"].append({
                "youtube_id": yid,
                "url": str(v.get("url", "")).strip(),
                "split": str(v.get("split", "train") or "train"),
                "priority": int(v.get("priority", 1) or 1),
                "use": True,
                "notes": str(v.get("notes", "")),
            })
        if len(celeb["videos"]) >= args.min_videos_per_speaker:
            celebrities.append(celeb)
        if args.limit_speakers and len(celebrities) >= args.limit_speakers:
            break

    config["celebrities"] = celebrities
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    print(f"YAML generado: {out}")
    print(f"Speakers incluidos: {len(celebrities)}")
    print(f"Vídeos incluidos: {sum(len(c['videos']) for c in celebrities)}")
    print(f"Excluidos por lista test: {len(exclusions)} nombres en lista")


if __name__ == "__main__":
    main()
