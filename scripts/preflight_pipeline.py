#!/usr/bin/env python3
#Comprobaciones rápidas antes de gastar horas de GPU.

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = raw.get("global", {})
    celebrities = raw.get("celebrities", []) or []
    errors: list[str] = []
    warnings: list[str] = []

    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            errors.append(f"no se encuentra {binary} en PATH")
    required_packages = [
        "numpy", "pandas", "cv2", "insightface", "soundfile", "torch",
        "python_speech_features",
    ]
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            errors.append(f"falta el paquete Python {package}")
    cv2 = None
    if importlib.util.find_spec("cv2") is not None:
        import cv2 as cv2_module

        cv2 = cv2_module

    models_dir = Path(cfg.get("models_dir", "models")).expanduser()
    syncnet_code = Path(cfg.get("syncnet_code_dir", models_dir / "syncnet" / "syncnet_python"))
    syncnet_model = Path(cfg.get("syncnet_model_path") or models_dir / "syncnet" / "syncnet_v2.model")
    for path in (syncnet_model, syncnet_code / "SyncNetInstance.py", syncnet_code / "SyncNetModel.py"):
        if not path.is_file():
            errors.append(f"falta componente SyncNet: {path}")

    download_path = Path(cfg.get("download_path", "videos")).expanduser()
    target = int(cfg.get("target_clips_per_speaker", 50))
    cap = int(cfg.get("segments_per_video", 20))
    min_per_video = int(cfg.get("min_selected_per_video", 5))
    early_stop_enabled = bool(cfg.get("early_stop_candidates_enabled", True))
    early_stop_cap = int(cfg.get("early_stop_candidates_per_video", 25))
    min_videos = int(cfg.get("min_source_videos_per_speaker", 3))
    min_refs = int(cfg.get("face_template_min_refs", 2))
    if not celebrities:
        errors.append("la configuración no contiene hablantes")
    if early_stop_enabled and early_stop_cap < cap:
        errors.append(
            "early_stop_candidates_per_video no puede ser menor que segments_per_video"
        )

    for celeb in celebrities:
        speaker_id = str(celeb.get("id", "unknown"))
        videos = [v for v in celeb.get("videos", []) if v.get("use", True)]
        refs = [Path(p) for p in celeb.get("reference_images", [])]
        resolved_refs = [p if p.is_absolute() else config_path.parent / p for p in refs]
        existing_refs = [p for p in resolved_refs if p.is_file()]
        readable_refs = []
        for reference_path in existing_refs:
            image = cv2.imread(str(reference_path)) if cv2 is not None else None
            if image is None:
                errors.append(
                    f"{speaker_id}: referencia no decodificable por OpenCV: {reference_path} "
                    "(puede ser AVIF/WEBP renombrado como .jpg)"
                )
            else:
                readable_refs.append(reference_path)
        existing_videos = [
            download_path / f"{str(video.get('youtube_id', '')).strip()}.mp4" for video in videos
        ]
        existing_videos = [p for p in existing_videos if p.is_file()]
        if len(readable_refs) < min_refs:
            errors.append(
                f"{speaker_id}: solo {len(readable_refs)}/{min_refs} referencias faciales legibles"
            )
        if len(existing_videos) < min_videos:
            errors.append(f"{speaker_id}: solo {len(existing_videos)}/{min_videos} vídeos locales")
        if len(videos) * cap < target:
            errors.append(
                f"{speaker_id}: imposible llegar a {target} clips con {len(videos)} vídeos y tope {cap}"
            )
        if len(videos) * min_per_video > target:
            errors.append(
                f"{speaker_id}: reservar {min_per_video} clips de cada uno de "
                f"{len(videos)} vídeos supera la cuota total de {target}"
            )
    print(f"Preflight: {len(celebrities)} hablantes, objetivo={target} clips/hablante")
    for warning in warnings:
        print(f"ADVERTENCIA: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(2)
    print("Preflight correcto: dependencias, modelos, referencias y vídeos disponibles")


if __name__ == "__main__":
    main()
