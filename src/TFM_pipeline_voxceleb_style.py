#!/usr/bin/env python
# coding: utf-8
#Pipeline VoxCeleb-style para crear VoxCeleb-ESP-Train.

from __future__ import annotations

import argparse
import copy
import datetime as dt
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from VOXCELEB_ESP.config_loader import ConfigLoader
from VOXCELEB_ESP.extractor import MediaExtractor
from VOXCELEB_ESP.face_detector import FaceDetector
from VOXCELEB_ESP.face_tracker import FaceTracker
from VOXCELEB_ESP.identity import IdentityManager
from VOXCELEB_ESP.active_speaker import ActiveSpeakerVerifier
from VOXCELEB_ESP.quality_filter import QualityFilter
from VOXCELEB_ESP.exporter import VoxCelebExporter
from VOXCELEB_ESP.real_verification import SyncNetTrackVerifier
from VOXCELEB_ESP.utils import sha1_file


def write_speaker_scoped_csv(
    path: Path,
    owned_rows: pd.DataFrame,
    owned_speaker_ids: Iterable[str],
    dedup_subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fusiona de forma atómica las filas de un grupo de hablantes.

    Dos jobs pueden procesar grupos disjuntos en paralelo. El bloqueo evita que
    ambos reescriban el CSV a la vez y la fusión conserva siempre las filas que
    pertenecen al otro job.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = {str(speaker_id) for speaker_id in owned_speaker_ids}
    # El volumen compartido de phobos no ofrece locks POSIX (fcntl devuelve
    # ENOLCK). La creación de un directorio sí es atómica en ese filesystem y
    # sirve como mutex entre los dos jobs.
    lock_dir = path.with_name(f".{path.name}.lockdir")
    deadline = time.monotonic() + 300.0
    acquired = False
    while not acquired:
        try:
            lock_dir.mkdir()
            acquired = True
        except FileExistsError:
            try:
                stale_seconds = time.time() - lock_dir.stat().st_mtime
            except FileNotFoundError:
                continue
            if stale_seconds > 600.0:
                try:
                    lock_dir.rmdir()
                except OSError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timeout while waiting for the lock on {path}")
            time.sleep(0.1)

    try:
        current = pd.DataFrame()
        if path.exists() and path.stat().st_size:
            current = pd.read_csv(path)

        if not current.empty and "speaker_id" in current:
            other_rows = current.loc[
                ~current["speaker_id"].astype(str).isin(selected)
            ].copy()
        else:
            other_rows = pd.DataFrame()

        if not owned_rows.empty and "speaker_id" in owned_rows:
            selected_rows = owned_rows.loc[
                owned_rows["speaker_id"].astype(str).isin(selected)
            ].copy()
        else:
            selected_rows = owned_rows.copy()

        merged = pd.concat([other_rows, selected_rows], ignore_index=True)
        if dedup_subset and not merged.empty:
            available = [column for column in dedup_subset if column in merged]
            if available:
                merged = merged.drop_duplicates(subset=available, keep="last")

        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            merged.to_csv(temporary, index=False)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        if acquired:
            try:
                lock_dir.rmdir()
            except FileNotFoundError:
                pass
    return merged


def speakers_with_completed_quota(cfg) -> set[str]:
    """Devuelve hablantes que ya tienen la cuota aprobada completa."""
    approved_csv = Path(cfg.data_candidates) / "approved_segments.csv"
    if not approved_csv.exists() or not approved_csv.stat().st_size:
        return set()
    approved = pd.read_csv(approved_csv)
    if approved.empty or "speaker_id" not in approved or "approved" not in approved:
        return set()
    target = int(
        getattr(cfg, "target_clips_per_speaker", getattr(cfg, "min_clips_per_speaker", 50))
    )
    selected = approved.loc[
        pd.to_numeric(approved["approved"], errors="coerce").fillna(0).ge(1)
    ].copy()
    if selected.empty:
        return set()
    counts = selected.groupby(selected["speaker_id"].astype(str)).size()
    return {str(speaker_id) for speaker_id, count in counts.items() if int(count) >= target}


def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(os.environ.get("JOB_ID", os.getpid()))
    log_file = log_dir / (
        f"voxceleb_train_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_id}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("VoxCeleb-ESP-Train pipeline started")
    logging.info("Log: %s", log_file)
    return log_file


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Convierte a mono float sin ocultar clipping ni alterar el SNR."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)


def slice_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    s = max(0, int(float(start) * sr))
    e = min(len(audio), int(float(end) * sr))
    if e <= s:
        return np.zeros(0, dtype=np.float32)
    return audio[s:e].astype(np.float32)


def active_segments_to_windows(active_segments: List[Dict], cfg) -> List[Dict]:
    """Convierte tramos activos en ventanas de duración objetivo."""
    min_d = float(getattr(cfg, "segment_min_duration", 2.0))
    target = float(getattr(cfg, "segment_target_duration", 3.0))
    max_d = float(getattr(cfg, "segment_max_duration", 5.0))
    max_candidates = int(getattr(cfg, "max_candidates_per_video", 80))

    windows: List[Dict] = []
    for seg in active_segments:
        st = float(seg["start"])
        en = float(seg["end"])
        if en <= st or en - st < min_d:
            continue
        dur = en - st
        if dur <= max_d:
            s = st
            e = en
            if e - s > target:
                c = (s + e) / 2.0
                s = max(st, c - target / 2.0)
                e = min(en, s + target)
            d = dict(seg)
            d.update({"start": s, "end": e, "duration": e - s})
            windows.append(d)
        else:
            # Ventanas no solapadas: evita casi duplicados temporales.
            step = max(target, min_d)
            t = st
            while t + min_d <= en + 1e-6:
                e = min(t + target, en)
                if e - t >= min_d:
                    d = dict(seg)
                    d.update({"start": float(t), "end": float(e), "duration": float(e - t)})
                    windows.append(d)
                t += step

    def rank(w: Dict) -> float:
        conf = float(w.get("confidence", 0.0))
        visual = float(w.get("visual_score", 0.0))
        audio = float(w.get("audio_score", 0.0))
        dur_pen = abs(float(w.get("duration", 0.0)) - target) / max(target, 1e-6)
        return conf + 0.10 * visual + 0.05 * audio - 0.10 * dur_pen

    windows.sort(key=rank, reverse=True)
    selected: List[Dict] = []
    for w in windows:
        # Evita candidatos casi idénticos temporalmente.
        overlap_too_high = False
        for p in selected:
            inter = max(0.0, min(w["end"], p["end"]) - max(w["start"], p["start"]))
            union = max(w["end"], p["end"]) - min(w["start"], p["start"])
            if union > 0 and inter / union > 0.70:
                overlap_too_high = True
                break
        if not overlap_too_high:
            selected.append(w)
        if len(selected) >= max_candidates:
            break
    return selected


def iter_videos(celebrities) -> Iterable[tuple]:
    for celeb in celebrities:
        videos = sorted(getattr(celeb, "videos", []), key=lambda v: getattr(v, "priority", 1))
        for video in videos:
            if getattr(video, "use", True):
                yield celeb, video


def _track_candidate_coverage(track: Dict, start: float, end: float, fps: float) -> float:
    """Fracción de frames esperados del intervalo cubiertos por un track."""
    frames = np.asarray(track.get("frames", []), dtype=int)
    if frames.size == 0 or end <= start:
        return 0.0
    first = int(np.floor(start * fps))
    last = int(np.ceil(end * fps))
    expected = max(1, last - first + 1)
    present = int(np.count_nonzero((frames >= first) & (frames <= last)))
    return float(present / expected)


def tracks_to_syncnet_probe_windows(tracks: List[Dict], cfg) -> List[Dict]:
    """Cubre exhaustivamente los tracks objetivo para que SyncNet diarice.

    La heurística de boca/energía ya no decide qué intervalos llegan a SyncNet:
    precisamente estaba omitiendo habla real. Cada probe puede producir varios
    subsegmentos framewise de 2--3 segundos.
    """
    fps = float(getattr(cfg, "detect_fps", 5.0))
    probe_duration = float(getattr(cfg, "syncnet_probe_duration_seconds", 12.0))
    probe_hop = float(getattr(cfg, "syncnet_probe_hop_seconds", 10.0))
    min_duration = float(getattr(cfg, "syncnet_probe_min_duration_seconds", 2.0))
    min_coverage = float(getattr(cfg, "syncnet_probe_min_track_coverage", 0.70))
    max_probes = int(getattr(cfg, "syncnet_max_probes_per_window", 160))
    probes: List[Dict] = []
    for track in tracks:
        frames = np.asarray(track.get("frames", []), dtype=int)
        if frames.size == 0:
            continue
        track_start = float(frames.min() / fps)
        track_end = float((frames.max() + 1) / fps)
        if track_end - track_start < min_duration:
            continue
        starts: List[float] = []
        cursor = track_start
        while cursor + min_duration <= track_end + 1e-6:
            starts.append(cursor)
            if cursor + probe_duration >= track_end:
                break
            cursor += probe_hop
        for start in starts:
            end = min(track_end, start + probe_duration)
            if end - start < min_duration:
                continue
            coverage = _track_candidate_coverage(track, start, end, fps)
            if coverage < min_coverage:
                continue
            probes.append({
                "track_id": str(track.get("track_id", track.get("id", ""))),
                "start": float(start),
                "end": float(end),
                "duration": float(end - start),
                "confidence": 0.0,
                "audio_score": 0.0,
                "visual_score": 0.0,
                "face_conf": float(track.get("face_conf", 0.0)),
                "face_other_conf": float(track.get("face_other_conf", -1.0)),
                "face_margin": float(track.get("face_margin", 0.0)),
                "face_track_consistency": float(track.get("face_track_consistency", 0.0)),
                "method": "syncnet_framewise_track_probe",
            })
    probes.sort(key=lambda row: (float(row["start"]), -float(row["face_conf"])))
    return probes[:max_probes]


def apply_relaxed_candidate_profile(cfg) -> None:
    """Segunda pasada inclusiva para hablantes con cobertura insuficiente.

    Los candidatos obtenidos con este perfil deben auditarse manualmente. Los
    umbrales siguen exigiendo coincidencia facial, actividad audiovisual y una
    calidad mínima de voz; no se acepta audio usando solo VAD.
    """
    if bool(getattr(cfg, "real_asd_required", True)):
        raise RuntimeError(
            "The relaxed profile is forbidden when real_asd_required=true. "
            "Add more videos instead of lowering the speaker identity requirements."
        )
    overrides = {
        "candidate_profile": "relaxed",
        "detect_fps": 5,
        "min_face_size": 60,
        "min_track_frames": 8,
        "max_lost_frames": 8,
        "face_template_threshold": 0.50,
        "segment_min_duration": 2.0,
        "segment_target_duration": 6.0,
        "syncnet_threshold": 0.40,
        "syncnet_min_confidence": 0.40,
        "active_min_audio_score": 0.20,
        "active_min_visual_score": 0.04,
        "min_speech_ratio": 0.45,
        "min_snr_db": 5.0,
    }
    for key, value in overrides.items():
        setattr(cfg, key, value)
    logging.warning("RELAXED candidate profile enabled; manual review is required")
    logging.info(
        "Relaxed profile: fps=%s face_threshold=%.2f active_threshold=%.2f min_speech=%.2f min_snr=%.1f",
        cfg.detect_fps,
        cfg.face_template_threshold,
        cfg.syncnet_threshold,
        cfg.min_speech_ratio,
        cfg.min_snr_db,
    )



def iter_video_windows(extractor: MediaExtractor, video_path: Path, cfg) -> List[Dict]:
    """Devuelve ventanas internas para procesar vídeos largos sin modificarlos.

    Cada ventana se extrae con ffmpeg -ss/-t desde el vídeo original. Así no se
    generan todos los frames de un vídeo de 30-60 minutos de golpe.
    """
    mode = str(getattr(cfg, "long_video_mode", "windowed")).lower()
    duration = extractor.probe_duration(video_path)
    if duration <= 0:
        duration = float(getattr(cfg, "long_video_window_seconds", 600))

    if mode in {"off", "full", "false", "0"}:
        return [{"idx": 0, "start": 0.0, "duration": duration, "tag": "full", "video_duration": duration}]

    win = float(getattr(cfg, "long_video_window_seconds", 600))
    hop = float(getattr(cfg, "long_video_hop_seconds", win))
    start0 = float(getattr(cfg, "long_video_start_seconds", 0))
    min_remaining = float(getattr(cfg, "long_video_min_remaining_seconds", 60))
    max_windows = int(getattr(cfg, "long_video_max_windows", 0))
    strategy = str(getattr(cfg, "long_video_strategy", "sliding")).lower()

    if duration <= win:
        return [{"idx": 0, "start": 0.0, "duration": duration, "tag": "w0000", "video_duration": duration}]

    windows: List[Dict] = []

    if max_windows > 0 and strategy == "uniform":
        # Reparte N ventanas a lo largo del vídeo sin crear archivos nuevos.
        usable_start = min(max(0.0, start0), max(0.0, duration - win))
        last_start = max(usable_start, duration - win)
        if max_windows == 1:
            starts = [usable_start]
        else:
            starts = np.linspace(usable_start, last_start, num=max_windows).tolist()
        for i, st in enumerate(starts):
            st = float(max(0.0, min(st, max(0.0, duration - min_remaining))))
            dur = float(min(win, max(min_remaining, duration - st)))
            windows.append({"idx": i, "start": st, "duration": dur, "tag": f"w{i:04d}_{int(st):06d}", "video_duration": duration})
        return windows

    t = max(0.0, start0)
    i = 0
    while t < duration:
        remaining = duration - t
        if remaining < min_remaining:
            break
        dur = min(win, remaining)
        windows.append({"idx": i, "start": float(t), "duration": float(dur), "tag": f"w{i:04d}_{int(t):06d}", "video_duration": duration})
        i += 1
        if max_windows > 0 and i >= max_windows:
            break
        t += hop
    return windows


def stratified_video_window_order(windows: List[Dict]) -> List[Dict]:
    """Prioriza inicio, centro y final antes de completar el resto.

    La parada temprana no debe concentrar todos los candidatos en los primeros
    minutos de una entrevista larga. Con tres o más ventanas se visitan primero
    tres zonas separadas y después se elige iterativamente la ventana más
    alejada de las ya visitadas.
    """
    ordered_input = list(windows)
    count = len(ordered_input)
    if count <= 2:
        return ordered_input

    selected_indices: List[int] = []
    for index in (0, int(round((count - 1) / 2.0)), count - 1):
        if index not in selected_indices:
            selected_indices.append(index)

    remaining = set(range(count)) - set(selected_indices)
    while remaining:
        next_index = max(
            remaining,
            key=lambda index: (min(abs(index - chosen) for chosen in selected_indices), -index),
        )
        selected_indices.append(next_index)
        remaining.remove(next_index)
    return [ordered_input[index] for index in selected_indices]


def stage_candidates(
    cfg,
    celebrities,
    limit_videos: int | None = None,
    resume: bool = False,
    force: bool = False,
    reprocess_all: bool = False,
    candidate_run_tag: str = "",
) -> Path:
    cfg.make_dirs()
    candidates_dir = Path(cfg.data_candidates)
    cand_audio_root = candidates_dir / "audio"
    candidates_csv = candidates_dir / "candidates.csv"
    processed_units_csv = candidates_dir / "processed_units.csv"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    cand_audio_root.mkdir(parents=True, exist_ok=True)

    selected_speaker_ids = {str(celeb.id) for celeb in celebrities}
    existing = pd.DataFrame()
    processed_units = pd.DataFrame()
    done_units = set()
    if processed_units_csv.exists():
        try:
            processed_units = pd.read_csv(processed_units_csv)
            if resume and "processing_unit_id" in processed_units:
                done_units.update(processed_units["processing_unit_id"].astype(str).unique())
        except Exception as exc:
            logging.warning("Could not read %s: %s", processed_units_csv, exc)
    if candidates_csv.exists() and (resume or reprocess_all or force):
        previous = pd.read_csv(candidates_csv)
        if force and not previous.empty and "speaker_id" in previous:
            existing = previous.loc[~previous["speaker_id"].astype(str).isin(selected_speaker_ids)].copy()
            logging.warning(
                "Selective force mode: replacing %s and keeping %d rows from other speakers",
                ", ".join(sorted(selected_speaker_ids)),
                len(existing),
            )
        elif not force:
            existing = previous
            pipeline_version = str(getattr(cfg, "pipeline_version", "2.1-syncnet-framewise"))
            existing_version = existing.get(
                "pipeline_version", pd.Series("legacy", index=existing.index)
            ).fillna("legacy").astype(str)
            obsolete_version = (
                existing["speaker_id"].astype(str).isin(selected_speaker_ids)
                & existing_version.ne(pipeline_version)
            ) if "speaker_id" in existing else pd.Series(False, index=existing.index)
            if reprocess_all and obsolete_version.any():
                logging.info(
                    "Removing %d candidates from a previous version to reprocess %s",
                    int(obsolete_version.sum()),
                    ", ".join(sorted(selected_speaker_ids)),
                )
                existing = existing.loc[~obsolete_version].copy()
            if reprocess_all and bool(getattr(cfg, "real_asd_required", True)) and "speaker_id" in existing:
                verified = pd.to_numeric(
                    existing.get("syncnet_verified", pd.Series(0, index=existing.index)),
                    errors="coerce",
                ).fillna(0)
                obsolete = existing["speaker_id"].astype(str).isin(selected_speaker_ids) & verified.lt(1)
                if obsolete.any():
                    logging.info(
                        "Removing %d old candidates without real SyncNet verification to reprocess %s",
                        int(obsolete.sum()),
                        ", ".join(sorted(selected_speaker_ids)),
                    )
                    existing = existing.loc[~obsolete].copy()
        if not existing.empty:
            if "processing_unit_id" in existing:
                done_units.update(existing["processing_unit_id"].astype(str).unique())
            elif "video_id" in existing:
                done_units.update(existing["video_id"].astype(str).unique())
        if resume and not reprocess_all and not force:
            logging.info("Resume enabled: %d units already had candidates", len(done_units))
        if reprocess_all:
            logging.info("Selective full reprocessing: %s", ", ".join(sorted(selected_speaker_ids)))

    # Inicializa PyTorch/SyncNet antes de que InsightFace ejecute la galería de
    # referencias. ONNX Runtime inicializa su propio contexto CUDA al hacer la
    # primera inferencia facial y, en algunos nodos, crear PyTorch después
    # provoca un CUDA driver error aunque queden más de 10 GiB libres.
    real_asd_verifier = SyncNetTrackVerifier(cfg)
    real_asd_verifier.validate_ready()

    detector = FaceDetector(cfg)
    tracker = FaceTracker(cfg)
    extractor = MediaExtractor(cfg)
    active_verifier = ActiveSpeakerVerifier(cfg)
    # Se construye una sola galería por ejecución (incluye identidades rivales),
    # no una vez por cada ventana de diez minutos.
    identity = IdentityManager(detector, celebrities, cfg)
    quality = QualityFilter(cfg)

    new_rows: List[Dict] = []
    video_items = list(iter_videos(celebrities))
    if limit_videos is not None:
        video_items = video_items[:int(limit_videos)]

    total_units_done = 0

    for celeb, video in tqdm(video_items, desc="Videos", unit="vid", disable=bool(getattr(cfg, "quiet_progress", True))):
        video_id = str(video.youtube_id)
        video_path = Path(cfg.download_path) / f"{video_id}.mp4"
        if not video_path.exists():
            logging.warning("Missing video %s for %s: %s", video_id, celeb.name, video_path)
            continue

        logging.info("\n%s", "=" * 72)
        logging.info("%s (%s) — %s", celeb.name, celeb.id, video_id)
        logging.info("%s", "=" * 72)

        try:
            video_windows = iter_video_windows(extractor, video_path, cfg)
            logging.info(
                "Long video: duration %.1f min | internal windows: %d | window=%.1f s | fps=%.2f",
                float(video_windows[0].get("video_duration", 0.0)) / 60.0 if video_windows else 0.0,
                len(video_windows),
                float(getattr(cfg, "long_video_window_seconds", 600)),
                float(getattr(cfg, "detect_fps", 2)),
            )
        except Exception as e:
            logging.exception("Could not calculate windows for %s: %s", video_id, e)
            continue

        video_rows: List[Dict] = []
        global_candidate_idx = 0

        early_stop_enabled = bool(getattr(cfg, "early_stop_candidates_enabled", True))
        early_stop_cap = int(getattr(cfg, "early_stop_candidates_per_video", 25))
        early_stop_min_windows = max(
            1, int(getattr(cfg, "early_stop_min_temporal_windows", 3))
        )
        existing_video_candidates = 0
        if not existing.empty and {"speaker_id", "video_id"}.issubset(existing.columns):
            existing_video_candidates = int((
                existing["speaker_id"].astype(str).eq(str(celeb.id))
                & existing["video_id"].astype(str).eq(video_id)
            ).sum())
        if early_stop_enabled and early_stop_cap > 0:
            video_windows = stratified_video_window_order(video_windows)
        required_temporal_windows = min(early_stop_min_windows, len(video_windows))
        processed_temporal_windows = 0
        early_stop_announced = False

        for vw in video_windows:
            total_video_candidates = existing_video_candidates + len(video_rows)
            if (
                early_stop_enabled
                and early_stop_cap > 0
                and total_video_candidates >= early_stop_cap
                and processed_temporal_windows >= required_temporal_windows
            ):
                logging.info(
                    "Early stopping %s: %d reliable candidates after %d temporal regions",
                    video_id,
                    total_video_candidates,
                    processed_temporal_windows,
                )
                early_stop_announced = True
                break

            frame_paths: List[Path] = []
            audio_path = None
            chunk_tag = str(vw["tag"])
            chunk_start = float(vw["start"])
            chunk_duration = float(vw["duration"])
            processing_unit_id = f"{video_id}::{chunk_tag}"

            if resume and processing_unit_id in done_units and not force and not reprocess_all:
                logging.info("Window already processed: %s", processing_unit_id)
                continue

            logging.info(
                "Window %s/%s: start=%.1fs end=%.1fs duration=%.1fs",
                int(vw["idx"]) + 1,
                len(video_windows),
                chunk_start,
                chunk_start + chunk_duration,
                chunk_duration,
            )
            processed_temporal_windows += 1

            unit_failed = False
            candidates_before_unit = len(video_rows)
            unit_candidate_budget: Optional[int] = None
            if early_stop_enabled and early_stop_cap > 0:
                remaining_candidates = max(
                    0,
                    early_stop_cap - existing_video_candidates - len(video_rows),
                )
                remaining_diversity_windows = max(
                    1, required_temporal_windows - processed_temporal_windows + 1
                )
                unit_candidate_budget = max(
                    1, int(math.ceil(remaining_candidates / remaining_diversity_windows))
                ) if remaining_candidates else 0
            try:
                audio_path, frame_paths = extractor.extract_window(video_path, video_id, chunk_start, chunk_duration, tag=chunk_tag)
                audio, sr = extractor.load_audio(audio_path)
                audio = normalize_audio(audio)

                tracks = tracker.track(frame_paths, detector, f"{video_id}_{chunk_tag}")
                tracks = identity.assign_identities(tracks) if tracks else []
                identified = [
                    t for t in tracks
                    if t.get("is_identified") and t.get("speaker_id") == celeb.id
                ]

                # Los frames se reducen para poder procesar entrevistas largas.
                # Un rostro en plano abierto puede quedar por debajo de 80 px y
                # no llegar jamás a ArcFace/SyncNet. Si la pasada normal no
                # encuentra al objetivo, repetimos solo esa ventana admitiendo
                # caras más pequeñas y tracks con huecos breves. Los umbrales
                # de identidad, margen, consistencia y SyncNet no cambian.
                if not identified and bool(getattr(cfg, "face_recovery_enabled", True)):
                    original_face_size = detector.min_face_size
                    original_max_lost = tracker.max_lost_frames
                    original_min_iou = tracker.min_association_iou
                    original_min_similarity = tracker.min_association_similarity
                    detector.min_face_size = min(
                        original_face_size,
                        int(getattr(cfg, "face_recovery_min_face_size", 48)),
                    )
                    tracker.max_lost_frames = max(
                        original_max_lost,
                        int(getattr(cfg, "face_recovery_max_lost_frames", 8)),
                    )
                    tracker.min_association_iou = min(
                        original_min_iou,
                        float(getattr(cfg, "face_recovery_min_association_iou", 0.08)),
                    )
                    tracker.min_association_similarity = max(
                        original_min_similarity,
                        float(getattr(cfg, "face_recovery_min_embedding_similarity", 0.40)),
                    )
                    logging.info(
                        "Adaptive face recovery in %s: minimum size %d -> %d px",
                        chunk_tag,
                        original_face_size,
                        detector.min_face_size,
                    )
                    try:
                        recovered_tracks = tracker.track(
                            frame_paths,
                            detector,
                            f"{video_id}_{chunk_tag}_recovery",
                        )
                    finally:
                        detector.min_face_size = original_face_size
                        tracker.max_lost_frames = original_max_lost
                        tracker.min_association_iou = original_min_iou
                        tracker.min_association_similarity = original_min_similarity
                    recovered_tracks = (
                        identity.assign_identities(recovered_tracks)
                        if recovered_tracks else []
                    )
                    recovered_identified = [
                        t for t in recovered_tracks
                        if t.get("is_identified") and t.get("speaker_id") == celeb.id
                    ]
                    logging.info(
                        "Face recovery %s: %d valid tracks, %d for the target",
                        chunk_tag,
                        len(recovered_tracks),
                        len(recovered_identified),
                    )
                    if recovered_identified:
                        tracks = recovered_tracks
                        identified = recovered_identified

                if not tracks:
                    logging.info("No face tracks in window %s", processing_unit_id)
                    continue

                logging.info("Tracks identified as %s in %s: %d/%d", celeb.id, chunk_tag, len(identified), len(tracks))
                face_scores = [float(t.get("face_conf", 0.0)) for t in tracks]
                if face_scores:
                    logging.info(
                        "Face similarity in %s: median=%.3f maximum=%.3f threshold=%.3f",
                        chunk_tag,
                        float(np.median(face_scores)),
                        float(np.max(face_scores)),
                        float(getattr(cfg, "face_template_threshold", 0.60)),
                    )
                if not identified:
                    continue

                tracks_by_id = {
                    str(track.get("track_id", track.get("id", ""))): track
                    for track in identified
                }
                if bool(getattr(cfg, "syncnet_framewise_enabled", True)):
                    windows = tracks_to_syncnet_probe_windows(identified, cfg)
                    logging.info(
                        "Exhaustive probes for framewise SyncNet diarization in %s: %d",
                        chunk_tag,
                        len(windows),
                    )
                else:
                    active_segments = active_verifier.detect_active_speaker_segments(
                        frame_paths=frame_paths,
                        audio=audio,
                        audio_sr=sr,
                        face_tracks=identified,
                        min_active_duration=float(getattr(cfg, "segment_min_duration", 2.0)),
                        step_seconds=float(getattr(cfg, "active_step_seconds", 0.5)),
                    )
                    logging.info("Proposed active regions in %s: %d", chunk_tag, len(active_segments))
                    windows = active_segments_to_windows(active_segments, cfg)
                logging.info("SyncNet windows in %s: %d", chunk_tag, len(windows))
                if not windows:
                    continue

                seen_hashes = set(row.get("sha1") for row in video_rows if row.get("sha1"))
                quality_rejected_count = 0
                syncnet_checked_count = 0
                syncnet_rejected_count = 0
                syncnet_segment_count = 0
                syncnet_confidences: List[float] = []
                unit_budget_reached = False
                for probe_index, win in enumerate(windows):
                    track_id = str(win.get("track_id", ""))
                    target_track = tracks_by_id.get(track_id)
                    if target_track is None:
                        logging.debug("Probe without an associated face track: %s", track_id)
                        continue
                    sync_result = real_asd_verifier.verify_candidate(
                        video_path=video_path,
                        global_start=chunk_start + float(win["start"]),
                        duration=float(win["end"]) - float(win["start"]),
                        local_start=float(win["start"]),
                        track=target_track,
                        frame_paths=frame_paths,
                        detect_fps=float(getattr(cfg, "detect_fps", 5.0)),
                        key=f"{celeb.id}_{video_id}_{chunk_tag}_probe_{probe_index:05d}",
                    )
                    syncnet_checked_count += 1
                    syncnet_confidences.append(float(sync_result.confidence))
                    if not sync_result.accepted:
                        syncnet_rejected_count += 1
                        logging.debug(
                            "Candidate rejected by SyncNet in %s: confidence=%.3f offset=%s reason=%s",
                            chunk_tag,
                            sync_result.confidence,
                            sync_result.offset_frames,
                            sync_result.reason,
                        )
                        continue
                    for segment_index, sync_segment in enumerate(sync_result.segments):
                        local_start = float(win["start"]) + float(sync_segment.start)
                        local_end = float(win["start"]) + float(sync_segment.end)
                        chunk = slice_audio(audio, sr, local_start, local_end)
                        q = quality.check_audio(chunk, sr)
                        if not q.ok:
                            quality_rejected_count += 1
                            logging.debug("SyncNet subsegment rejected for quality in %s: %s", chunk_tag, q.reason)
                            continue

                        # Si otro rostro visible también sincroniza en este
                        # subsegmento exacto, se rechaza como habla ambigua.
                        competitor_confidence = -99.0
                        competitor_count = 0
                        ambiguous = False
                        if bool(getattr(cfg, "syncnet_competitor_check", True)):
                            competitors = []
                            detect_fps = float(getattr(cfg, "detect_fps", 5.0))
                            for other_track in tracks:
                                other_id = str(other_track.get("track_id", other_track.get("id", "")))
                                if other_id == track_id:
                                    continue
                                coverage = _track_candidate_coverage(
                                    other_track, local_start, local_end, detect_fps
                                )
                                if coverage >= float(getattr(cfg, "syncnet_competitor_min_coverage", 0.65)):
                                    competitors.append((coverage, other_track))
                            competitors.sort(key=lambda item: item[0], reverse=True)
                            for competitor_index, (_, other_track) in enumerate(
                                competitors[:int(getattr(cfg, "syncnet_max_competitors", 2))]
                            ):
                                try:
                                    competitor = real_asd_verifier.verify_candidate(
                                        video_path=video_path,
                                        global_start=chunk_start + local_start,
                                        duration=local_end - local_start,
                                        local_start=local_start,
                                        track=other_track,
                                        frame_paths=frame_paths,
                                        detect_fps=detect_fps,
                                        key=(f"{celeb.id}_{video_id}_{chunk_tag}_probe_{probe_index:05d}_"
                                             f"segment_{segment_index:03d}_competitor_{competitor_index}"),
                                    )
                                except Exception as exc:
                                    logging.debug("Could not check a competing track: %s", exc)
                                    continue
                                competitor_count += 1
                                if competitor.segments:
                                    competitor_confidence = max(
                                        competitor_confidence,
                                        max(float(item.confidence) for item in competitor.segments),
                                    )
                                if competitor.accepted:
                                    ambiguous = True
                                    break
                        if ambiguous:
                            logging.debug(
                                "Ambiguous subsegment: another face is also synchronized (target=%.3f, competitor=%.3f)",
                                sync_segment.confidence,
                                competitor_confidence,
                            )
                            continue

                        run_suffix = f"_{candidate_run_tag}" if candidate_run_tag else ""
                        out_name = f"{video_id}_{chunk_tag}_{global_candidate_idx:05d}{run_suffix}.wav"
                        out_path = cand_audio_root / celeb.id / video_id / out_name
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(str(out_path), chunk, sr, subtype="PCM_16")
                        h = sha1_file(out_path)
                        if h in seen_hashes:
                            out_path.unlink(missing_ok=True)
                            continue
                        seen_hashes.add(h)

                        global_start = chunk_start + local_start
                        global_end = chunk_start + local_end
                        row = {
                        "pipeline_version": str(getattr(cfg, "pipeline_version", "2.1-syncnet-framewise")),
                        "speaker_id": celeb.id,
                        "speaker_name": celeb.name,
                        "gender": getattr(celeb, "gender", "unknown"),
                        "category": getattr(celeb, "category", "unknown"),
                        "region": getattr(celeb, "region", "unknown"),
                        "video_id": video_id,
                        "processing_unit_id": processing_unit_id,
                        "chunk_tag": chunk_tag,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_start + chunk_duration,
                        "url": getattr(video, "url", ""),
                        "candidate_idx": global_candidate_idx,
                        "candidate_path": str(out_path),
                        "start": float(global_start),
                        "end": float(global_end),
                        "local_start": float(local_start),
                        "local_end": float(local_end),
                        "duration": float(q.duration),
                        "active_confidence": float(sync_segment.high_ratio),
                        "audio_score": float(q.vad_ratio),
                        "visual_score": float(win.get("visual_score", 0.0)),
                        "face_conf": float(win.get("face_conf", 0.0)) if "face_conf" in win else None,
                        "face_other_conf": float(win.get("face_other_conf", -1.0)),
                        "face_margin": float(win.get("face_margin", 0.0)),
                        "face_track_consistency": float(win.get("face_track_consistency", 0.0)),
                        "vad_ratio": float(q.vad_ratio),
                        "snr_db": float(q.snr_db),
                        "rms": float(q.rms),
                        "peak": float(q.peak),
                        "clipping_ratio": float(q.clipping_ratio),
                        "quality_score": float(q.quality_score),
                        "quality_reason": q.reason,
                        "syncnet_confidence": float(sync_segment.confidence),
                        "syncnet_global_confidence": float(sync_result.confidence),
                        "syncnet_framewise_high_ratio": float(sync_segment.high_ratio),
                        "syncnet_framewise_threshold": float(getattr(cfg, "syncnet_framewise_threshold", 4.0)),
                        "syncnet_offset_frames": int(sync_segment.offset_frames),
                        "syncnet_min_distance": float(sync_result.min_distance),
                        "syncnet_verified": 1,
                        "syncnet_competitor_confidence": float(competitor_confidence),
                        "syncnet_competitors_checked": int(competitor_count),
                        "syncnet_unambiguous": 1,
                        "candidate_profile": str(getattr(cfg, "candidate_profile", "standard")),
                        "active_speaker_method": "syncnet_framewise_target_track",
                        "requires_manual_validation": bool(
                            win.get("requires_manual_validation", False)
                            or str(getattr(cfg, "candidate_profile", "standard")) == "relaxed"
                        ),
                        "approved": 0,
                        "sha1": h,
                        }
                        row["rank_score"] = (
                            0.35 * row["quality_score"]
                            + 0.35 * min(max((row["syncnet_confidence"] - 4.0) / 4.0, 0.0), 1.0)
                            + 0.20 * row["active_confidence"]
                            + 0.10 * min(max((row["snr_db"] - float(getattr(cfg, "min_snr_db", 8.0))) / 20.0, 0.0), 1.0)
                        )
                        video_rows.append(row)
                        global_candidate_idx += 1
                        syncnet_segment_count += 1
                        if (
                            unit_candidate_budget is not None
                            and len(video_rows) - candidates_before_unit >= unit_candidate_budget
                        ):
                            unit_budget_reached = True
                            break
                    if unit_budget_reached:
                        logging.info(
                            "Time budget reached in %s: +%d reliable candidates",
                            chunk_tag,
                            len(video_rows) - candidates_before_unit,
                        )
                        break

                if syncnet_checked_count:
                    logging.info(
                        "Framewise SyncNet %s: probes=%d with_regions=%d without_regions=%d subsegments=%d median_global_confidence=%.3f | quality_rejected=%d",
                        chunk_tag,
                        syncnet_checked_count,
                        syncnet_checked_count - syncnet_rejected_count,
                        syncnet_rejected_count,
                        syncnet_segment_count,
                        float(np.median(syncnet_confidences)),
                        quality_rejected_count,
                    )
                logging.info("Candidates accumulated in %s through %s: %d", video_id, chunk_tag, len(video_rows))
                total_units_done += 1

                # Guardado incremental por ventana para no perder trabajo si el job se corta.
                tmp_new = new_rows + video_rows
                all_df = pd.concat([existing, pd.DataFrame(tmp_new)], ignore_index=True) if not existing.empty else pd.DataFrame(tmp_new)
                if not all_df.empty:
                    write_speaker_scoped_csv(
                        candidates_csv,
                        all_df,
                        selected_speaker_ids,
                        dedup_subset=["speaker_id", "video_id", "sha1", "candidate_path"],
                    )

            except Exception as e:
                unit_failed = True
                logging.exception("Error processing window %s: %s", processing_unit_id, e)
            except BaseException:
                unit_failed = True
                raise
            finally:
                extractor.cleanup_frames(frame_paths)
                if bool(getattr(cfg, "cleanup_window_audio", True)) and audio_path is not None:
                    try:
                        Path(audio_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                if not unit_failed:
                    unit_row = pd.DataFrame([{
                        "processing_unit_id": processing_unit_id,
                        "speaker_id": str(celeb.id),
                        "video_id": video_id,
                        "chunk_tag": chunk_tag,
                        "candidate_profile": str(getattr(cfg, "candidate_profile", "standard")),
                        "n_candidates": int(max(0, len(video_rows) - candidates_before_unit)),
                        "processed_at": dt.datetime.now().isoformat(timespec="seconds"),
                    }])
                    processed_units = pd.concat([processed_units, unit_row], ignore_index=True)
                    processed_units = processed_units.drop_duplicates(
                        subset=["processing_unit_id"], keep="last"
                    )
                    write_speaker_scoped_csv(
                        processed_units_csv,
                        processed_units,
                        selected_speaker_ids,
                        dedup_subset=["processing_unit_id"],
                    )
                    done_units.add(processing_unit_id)

        total_video_candidates = existing_video_candidates + len(video_rows)
        if (
            early_stop_enabled
            and early_stop_cap > 0
            and total_video_candidates >= early_stop_cap
            and not early_stop_announced
        ):
            logging.info(
                "Early stopping %s: %d reliable candidates after %d temporal regions",
                video_id,
                total_video_candidates,
                processed_temporal_windows,
            )

        logging.info("Valid candidates in video %s: %d", video_id, len(video_rows))
        new_rows.extend(video_rows)

    final_df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if not existing.empty else pd.DataFrame(new_rows)
    if not final_df.empty:
        if "sha1" in final_df:
            digest = final_df["sha1"].fillna("").astype(str).str.strip()
            fallback = final_df.get("candidate_path", pd.Series("", index=final_df.index)).fillna("").astype(str)
            final_df["_dedup_key"] = digest.where(digest.ne(""), fallback)
            final_df = final_df.drop_duplicates(subset=["speaker_id", "video_id", "_dedup_key"])
            final_df = final_df.drop(columns=["_dedup_key"])
        else:
            keys = ["speaker_id", "video_id", "candidate_idx", "candidate_path"]
            final_df = final_df.drop_duplicates(subset=[k for k in keys if k in final_df.columns])
        write_speaker_scoped_csv(
            candidates_csv,
            final_df,
            selected_speaker_ids,
            dedup_subset=["speaker_id", "video_id", "sha1", "candidate_path"],
        )
    logging.info("Candidates saved: %s (%d rows, %d processed windows)", candidates_csv, len(final_df), total_units_done)
    return candidates_csv

def _manual_rejection_keys(cfg, candidates: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Carga rechazos persistentes y devuelve SHA-1 y rutas de candidatos.

    ``manual_rejections.csv`` puede contener ``sha1``, ``candidate_path`` o el
    ``utterance_id``/nombre WAV exportado. Este último se resuelve usando la
    metadata de la exportación anterior, de modo que el rechazo queda asociado
    al candidato original aunque al reexportar cambie su índice.
    """
    configured = getattr(cfg, "manual_rejections_file", "manual_rejections.csv")
    rejection_path = Path(str(configured)).expanduser()
    if not rejection_path.is_absolute():
        rejection_path = Path(cfg.project_dir) / rejection_path
    if not rejection_path.exists() or rejection_path.stat().st_size == 0:
        return set(), set()

    rejected = pd.read_csv(rejection_path, dtype=str).fillna("")
    rejected_sha1: set[str] = set()
    rejected_paths: set[str] = set()

    if "sha1" in rejected:
        rejected_sha1.update(v.strip() for v in rejected["sha1"] if v.strip())
    if "candidate_path" in rejected:
        rejected_paths.update(str(Path(v.strip()).expanduser()) for v in rejected["candidate_path"] if v.strip())

    utterance_values: set[str] = set()
    for column in ("utterance_id", "audio_filename"):
        if column in rejected:
            utterance_values.update(Path(v.strip()).stem for v in rejected[column] if v.strip())

    if utterance_values:
        metadata_path = Path(cfg.data_processed) / "metadata.csv"
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path, dtype=str).fillna("")
            metadata_ids = pd.Series("", index=metadata.index, dtype=str)
            if "utterance_id" in metadata:
                metadata_ids = metadata["utterance_id"].map(lambda v: Path(v).stem)
            elif "audio_path" in metadata:
                metadata_ids = metadata["audio_path"].map(lambda v: Path(v).stem)
            matched = metadata.loc[metadata_ids.isin(utterance_values)]
            if "sha1" in matched:
                rejected_sha1.update(v.strip() for v in matched["sha1"] if v.strip())
            if "candidate_path" in matched:
                rejected_paths.update(str(Path(v.strip()).expanduser()) for v in matched["candidate_path"] if v.strip())
            unresolved = utterance_values - set(metadata_ids.loc[metadata_ids.isin(utterance_values)])
            if unresolved:
                logging.warning("Rejections not found in metadata.csv: %s", sorted(unresolved))
        else:
            logging.warning(
                "Some rejections use utterance_id, but %s does not exist yet; use sha1/candidate_path or keep the previous metadata",
                metadata_path,
            )

    # Permite que un SHA-1 resuelto por metadata bloquee todas sus copias.
    if rejected_paths and "candidate_path" in candidates and "sha1" in candidates:
        candidate_paths = candidates["candidate_path"].fillna("").astype(str)
        matched_candidates = candidates.loc[candidate_paths.isin(rejected_paths)]
        rejected_sha1.update(
            str(v).strip() for v in matched_candidates["sha1"]
            if str(v).strip() and str(v).strip().lower() != "nan"
        )

    logging.info(
        "Manual rejections loaded from %s: %d SHA-1 hashes, %d paths",
        rejection_path,
        len(rejected_sha1),
        len(rejected_paths),
    )
    return rejected_sha1, rejected_paths


def stage_approve_auto(
    cfg,
    allowed_speaker_ids: Optional[Iterable[str]] = None,
) -> Path:
    candidates_csv = Path(cfg.data_candidates) / "candidates.csv"
    approved_csv = Path(cfg.data_candidates) / "approved_segments.csv"
    if not candidates_csv.exists():
        raise FileNotFoundError(f"{candidates_csv} does not exist. Run --stage candidates first")
    df = pd.read_csv(candidates_csv)
    if df.empty:
        raise RuntimeError("candidates.csv is empty")

    # Un speaker_id puede reutilizarse cuando se sustituye una identidad (por
    # ejemplo, Alaska por Carolina Yuste). En ese caso no deben sobrevivir
    # candidatos ni registros de vídeos que ya no figuran en el manifiesto
    # actual. La lista completa se adjunta al cfg en main antes de aplicar el
    # filtro --speaker-id.
    configured_videos = getattr(cfg, "configured_video_ids_by_speaker", {})
    if configured_videos and {"speaker_id", "video_id"}.issubset(df.columns):
        configured_pairs = {
            (str(speaker_id), str(video_id))
            for speaker_id, video_ids in configured_videos.items()
            for video_id in video_ids
        }
        configured_rows = pd.Series(
            [
                (str(speaker_id), str(video_id)) in configured_pairs
                for speaker_id, video_id in zip(df["speaker_id"], df["video_id"])
            ],
            index=df.index,
            dtype=bool,
        )
        obsolete_count = int((~configured_rows).sum())
        if obsolete_count:
            df = df.loc[configured_rows].copy()
            df.to_csv(candidates_csv, index=False)
            logging.info(
                "Removed %d candidates from videos absent from the current manifest",
                obsolete_count,
            )
            if df.empty:
                raise RuntimeError("No candidates remain for the currently configured videos")
    if allowed_speaker_ids is not None:
        allowed_ids = {str(speaker_id) for speaker_id in allowed_speaker_ids}
        configured = df["speaker_id"].astype(str).isin(allowed_ids)
        excluded_rows = int((~configured).sum())
        if excluded_rows:
            logging.info(
                "Excluded %d candidates from speakers absent from the current configuration",
                excluded_rows,
            )
        df = df.loc[configured].copy()
        if df.empty:
            raise RuntimeError("No candidates remain for the configured speakers")

    rejected_sha1, rejected_paths = _manual_rejection_keys(cfg, df)
    candidate_sha1 = df.get("sha1", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    candidate_paths = df["candidate_path"].fillna("").astype(str).map(
        lambda value: str(Path(value).expanduser()) if value else ""
    )
    df["manual_rejected"] = (
        candidate_sha1.isin(rejected_sha1) | candidate_paths.isin(rejected_paths)
    ).astype(int)

    target_per_speaker = int(
        getattr(cfg, "target_clips_per_speaker", getattr(cfg, "min_clips_per_speaker", 50))
    )
    max_per_video = int(getattr(cfg, "segments_per_video", 20))
    min_per_video = int(getattr(cfg, "min_selected_per_video", 5))
    if target_per_speaker < 1:
        raise ValueError("target_clips_per_speaker must be >= 1")
    if max_per_video < 1:
        raise ValueError("segments_per_video must be >= 1")
    if min_per_video < 0:
        raise ValueError("min_selected_per_video must be >= 0")
    if min_per_video > max_per_video:
        raise ValueError("min_selected_per_video cannot exceed segments_per_video")

    def numeric(column: str, default: float = 0.0) -> pd.Series:
        if column not in df:
            return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(df[column], errors="coerce").fillna(default).astype(float)

    profiles = (
        df["candidate_profile"].fillna("standard").astype(str).str.lower()
        if "candidate_profile" in df
        else pd.Series("standard", index=df.index)
    )
    relaxed = profiles.eq("relaxed")
    standard_face_threshold = float(getattr(cfg, "face_template_threshold", 0.60))
    relaxed_face_threshold = float(getattr(cfg, "relaxed_face_template_threshold", 0.50))
    face_threshold = pd.Series(
        np.where(relaxed, relaxed_face_threshold, standard_face_threshold),
        index=df.index,
        dtype=float,
    )
    face_conf = numeric("face_conf", -1.0)
    face_margin = numeric("face_margin", -1.0)
    face_consistency = numeric("face_track_consistency", 0.0)
    quality = numeric("quality_score").clip(0.0, 1.0)
    active = numeric("active_confidence").clip(0.0, 1.0)
    visual = numeric("visual_score").clip(0.0, 1.0)
    snr = numeric("snr_db", -99.0)
    face_denominator = (0.80 - face_threshold).clip(lower=0.01)
    face_score = ((face_conf - face_threshold) / face_denominator).clip(0.0, 1.0)
    min_snr = pd.Series(
        np.where(relaxed, 5.0, float(getattr(cfg, "min_snr_db", 5.0))),
        index=df.index,
        dtype=float,
    )
    min_active = pd.Series(
        np.where(relaxed, 0.40, float(getattr(cfg, "syncnet_threshold", 0.45))),
        index=df.index,
        dtype=float,
    )
    min_audio = pd.Series(0.20, index=df.index, dtype=float)
    min_visual = pd.Series(0.04, index=df.index, dtype=float)
    min_speech = pd.Series(0.45, index=df.index, dtype=float)
    audio_score = numeric("audio_score").clip(0.0, 1.0)
    vad_ratio = numeric("vad_ratio").clip(0.0, 1.0)
    sync_confidence = numeric("syncnet_confidence", -99.0)
    sync_offset = numeric("syncnet_offset_frames", 999.0).abs()
    sync_verified = numeric("syncnet_verified", 0.0).ge(1.0)
    sync_unambiguous = numeric("syncnet_unambiguous", 0.0).ge(1.0)
    min_sync_confidence = float(getattr(cfg, "syncnet_confidence_threshold", 4.0))
    max_sync_offset = int(getattr(cfg, "syncnet_max_abs_offset_frames", 3))
    sync_score = ((sync_confidence - min_sync_confidence) / 7.0).clip(0.0, 1.0)
    require_unambiguous = bool(getattr(cfg, "syncnet_competitor_check", True))
    real_asd_gate = (
        sync_verified
        & sync_confidence.ge(min_sync_confidence)
        & sync_offset.le(max_sync_offset)
        & (sync_unambiguous if require_unambiguous else True)
        if bool(getattr(cfg, "real_asd_required", True))
        else pd.Series(True, index=df.index)
    )
    snr_score = ((snr - min_snr) / 20.0).clip(0.0, 1.0)

    df["approval_face_threshold"] = face_threshold
    df["approval_face_score"] = face_score
    df["approval_snr_score"] = snr_score
    df["approval_syncnet_score"] = sync_score
    df["balanced_rank_score"] = (
        0.30 * face_score + 0.20 * quality + 0.35 * sync_score
        + 0.10 * active + 0.05 * snr_score
    )
    df["approved"] = 0
    df["approved_index"] = -1
    df["selection_rank"] = -1
    file_exists = df["candidate_path"].fillna("").astype(str).map(lambda value: Path(value).is_file())
    manually_rejected = df["manual_rejected"].eq(1)
    expected_pipeline_version = str(getattr(cfg, "pipeline_version", "2.1-syncnet-framewise"))
    version_matches = df.get(
        "pipeline_version", pd.Series("legacy", index=df.index)
    ).fillna("legacy").astype(str).eq(expected_pipeline_version)
    # Con SyncNet real, energía y movimiento de boca son solo un generador de
    # propuestas. No deben vetar un clip que ya superó el modelo audiovisual.
    proposal_gate = (
        pd.Series(True, index=df.index)
        if bool(getattr(cfg, "real_asd_required", True))
        else active.ge(min_active) & audio_score.ge(min_audio) & visual.ge(min_visual)
    )
    eligible = (
        file_exists
        & version_matches
        & ~manually_rejected
        & face_conf.ge(face_threshold)
        & face_margin.ge(float(getattr(cfg, "face_template_margin", 0.08)))
        & face_consistency.ge(float(getattr(cfg, "face_min_track_consistency", 0.75)))
        & proposal_gate
        & vad_ratio.ge(min_speech)
        & snr.ge(min_snr)
        & real_asd_gate
    )

    # ``candidates.csv`` no contiene filas para los vídeos que produjeron cero
    # clips. ``processed_units.csv`` permite incluirlos en el informe de cuota
    # mínima y evita que un vídeo fallido desaparezca silenciosamente.
    processed_units_csv = Path(cfg.data_candidates) / "processed_units.csv"
    processed_videos_by_speaker: Dict[str, List[str]] = {}
    if processed_units_csv.exists():
        try:
            processed_units = pd.read_csv(processed_units_csv)
            if {"speaker_id", "video_id"}.issubset(processed_units.columns):
                if configured_videos:
                    configured_pairs = {
                        (str(speaker_id), str(video_id))
                        for speaker_id, video_ids in configured_videos.items()
                        for video_id in video_ids
                    }
                    configured_units = pd.Series(
                        [
                            (str(speaker_id), str(video_id)) in configured_pairs
                            for speaker_id, video_id in zip(
                                processed_units["speaker_id"], processed_units["video_id"]
                            )
                        ],
                        index=processed_units.index,
                        dtype=bool,
                    )
                    obsolete_units = int((~configured_units).sum())
                    if obsolete_units:
                        processed_units = processed_units.loc[configured_units].copy()
                        processed_units.to_csv(processed_units_csv, index=False)
                        logging.info(
                            "Removed %d processed units from videos absent "
                            "from the current manifest",
                            obsolete_units,
                        )
                for processed_speaker, units in processed_units.groupby("speaker_id"):
                    processed_videos_by_speaker[str(processed_speaker)] = sorted(
                        units["video_id"].dropna().astype(str).unique().tolist()
                    )
        except Exception as exc:
            logging.warning("Could not read %s for the per-video distribution: %s", processed_units_csv, exc)

    selected_idx: List[int] = []
    for speaker_id, group in df.loc[eligible].groupby("speaker_id", sort=True):
        ordered = group.sort_values(
            ["balanced_rank_score", "face_conf", "quality_score", "active_confidence"],
            ascending=[False, False, False, False],
        )
        seen_hashes = set()
        video_counts: Dict[str, int] = {}
        video_intervals: Dict[str, List[tuple[float, float]]] = {}
        speaker_selected: List[int] = []

        def try_select(idx, row, enforce_video_cap: bool) -> bool:
            if len(speaker_selected) >= target_per_speaker or idx in speaker_selected:
                return False
            video_id = str(row.get("video_id", ""))
            if enforce_video_cap and video_counts.get(video_id, 0) >= max_per_video:
                return False
            digest = str(row.get("sha1", "")).strip()
            if not digest or digest.lower() == "nan":
                digest = str(row.get("candidate_path", ""))
            if digest in seen_hashes:
                return False
            start = float(row.get("start", -1.0))
            end = float(row.get("end", -1.0))
            max_temporal_iou = float(getattr(cfg, "max_selected_temporal_iou", 0.05))
            if end > start:
                for previous_start, previous_end in video_intervals.get(video_id, []):
                    intersection = max(0.0, min(end, previous_end) - max(start, previous_start))
                    union = max(end, previous_end) - min(start, previous_start)
                    if union > 0 and intersection / union > max_temporal_iou:
                        return False
            seen_hashes.add(digest)
            speaker_selected.append(idx)
            video_counts[video_id] = video_counts.get(video_id, 0) + 1
            video_intervals.setdefault(video_id, []).append((start, end))
            return True

        # Primera pasada: reserva los mejores clips de cada fuente. Esta cuota
        # nunca salta ningún filtro; un vídeo con menos candidatos elegibles
        # aporta todos los que tenga y queda señalado en el log.
        eligible_video_values = list(ordered["video_id"].fillna("").astype(str).unique())
        eligible_videos = list(dict.fromkeys(
            processed_videos_by_speaker.get(str(speaker_id), []) + eligible_video_values
        ))
        if min_per_video * len(eligible_videos) > target_per_speaker:
            raise RuntimeError(
                f"Cannot reserve {min_per_video} clips from each of "
                f"{len(eligible_videos)} videos within a quota of {target_per_speaker}"
            )
        video_shortfalls: Dict[str, int] = {}
        if min_per_video:
            for video_id in eligible_videos:
                video_group = ordered.loc[ordered["video_id"].fillna("").astype(str).eq(video_id)]
                for idx, row in video_group.iterrows():
                    try_select(idx, row, enforce_video_cap=True)
                    if video_counts.get(video_id, 0) >= min_per_video:
                        break
                selected_from_video = video_counts.get(video_id, 0)
                if selected_from_video < min_per_video:
                    video_shortfalls[video_id] = selected_from_video

        # Segunda pasada: completa con los mejores del ranking global,
        # independientemente del vídeo, manteniendo inicialmente el máximo por
        # fuente para evitar que una única entrevista monopolice el dataset.
        for idx, row in ordered.iterrows():
            try_select(idx, row, enforce_video_cap=True)
            if len(speaker_selected) >= target_per_speaker:
                break

        # El desbordamiento reduce la diversidad y puede llenar el dataset con
        # un único programa. Solo se permite si se solicita explícitamente.
        if len(speaker_selected) < target_per_speaker and bool(getattr(cfg, "allow_video_cap_overflow", False)):
            for idx, row in ordered.iterrows():
                try_select(idx, row, enforce_video_cap=False)
                if len(speaker_selected) >= target_per_speaker:
                    break

        for rank, idx in enumerate(speaker_selected):
            df.loc[idx, "selection_rank"] = rank
        selected_idx.extend(speaker_selected)
        logging.info(
            "Balanced selection %s: %d/%d clips from %d videos",
            speaker_id,
            len(speaker_selected),
            target_per_speaker,
            len(video_counts),
        )
        logging.info(
            "Per-video distribution %s: %s",
            speaker_id,
            ", ".join(f"{video_id}={count}" for video_id, count in sorted(video_counts.items())),
        )
        if video_shortfalls:
            logging.warning(
                "Videos with fewer than %d valid candidates for %s: %s",
                min_per_video,
                speaker_id,
                ", ".join(
                    f"{video_id}={count}/{min_per_video}"
                    for video_id, count in sorted(video_shortfalls.items())
                ),
            )

    if (~file_exists).any():
        logging.warning("%d candidates without WAV files were ignored", int((~file_exists).sum()))
    if manually_rejected.any():
        logging.info("%d manually rejected candidates were excluded", int(manually_rejected.sum()))
    if (~version_matches).any():
        logging.info("%d candidates from previous versions were excluded", int((~version_matches).sum()))
    if bool(getattr(cfg, "real_asd_required", True)):
        syncnet_failed = file_exists & ~manually_rejected & ~real_asd_gate
        logging.info("Final SyncNet result: %d accepted, %d rejected", int(real_asd_gate.sum()), int(syncnet_failed.sum()))
    quality_rejected = file_exists & ~manually_rejected & ~eligible
    if quality_rejected.any():
        logging.info("%d candidates were excluded by the final audio-visual thresholds", int(quality_rejected.sum()))

    df.loc[selected_idx, "approved"] = 1
    selected_df = df.loc[selected_idx].sort_values(
        ["speaker_id", "video_id", "selection_rank"]
    )
    for (_, _), group in selected_df.groupby(["speaker_id", "video_id"], sort=True):
        for approved_index, idx in enumerate(group.index):
            df.loc[idx, "approved_index"] = approved_index

    df.to_csv(approved_csv, index=False)
    logging.info(
        "Balanced selection: %d clips (maximum %d per speaker) -> %s",
        len(selected_idx),
        target_per_speaker,
        approved_csv,
    )
    return approved_csv


def _approved_speaker_coverage(cfg, speaker_ids: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Cuenta solo aprobados cuyo fichero de audio existe realmente."""
    approved_csv = Path(cfg.data_candidates) / "approved_segments.csv"
    coverage = {
        str(speaker_id): {"n_clips": 0, "duration_sec": 0.0, "n_videos": 0, "missing_files": 0}
        for speaker_id in speaker_ids
    }
    if not approved_csv.exists():
        return coverage

    df = pd.read_csv(approved_csv)
    if df.empty or "speaker_id" not in df or "candidate_path" not in df:
        return coverage

    if "approved" in df:
        approved = pd.to_numeric(df["approved"], errors="coerce").fillna(0).astype(int)
        df = df.loc[approved == 1].copy()
    df["speaker_id"] = df["speaker_id"].astype(str)
    df["_file_exists"] = df["candidate_path"].fillna("").astype(str).map(lambda value: Path(value).is_file())
    if "duration" in df:
        df["_duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        df["_duration"] = 0.0

    for speaker_id, group in df.groupby("speaker_id"):
        if speaker_id not in coverage:
            continue
        valid = group.loc[group["_file_exists"]]
        coverage[speaker_id] = {
            "n_clips": int(len(valid)),
            "duration_sec": float(valid["_duration"].sum()),
            "n_videos": int(valid["video_id"].nunique()) if "video_id" in valid else 0,
            "missing_files": int((~group["_file_exists"]).sum()),
        }
    return coverage


def stage_ensure_speaker_coverage(
    cfg,
    celebrities,
    min_clips: int | None = None,
    min_duration_sec: float | None = None,
    max_attempts: int | None = None,
    relaxed_retry: bool | None = None,
) -> Path:
    """Reintenta únicamente hablantes que no alcanzan la cobertura requerida.

    Cada reintento conserva todos los candidatos existentes, pero vuelve a
    procesar todas las ventanas del hablante insuficiente. Los WAV nuevos usan
    nombres separados y el CSV se deduplica por SHA-1.
    """
    min_clips = int(
        min_clips if min_clips is not None
        else getattr(cfg, "min_clips_per_speaker", getattr(cfg, "min_utterances_per_speaker_report", 50))
    )
    min_duration_sec = float(
        min_duration_sec if min_duration_sec is not None
        else getattr(cfg, "min_duration_per_speaker_sec", 0.0)
    )
    max_attempts = int(
        max_attempts if max_attempts is not None
        else getattr(cfg, "max_speaker_retry_attempts", 1)
    )
    relaxed_retry = bool(
        relaxed_retry if relaxed_retry is not None
        else getattr(cfg, "speaker_retry_relaxed", True)
    )
    if min_clips < 1:
        raise ValueError("min_clips_per_speaker must be >= 1")
    if min_duration_sec < 0:
        raise ValueError("min_duration_per_speaker_sec must be >= 0")
    if max_attempts < 0:
        raise ValueError("max_speaker_retry_attempts must be >= 0")

    logging.info(
        "Required coverage: clips>=%d duration>=%.1fs | retries=%d | relaxed=%s",
        min_clips,
        min_duration_sec,
        max_attempts,
        relaxed_retry,
    )

    # En un dataset nuevo no habrá candidatos todavía. En uno reanudado se
    # cuenta primero lo ya existente, sin recorrer otra vez todos los vídeos.
    candidates_csv = Path(cfg.data_candidates) / "candidates.csv"
    if not candidates_csv.exists():
        logging.info("candidates.csv does not exist; running the complete first pass")
        stage_candidates(cfg, celebrities, resume=True, force=False)

    # La decisión siempre se toma sobre la selección aprobada persistente.
    speaker_ids = [str(celeb.id) for celeb in celebrities]
    stage_approve_auto(cfg, speaker_ids)
    initial = _approved_speaker_coverage(cfg, speaker_ids)
    report_rows: List[Dict] = []
    min_source_videos = int(getattr(cfg, "min_source_videos_per_speaker", 3))

    for celeb in celebrities:
        speaker_id = str(celeb.id)
        before = dict(initial[speaker_id])
        current = dict(before)
        attempts_used = 0

        def sufficient(values: Dict[str, float]) -> bool:
            return (
                int(values["n_clips"]) >= min_clips
                and float(values["duration_sec"]) >= min_duration_sec
                and int(values["n_videos"]) >= min_source_videos
            )

        if sufficient(current):
            logging.info(
                "%s (%s) already has sufficient coverage: %d clips, %.1fs; moving to the next speaker",
                celeb.name,
                speaker_id,
                current["n_clips"],
                current["duration_sec"],
            )
        else:
            logging.warning(
                "%s (%s) has insufficient coverage: %d/%d clips, %.1f/%.1fs",
                celeb.name,
                speaker_id,
                current["n_clips"],
                min_clips,
                current["duration_sec"],
                min_duration_sec,
            )
            for attempt in range(1, max_attempts + 1):
                attempts_used = attempt
                retry_cfg = copy.copy(cfg)
                if relaxed_retry:
                    apply_relaxed_candidate_profile(retry_cfg)
                logging.info("Retry %d/%d for %s (%s)", attempt, max_attempts, celeb.name, speaker_id)
                videos = sorted(
                    [video for video in getattr(celeb, "videos", []) if getattr(video, "use", True)],
                    key=lambda video: getattr(video, "priority", 1),
                )
                retry_celeb = copy.copy(celeb)
                retry_celeb.videos = videos
                run_tag = (
                    f"coverage_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}_{attempt}"
                )
                logging.info(
                    "Jointly reprocessing %d videos for %s to increase coverage",
                    len(videos),
                    celeb.name,
                )
                stage_candidates(
                    retry_cfg,
                    [retry_celeb],
                    resume=True,
                    force=False,
                    reprocess_all=True,
                    candidate_run_tag=run_tag,
                )
                stage_approve_auto(cfg, speaker_ids)
                current = _approved_speaker_coverage(cfg, [speaker_id])[speaker_id]
                logging.info(
                    "Current coverage: %d clips, %.1fs, %d videos",
                    current["n_clips"],
                    current["duration_sec"],
                    current["n_videos"],
                )
                if sufficient(current):
                    break

        status = "sufficient" if sufficient(current) else "insufficient"
        if status == "insufficient":
            logging.error(
                "%s still has insufficient coverage after %d retries: %d clips, %.1fs",
                celeb.name,
                attempts_used,
                current["n_clips"],
                current["duration_sec"],
            )
        report_rows.append({
            "speaker_id": speaker_id,
            "speaker_name": celeb.name,
            "initial_clips": int(before["n_clips"]),
            "initial_duration_sec": float(before["duration_sec"]),
            "final_clips": int(current["n_clips"]),
            "final_duration_sec": float(current["duration_sec"]),
            "final_videos": int(current["n_videos"]),
            "missing_candidate_files": int(current["missing_files"]),
            "attempts_used": attempts_used,
            "required_clips": min_clips,
            "required_duration_sec": min_duration_sec,
            "required_source_videos": min_source_videos,
            "status": status,
        })

    # Recalcula la selección conjunta después de todos los reintentos.
    stage_approve_auto(cfg, speaker_ids)
    final_coverage = _approved_speaker_coverage(cfg, speaker_ids)
    for row in report_rows:
        values = final_coverage[str(row["speaker_id"])]
        row["final_clips"] = int(values["n_clips"])
        row["final_duration_sec"] = float(values["duration_sec"])
        row["final_videos"] = int(values["n_videos"])
        row["missing_candidate_files"] = int(values["missing_files"])
        row["status"] = "sufficient" if (
            int(values["n_clips"]) >= min_clips
            and float(values["duration_sec"]) >= min_duration_sec
            and int(values["n_videos"]) >= min_source_videos
        ) else "insufficient"

    report_dir = Path(cfg.reports_dir) / "coverage"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "speaker_coverage.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    insufficient = sum(row["status"] == "insufficient" for row in report_rows)
    logging.info("Speaker coverage: %s", report_path)
    logging.info("Coverage result: %d sufficient, %d insufficient", len(report_rows) - insufficient, insufficient)
    if insufficient and bool(getattr(cfg, "fail_if_insufficient_coverage", True)):
        raise RuntimeError(
            f"{insufficient} speaker(s) do not reach the required {min_clips} clips; "
            f"see {report_path}"
        )
    return report_path


def stage_export(cfg, celebrities) -> Path:
    approved_csv = Path(cfg.data_candidates) / "approved_segments.csv"
    if not approved_csv.exists():
        raise FileNotFoundError(f"{approved_csv} does not exist. Run approve-auto or create approved_segments.csv manually")
    target = int(getattr(cfg, "target_clips_per_speaker", 50))
    coverage = _approved_speaker_coverage(cfg, [str(celeb.id) for celeb in celebrities])
    unbalanced = {
        speaker_id: {
            "clips": int(values["n_clips"]),
            "videos": int(values["n_videos"]),
        }
        for speaker_id, values in coverage.items()
        if (
            int(values["n_clips"]) != target
            or int(values["n_videos"]) < int(getattr(cfg, "min_source_videos_per_speaker", 3))
        )
    }
    if unbalanced:
        if bool(getattr(cfg, "require_exact_balanced_export", True)):
            raise RuntimeError(
                f"Export cancelled: exactly {target} clips per speaker are required; "
                f"current counts={unbalanced}"
            )
        logging.warning(
            "Partial export: some speakers do not reach the quota, "
            "but only verified clips will be exported: %s",
            unbalanced,
        )

    exporter = VoxCelebExporter(cfg)
    utt_df = exporter.export_from_approved(approved_csv, celebrities_config=celebrities)
    exporter.make_trials(max_trials=int(getattr(cfg, "max_internal_trials", 200000)))

    processed_dir = Path(cfg.data_processed)
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = processed_dir / "metadata.csv"
    utt_df.to_csv(metadata_path, index=False)
    logging.info("Metadata processed: %s", metadata_path)
    return metadata_path


def stage_validate(cfg) -> Path:
    metadata_path = Path(cfg.data_processed) / "metadata.csv"
    report_dir = Path(cfg.reports_dir) / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} does not exist")
    df = pd.read_csv(metadata_path)
    if df.empty:
        raise RuntimeError("metadata.csv is empty")

    validation_errors: List[str] = []
    expected_version = str(getattr(cfg, "pipeline_version", "2.1-syncnet-framewise"))
    if "pipeline_version" not in df or not df["pipeline_version"].fillna("").astype(str).eq(expected_version).all():
        validation_errors.append("some clips were generated with an outdated pipeline version")
    required_binary_gates = ["syncnet_verified", "syncnet_unambiguous"]
    for column in required_binary_gates:
        values = pd.to_numeric(df.get(column, pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        if not values.ge(1).all():
            validation_errors.append(f"{column} is not satisfied for all clips")
    max_clip_duration = float(getattr(cfg, "segment_max_duration", 5.0))
    durations = pd.to_numeric(df.get("duration", pd.Series(np.nan, index=df.index)), errors="coerce")
    if durations.isna().any():
        validation_errors.append("some clips do not have a valid duration")
    elif durations.gt(max_clip_duration + 1e-3).any():
        longest = float(durations.max())
        validation_errors.append(
            f"some clips are longer than {max_clip_duration:.1f} s (maximum detected: {longest:.3f} s)"
        )
    if "sha1" in df and df["sha1"].fillna("").astype(str).duplicated().any():
        validation_errors.append("duplicate WAV files were found by SHA-1")
    audio_column = "audio_path" if "audio_path" in df else "candidate_path"
    if audio_column not in df or not df[audio_column].fillna("").astype(str).map(lambda p: Path(p).is_file()).all():
        validation_errors.append("some WAV files declared in the metadata are missing")

    max_temporal_iou = float(getattr(cfg, "max_selected_temporal_iou", 0.05))
    temporal_duplicates = 0
    if {"speaker_id", "video_id", "start", "end"}.issubset(df.columns):
        for _, group in df.groupby(["speaker_id", "video_id"]):
            intervals = sorted(
                (float(row["start"]), float(row["end"])) for _, row in group.iterrows()
            )
            for i, (start, end) in enumerate(intervals):
                for other_start, other_end in intervals[i + 1:]:
                    if other_start >= end:
                        break
                    intersection = max(0.0, min(end, other_end) - max(start, other_start))
                    union = max(end, other_end) - min(start, other_start)
                    if union > 0 and intersection / union > max_temporal_iou:
                        temporal_duplicates += 1
    if temporal_duplicates:
        validation_errors.append(f"there are {temporal_duplicates} pairs with excessive temporal overlap")

    speaker_stats = df.groupby("speaker_id").agg(
        n_utterances=("speaker_id", "size"),
        n_videos=("video_id", "nunique"),
        duration_sec=("duration", "sum"),
        mean_duration=("duration", "mean"),
        mean_snr=("snr_db", "mean"),
        mean_vad=("vad_ratio", "mean"),
    ).reset_index().sort_values("n_utterances")
    video_stats = df.groupby(["speaker_id", "video_id"]).agg(
        n_utterances=("video_id", "size"),
        duration_sec=("duration", "sum"),
    ).reset_index().sort_values("n_utterances")

    speaker_stats.to_csv(report_dir / "speaker_stats.csv", index=False)
    video_stats.to_csv(report_dir / "video_stats.csv", index=False)

    min_utt = int(getattr(cfg, "min_utterances_per_speaker_report", 50))
    exact_target = int(getattr(cfg, "target_clips_per_speaker", min_utt))
    summary = {
        "n_speakers": int(df["speaker_id"].nunique()),
        "n_videos": int(df["video_id"].nunique()),
        "n_utterances": int(len(df)),
        "duration_hours": float(df["duration"].sum() / 3600.0),
        "mean_duration": float(df["duration"].mean()),
        "speakers_below_min_utt": int((speaker_stats["n_utterances"] < min_utt).sum()),
        "speakers_not_exact_target": int((speaker_stats["n_utterances"] != exact_target).sum()),
        "min_utt_threshold": min_utt,
        "validation_errors": validation_errors,
    }
    lines = [
        "# VoxCeleb-ESP-Train validation",
        "",
        f"- Speakers: **{summary['n_speakers']}**",
        f"- Videos: **{summary['n_videos']}**",
        f"- Utterances: **{summary['n_utterances']}**",
        f"- Total duration: **{summary['duration_hours']:.2f} h**",
        f"- Mean duration: **{summary['mean_duration']:.2f} s**",
        f"- Speakers with fewer than {min_utt} utterances: **{summary['speakers_below_min_utt']}**",
        f"- Speakers with a count different from {exact_target}: **{summary['speakers_not_exact_target']}**",
        f"- Integrity errors: **{len(validation_errors)}**",
        "",
        "Generated files:",
        "- speaker_stats.csv",
        "- video_stats.csv",
    ]
    out_md = report_dir / "VALIDATION_SUMMARY.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Validation saved: %s", out_md)
    if validation_errors:
        raise RuntimeError("Validation failed: " + "; ".join(validation_errors))
    return out_md


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Generated YAML or direct YAML file")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["candidates", "approve-auto", "ensure-coverage", "export", "validate", "all"],
    )
    parser.add_argument("--limit-videos", type=int, default=None, help="Quick test with N videos")
    parser.add_argument("--resume", action="store_true", help="Do not reprocess videos already present in candidates.csv")
    parser.add_argument("--force", action="store_true", help="Ignore resume state and reprocess")
    parser.add_argument(
        "--skip-complete-speakers",
        action="store_true",
        help="Skip speakers whose approved quota is already complete",
    )
    parser.add_argument(
        "--speaker-id",
        action="append",
        default=[],
        help="Process only this speaker_id; this option can be repeated",
    )
    parser.add_argument(
        "--relaxed-candidates",
        action="store_true",
        help="More inclusive second pass; marks candidates for manual review",
    )
    parser.add_argument("--min-clips-per-speaker", type=int, default=None)
    parser.add_argument("--min-duration-per-speaker-sec", type=float, default=None)
    parser.add_argument("--max-speaker-attempts", type=int, default=None)
    parser.add_argument(
        "--coverage-relaxed-retry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the relaxed profile during coverage retries",
    )
    args = parser.parse_args()

    loader = ConfigLoader(args.config)
    cfg = loader.global_config
    celebrities = loader.celebrities
    cfg.configured_video_ids_by_speaker = {
        str(celeb.id): [str(video.youtube_id) for video in celeb.videos if video.use]
        for celeb in celebrities
    }
    cfg.make_dirs()
    setup_logging(Path(cfg.logs_dir))

    if args.speaker_id:
        selected_ids = {str(s).strip() for s in args.speaker_id if str(s).strip()}
        celebrities = [c for c in celebrities if str(c.id) in selected_ids]
        missing_ids = selected_ids - {str(c.id) for c in celebrities}
        if missing_ids:
            raise ValueError(f"speaker_id not found in the configuration: {sorted(missing_ids)}")
        logging.info("Speaker filter enabled: %s", ", ".join(sorted(selected_ids)))

    if args.skip_complete_speakers and args.stage in {"candidates", "all"}:
        completed_ids = speakers_with_completed_quota(cfg)
        skipped = [str(celeb.id) for celeb in celebrities if str(celeb.id) in completed_ids]
        celebrities = [celeb for celeb in celebrities if str(celeb.id) not in completed_ids]
        if skipped:
            logging.info(
                "Speakers skipped because their quota is complete: %s",
                ", ".join(sorted(skipped)),
            )

    if args.relaxed_candidates:
        if args.stage not in {"candidates", "all"}:
            raise ValueError("--relaxed-candidates can only be used with stage candidates/all")
        apply_relaxed_candidate_profile(cfg)

    logging.info("Config: %s", args.config)
    logging.info("Speakers loaded: %d", len(celebrities))
    logging.info("Videos loaded: %d", sum(len(c.videos) for c in celebrities))
    logging.info("Output: %s", cfg.output_base)

    if args.stage in {"candidates", "all"}:
        if celebrities:
            stage_candidates(cfg, celebrities, limit_videos=args.limit_videos, resume=args.resume, force=args.force)
        else:
            logging.info("No speakers are pending for the candidates stage")
    if args.stage == "approve-auto":
        stage_approve_auto(cfg, [str(celeb.id) for celeb in celebrities])
    if args.stage in {"ensure-coverage", "all"}:
        stage_ensure_speaker_coverage(
            cfg,
            celebrities,
            min_clips=args.min_clips_per_speaker,
            min_duration_sec=args.min_duration_per_speaker_sec,
            max_attempts=args.max_speaker_attempts,
            relaxed_retry=args.coverage_relaxed_retry,
        )
    if args.stage in {"export", "all"}:
        stage_export(cfg, celebrities)
    if args.stage in {"validate", "all"}:
        stage_validate(cfg)

    logging.info("Finished")


if __name__ == "__main__":
    main()
