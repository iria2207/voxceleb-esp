#Carga de configuración YAML para el pipeline VoxCeleb-ESP.


from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match) -> str:
            name = match.group(1)
            default = match.group(2)
            return os.environ.get(name, default or "")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value


def _to_path(value: Any, default: Optional[str] = None) -> Path:
    if value is None:
        if default is None:
            raise ValueError("Valor de path obligatorio")
        value = default
    return Path(str(value)).expanduser()


@dataclass
class VideoConfig:
    youtube_id: str
    url: str = ""
    split: str = "train"
    priority: int = 1
    use: bool = True
    source: str = "youtube"
    notes: str = ""


@dataclass
class CelebrityConfig:
    id: str
    name: str
    gender: str = "unknown"
    category: str = "unknown"
    region: str = "unknown"
    birth_year: Optional[int] = None
    reference_images: List[Path] = field(default_factory=list)
    videos: List[VideoConfig] = field(default_factory=list)
    notes: str = ""


class GlobalConfig:

    PATH_KEYS = {
        "download_path", "output_base", "temp_frames_dir", "data_raw",
        "data_processed", "data_candidates", "data_refined", "models_dir",
        "logs_dir", "reports_dir", "references_dir",
    }

    def __init__(self, data: Dict[str, Any], config_path: Path):
        self.config_path = config_path
        self.project_dir = config_path.parent.resolve()

        defaults: Dict[str, Any] = {
            "pipeline_version": "2.1-syncnet-framewise",
            "download_path": str(self.project_dir / "videos"),
            "output_base": str(self.project_dir / "output"),
            "data_raw": str(self.project_dir / "data" / "raw"),
            "data_processed": str(self.project_dir / "data" / "processed"),
            "data_candidates": str(self.project_dir / "data" / "candidates"),
            "temp_frames_dir": str(self.project_dir / "temp" / "frames"),
            "models_dir": str(self.project_dir / "models"),
            "logs_dir": str(self.project_dir / "logs"),
            "reports_dir": str(self.project_dir / "reports"),
            "references_dir": str(self.project_dir / "references"),
            "fps": 2,
            "detect_fps": 2,
            "frame_width": 640,
            "ffmpeg_loglevel": "error",
            "long_video_mode": "windowed",
            "long_video_window_seconds": 600,
            "long_video_hop_seconds": 600,
            "long_video_max_windows": 0,
            "long_video_start_seconds": 0,
            "long_video_min_remaining_seconds": 60,
            "long_video_strategy": "sliding",
            "cleanup_window_audio": True,
            "quiet_progress": True,
            "audio_sr": 16000,
            "face_det_size": [640, 640],
            "min_face_size": 80,
            "iou_threshold": 0.45,
            "min_track_frames": 10,
            "max_lost_frames": 5,
            "face_recovery_enabled": True,
            "face_recovery_min_face_size": 48,
            "face_recovery_max_lost_frames": 8,
            "face_recovery_min_association_iou": 0.08,
            "face_recovery_min_embedding_similarity": 0.40,
            "face_template_threshold": 0.60,
            "face_template_margin": 0.08,
            "face_min_track_consistency": 0.75,
            "face_template_min_refs": 2,
            "track_min_association_iou": 0.15,
            "track_min_embedding_similarity": 0.30,
            "use_face_classifier": False,
            "segment_min_duration": 2.0,
            "segment_target_duration": 3.0,
            "segment_max_duration": 5.0,
            "segments_per_video": 20,
            "min_selected_per_video": 5,
            "target_clips_per_speaker": 50,
            "max_candidates_per_video": 80,
            "early_stop_candidates_enabled": True,
            "early_stop_candidates_per_video": 25,
            "early_stop_min_temporal_windows": 3,
            "max_selected_temporal_iou": 0.05,
            "vad_aggressiveness": 2,
            "min_snr_db": 5.0,
            "min_speech_ratio": 0.45,
            "max_clipping_ratio": 0.02,
            "max_overlap_ratio": 0.10,
            "syncnet_model_path": None,
            "syncnet_threshold": 0.45,
            "syncnet_min_confidence": 0.45,
            "active_proposal_threshold": 0.30,
            "active_merge_gap_seconds": 1.10,
            "active_max_bbox_gap_frames": 3,
            "active_fallback_to_track_windows": True,
            "real_asd_enabled": True,
            "real_asd_required": True,
            "syncnet_confidence_threshold": 4.0,
            # Fallback automatico a CPU si no queda VRAM tras InsightFace.
            "syncnet_device": "cuda:0",
            "syncnet_max_abs_offset_frames": 3,
            "syncnet_crop_fps": 25,
            "syncnet_crop_size": 224,
            "syncnet_crop_margin": 0.40,
            "syncnet_batch_size": 20,
            "syncnet_vshift": 10,
            "keep_syncnet_crops": False,
            "syncnet_framewise_enabled": True,
            "syncnet_framewise_threshold": 4.0,
            "syncnet_framewise_min_high_seconds": 0.80,
            "syncnet_framewise_min_span_seconds": 1.20,
            "syncnet_framewise_bridge_gap_seconds": 0.32,
            "syncnet_framewise_output_min_seconds": 2.0,
            "syncnet_framewise_output_target_seconds": 3.0,
            "syncnet_framewise_output_max_seconds": 5.0,
            "syncnet_framewise_min_high_ratio": 0.30,
            "syncnet_local_offset_enabled": True,
            "syncnet_local_offset_window_seconds": 2.0,
            "syncnet_local_offset_min_stability_ratio": 0.60,
            "syncnet_probe_duration_seconds": 12.0,
            "syncnet_probe_hop_seconds": 10.0,
            "syncnet_probe_min_duration_seconds": 2.0,
            "syncnet_probe_min_track_coverage": 0.70,
            "syncnet_max_probes_per_window": 160,
            "syncnet_competitor_check": True,
            "syncnet_competitor_min_coverage": 0.65,
            "syncnet_max_competitors": 2,
            "allow_video_cap_overflow": True,
            "active_step_seconds": 0.5,
            "dedup_exact": True,
            "dedup_fingerprint_threshold": 0.985,
            "device": "cuda",
            "max_workers": 4,
            "batch_size": 16,
            "min_utterances_per_speaker_report": 50,
            "min_clips_per_speaker": 50,
            "min_duration_per_speaker_sec": 0.0,
            "min_source_videos_per_speaker": 2,
            "max_speaker_retry_attempts": 1,
            "speaker_retry_relaxed": False,
            "fail_if_insufficient_coverage": True,
            "require_exact_balanced_export": True,
        }

        merged = {**defaults, **(data or {})}
        for key, value in merged.items():
            if key in self.PATH_KEYS:
                setattr(self, key, _to_path(value))
            else:
                setattr(self, key, value)

        # Directorios derivados cómodos.
        self.data_raw = _to_path(getattr(self, "data_raw"))
        self.data_processed = _to_path(getattr(self, "data_processed"))
        self.data_candidates = _to_path(getattr(self, "data_candidates"))
        self.logs_dir = _to_path(getattr(self, "logs_dir"))
        self.reports_dir = _to_path(getattr(self, "reports_dir"))

    def make_dirs(self) -> None:
        for attr in [
            "download_path", "output_base", "data_raw", "data_processed",
            "data_candidates", "temp_frames_dir", "models_dir", "logs_dir",
            "reports_dir", "references_dir",
        ]:
            path = getattr(self, attr, None)
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)


class ConfigLoader:
    def __init__(self, yaml_path: str | Path):
        self.yaml_path = Path(yaml_path).expanduser().resolve()
        if not self.yaml_path.exists():
            raise FileNotFoundError(f"No existe el YAML: {self.yaml_path}")

        with open(self.yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw = _expand_env(raw)

        self.raw = raw
        self.global_config = GlobalConfig(raw.get("global", {}), self.yaml_path)
        self.celebrities = self._parse_celebrities(raw.get("celebrities", []))

    def _resolve_ref_path(self, ref: str | Path) -> Path:
        p = Path(str(ref)).expanduser()
        if p.is_absolute():
            return p
        # Primero relativo al directorio del YAML; luego relativo a references_dir.
        direct = (self.yaml_path.parent / p).resolve()
        if direct.exists():
            return direct
        return (self.global_config.references_dir / p.name).resolve() if len(p.parts) == 1 else direct

    def _parse_celebrities(self, rows: List[Dict[str, Any]]) -> List[CelebrityConfig]:
        celebrities: List[CelebrityConfig] = []
        for row in rows or []:
            if not row:
                continue
            videos = []
            for v in row.get("videos", []) or []:
                if not v or not v.get("youtube_id"):
                    continue
                use = v.get("use", True)
                if isinstance(use, str):
                    use = use.strip().lower() not in {"0", "false", "no", "n"}
                videos.append(VideoConfig(
                    youtube_id=str(v.get("youtube_id", "")).strip(),
                    url=str(v.get("url", "")).strip(),
                    split=str(v.get("split", "train")).strip(),
                    priority=int(v.get("priority", 1) or 1),
                    use=bool(use),
                    source=str(v.get("source", "youtube")),
                    notes=str(v.get("notes", "")),
                ))
            refs = [self._resolve_ref_path(p) for p in row.get("reference_images", []) or []]
            birth_year = row.get("birth_year")
            try:
                birth_year = int(birth_year) if birth_year not in (None, "") else None
            except Exception:
                birth_year = None
            celebrities.append(CelebrityConfig(
                id=str(row.get("id", "")).strip(),
                name=str(row.get("name", "")).strip(),
                gender=str(row.get("gender", "unknown")).strip() or "unknown",
                category=str(row.get("category", "unknown")).strip() or "unknown",
                region=str(row.get("region", "unknown")).strip() or "unknown",
                birth_year=birth_year,
                reference_images=refs,
                videos=videos,
                notes=str(row.get("notes", "")),
            ))
        return [c for c in celebrities if c.id and c.name and c.videos]
