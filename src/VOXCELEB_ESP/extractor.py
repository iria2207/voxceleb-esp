#Extracción robusta para vídeos largos.


from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf


class MediaExtractor:
    def __init__(self, config):
        self.config = config
        self.audio_sr = int(getattr(config, "audio_sr", 16000))
        self.detect_fps = float(getattr(config, "detect_fps", getattr(config, "fps", 2)))
        self.frame_width = int(getattr(config, "frame_width", 640))
        self.ffmpeg_loglevel = str(getattr(config, "ffmpeg_loglevel", "error"))
        self.data_raw = Path(getattr(config, "data_raw", "data/raw"))
        self.temp_frames_dir = Path(getattr(config, "temp_frames_dir", "temp/frames"))
        self.audio_dir = self.data_raw / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.temp_frames_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _run(cmd: List[str], label: str) -> None:
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "")[-3000:]
            logging.error("ffmpeg/ffprobe error en %s:\n%s", label, stderr)
            raise

    def probe_duration(self, video_path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        try:
            r = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return float((r.stdout or "0").strip())
        except Exception:
            logging.warning("No se pudo obtener duración con ffprobe: %s", video_path)
            return 0.0

    def extract_audio(self, video_path: Path, video_id: str) -> Path:
        """Extrae audio completo. Solo usar si el vídeo no es enorme."""
        audio_path = self.audio_dir / f"{video_id}.wav"
        if audio_path.exists() and audio_path.stat().st_size > 1024:
            return audio_path
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", self.ffmpeg_loglevel, "-y",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(self.audio_sr),
            "-acodec", "pcm_s16le", str(audio_path),
        ]
        self._run(cmd, f"audio completo {video_id}")
        return audio_path

    def extract_audio_segment(self, video_path: Path, video_id: str, start: float, duration: float, tag: str) -> Path:
        audio_path = self.audio_dir / "windows" / video_id / f"{tag}.wav"
        if audio_path.exists() and audio_path.stat().st_size > 1024:
            return audio_path
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", self.ffmpeg_loglevel, "-y",
            "-ss", f"{float(start):.3f}",
            "-i", str(video_path),
            "-t", f"{float(duration):.3f}",
            "-vn", "-ac", "1", "-ar", str(self.audio_sr),
            "-acodec", "pcm_s16le", str(audio_path),
        ]
        self._run(cmd, f"audio ventana {video_id} {tag}")
        return audio_path

    def extract_frames(self, video_path: Path, video_id: str, overwrite: bool = True) -> List[Path]:
        out_dir = self.temp_frames_dir / video_id
        if overwrite and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        pattern = out_dir / "frame_%06d.jpg"
        vf = f"fps={self.detect_fps},scale={self.frame_width}:-1"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", self.ffmpeg_loglevel, "-y",
            "-i", str(video_path),
            "-vf", vf,
            "-q:v", "5", str(pattern),
        ]
        self._run(cmd, f"frames completos {video_id}")
        return sorted(out_dir.glob("frame_*.jpg"))

    def extract_frames_segment(self, video_path: Path, video_id: str, start: float, duration: float,
                               tag: str, overwrite: bool = True) -> List[Path]:
       
        out_dir = self.temp_frames_dir / video_id / tag
        if overwrite and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        pattern = out_dir / "frame_%06d.jpg"
        vf = f"fps={self.detect_fps},scale={self.frame_width}:-1"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", self.ffmpeg_loglevel, "-y",
            "-ss", f"{float(start):.3f}",
            "-i", str(video_path),
            "-t", f"{float(duration):.3f}",
            "-vf", vf,
            "-q:v", "5", str(pattern),
        ]
        self._run(cmd, f"frames ventana {video_id} {tag}")
        return sorted(out_dir.glob("frame_*.jpg"))

    def extract_window(self, video_path: Path, video_id: str, start: float, duration: float,
                       tag: Optional[str] = None) -> Tuple[Path, List[Path]]:
        tag = tag or f"w{int(start):06d}_{int(duration):04d}"
        audio_path = self.extract_audio_segment(video_path, video_id, start, duration, tag)
        frames = self.extract_frames_segment(video_path, video_id, start, duration, tag, overwrite=True)
        if not frames:
            raise RuntimeError(f"No se extrajeron frames de {video_path} en ventana {tag}")
        return audio_path, frames

    def extract(self, video_path: Path, video_id: str, for_detection: bool = True) -> Tuple[Path, List[Path]]:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Vídeo no encontrado: {video_path}")
        audio_path = self.extract_audio(video_path, video_id)
        frames = self.extract_frames(video_path, video_id, overwrite=True)
        if not frames:
            raise RuntimeError(f"No se extrajeron frames de {video_path}")
        return audio_path, frames

    @staticmethod
    def load_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size == 0:
            raise ValueError(f"Audio vacío: {audio_path}")
        return audio.astype(np.float32), int(sr)

    @staticmethod
    def cleanup_frames(frame_paths: List[Path]) -> None:
        if not frame_paths:
            return
        parent = Path(frame_paths[0]).parent
        for fp in frame_paths:
            try:
                Path(fp).unlink(missing_ok=True)
            except Exception:
                pass
        # Borra carpeta de ventana y, si queda vacía, carpeta del vídeo.
        for p in [parent, parent.parent]:
            try:
                p.rmdir()
            except Exception:
                pass
