#Verificación audiovisual fuerte para VoxCeleb-ESP.


from __future__ import annotations

import logging
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Sequence

import cv2
import numpy as np


@dataclass
class SyncNetSegment:
    start: float
    end: float
    confidence: float
    high_ratio: float
    offset_frames: int = 0


@dataclass
class SyncNetResult:
    accepted: bool
    confidence: float
    offset_frames: int
    min_distance: float
    reason: str
    segments: tuple[SyncNetSegment, ...] = ()
    frame_confidence: tuple[float, ...] = ()


class SyncNetTrackVerifier:
    """Ejecuta SyncNet sobre un recorte de un único track facial."""

    def __init__(self, config):
        self.config = config
        self.enabled = bool(getattr(config, "real_asd_enabled", True))
        self.required = bool(getattr(config, "real_asd_required", True))
        requested_device = str(getattr(config, "syncnet_device", getattr(config, "device", "cpu"))).lower()
        self.device = "cuda:0" if "cuda" in requested_device else "cpu"
        self.confidence_threshold = float(getattr(config, "syncnet_confidence_threshold", 4.0))
        self.max_abs_offset = int(getattr(config, "syncnet_max_abs_offset_frames", 3))
        self.crop_fps = float(getattr(config, "syncnet_crop_fps", 25.0))
        self.crop_size = int(getattr(config, "syncnet_crop_size", 224))
        self.crop_margin = float(getattr(config, "syncnet_crop_margin", 0.40))
        self.keep_crops = bool(getattr(config, "keep_syncnet_crops", False))
        self.framewise_enabled = bool(getattr(config, "syncnet_framewise_enabled", True))
        self.framewise_threshold = float(
            getattr(config, "syncnet_framewise_threshold", self.confidence_threshold)
        )
        self.framewise_min_high_seconds = float(
            getattr(config, "syncnet_framewise_min_high_seconds", 0.80)
        )
        self.framewise_min_span_seconds = float(
            getattr(config, "syncnet_framewise_min_span_seconds", 1.20)
        )
        self.framewise_bridge_gap_seconds = float(
            getattr(config, "syncnet_framewise_bridge_gap_seconds", 0.32)
        )
        self.framewise_output_min_seconds = float(
            getattr(config, "syncnet_framewise_output_min_seconds", 2.0)
        )
        self.framewise_output_max_seconds = float(
            getattr(config, "syncnet_framewise_output_max_seconds", 5.0)
        )
        requested_output_target = float(
            getattr(config, "syncnet_framewise_output_target_seconds", 3.0)
        )
        if self.framewise_output_max_seconds < self.framewise_output_min_seconds:
            raise ValueError(
                "syncnet_framewise_output_max_seconds no puede ser menor que "
                "syncnet_framewise_output_min_seconds"
            )
        self.framewise_output_target_seconds = min(
            requested_output_target,
            self.framewise_output_max_seconds,
        )
        self.framewise_min_high_ratio = float(
            getattr(config, "syncnet_framewise_min_high_ratio", 0.30)
        )
        self.local_offset_enabled = bool(
            getattr(config, "syncnet_local_offset_enabled", True)
        )
        self.local_offset_window_seconds = float(
            getattr(config, "syncnet_local_offset_window_seconds", 2.0)
        )
        self.local_offset_min_stability_ratio = float(
            getattr(config, "syncnet_local_offset_min_stability_ratio", 0.60)
        )

        models_dir = Path(getattr(config, "models_dir", "models"))
        model_path = getattr(config, "syncnet_model_path", None)
        self.model_path = Path(model_path) if model_path else models_dir / "syncnet" / "syncnet_v2.model"
        self.code_dir = Path(getattr(config, "syncnet_code_dir", models_dir / "syncnet" / "syncnet_python"))
        default_temp = Path(getattr(config, "temp_frames_dir", "temp/frames")).parent / "syncnet_crops"
        self.temp_dir = Path(getattr(config, "syncnet_temp_dir", default_temp))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline = None

    def validate_ready(self) -> None:
        if not self.enabled:
            if self.required:
                raise RuntimeError("real_asd_required=true pero real_asd_enabled=false")
            return
        required_files = [self.model_path, self.code_dir / "SyncNetInstance.py", self.code_dir / "SyncNetModel.py"]
        missing = [path for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Faltan pesos de SyncNet: " + ", ".join(map(str, missing))
                + ". Ejecuta setup_real_verification.sh antes del pipeline."
            )
        try:
            code_dir = str(self.code_dir.resolve())
            if code_dir not in sys.path:
                sys.path.insert(0, code_dir)
            from SyncNetInstance import SyncNetInstance  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "No se puede importar la implementación oficial de SyncNet. Ejecuta setup_real_verification.sh "
                f"en el mismo entorno Python. Error original: {exc}"
            ) from exc
        try:
            self._pipeline = SyncNetInstance(device=self.device)
        except RuntimeError as exc:
            # InsightFace permanece cargado durante la verificacion audiovisual
            # y algunos nodos no tienen VRAM suficiente para ambos modelos. No
            # abortamos horas de trabajo: SyncNet funciona tambien en CPU.
            cuda_memory_error = self.device.startswith("cuda") and (
                "out of memory" in str(exc).lower()
                or "cuda driver error" in str(exc).lower()
            )
            if not cuda_memory_error:
                raise
            logging.warning(
                "⚠️ SyncNet no pudo inicializarse en %s (%s: %s); "
                "continúo automáticamente en CPU",
                self.device,
                type(exc).__name__,
                exc,
            )
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            self.device = "cpu"
            self._pipeline = SyncNetInstance(device=self.device)
        self._pipeline.loadParameters(str(self.model_path))
        logging.info(
            "✅ SyncNet real preparado: confianza>=%.2f | |offset|<=%d frames | %s",
            self.confidence_threshold,
            self.max_abs_offset,
            self.device,
        )

    @staticmethod
    def _interpolated_bbox(track: Dict, local_time: float, detect_fps: float) -> Optional[np.ndarray]:
        frames = np.asarray(track.get("frames", []), dtype=np.float32)
        bboxes = np.asarray(track.get("bboxes", []), dtype=np.float32)
        if frames.size == 0 or bboxes.ndim != 2 or bboxes.shape[1] != 4:
            return None
        n = min(len(frames), len(bboxes))
        frames, bboxes = frames[:n], bboxes[:n]
        # SyncNet es sensible a recortes que tiemblan. Suavizamos el track con
        # una mediana temporal antes de interpolarlo de 5 a 25 FPS.
        if n >= 5:
            cached = track.get("_syncnet_smoothed_bboxes")
            if cached is None or np.asarray(cached).shape != bboxes.shape:
                padded = np.pad(bboxes, ((2, 2), (0, 0)), mode="edge")
                cached = np.vstack([np.median(padded[i:i + 5], axis=0) for i in range(n)]).astype(np.float32)
                track["_syncnet_smoothed_bboxes"] = cached
            bboxes = np.asarray(cached, dtype=np.float32)
        position = float(local_time) * float(detect_fps)
        if position < frames[0] - detect_fps or position > frames[-1] + detect_fps:
            return None
        return np.asarray(
            [np.interp(position, frames, bboxes[:, coordinate]) for coordinate in range(4)],
            dtype=np.float32,
        )

    @staticmethod
    def _median_filter(values: np.ndarray, kernel_size: int = 9) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return values
        kernel_size = max(1, int(kernel_size) | 1)
        radius = kernel_size // 2
        padded = np.pad(values, (radius, radius), mode="edge")
        return np.asarray(
            [np.median(padded[i:i + kernel_size]) for i in range(values.size)],
            dtype=np.float32,
        )

    def _framewise_from_distances(self, distances: np.ndarray) -> np.ndarray:
        """Reproduce ``fconfm`` de SyncNetInstance.evaluate."""
        matrix = np.asarray(distances, dtype=np.float32)
        if matrix.ndim != 2 or matrix.size == 0:
            return np.zeros(0, dtype=np.float32)
        expected_shifts = int(getattr(self.config, "syncnet_vshift", 10)) * 2 + 1
        if matrix.shape[1] != expected_shifts and matrix.shape[0] == expected_shifts:
            matrix = matrix.T
        if matrix.shape[1] != expected_shifts:
            logging.warning("Forma de distancias SyncNet inesperada: %s", matrix.shape)
            return np.zeros(0, dtype=np.float32)
        mean_by_shift = np.mean(matrix, axis=0)
        best_shift = int(np.argmin(mean_by_shift))
        baseline = float(np.median(mean_by_shift))
        raw = baseline - matrix[:, best_shift]
        return self._median_filter(raw, kernel_size=9)

    def _local_framewise_from_distances(
        self, distances: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calcula confianza y offset local sin cambiar el umbral oficial.

        ``SyncNetInstance.evaluate`` escoge un único offset promediando todo el
        probe. Eso es apropiado para sincronizar un clip homogéneo, pero en una
        entrevista un silencio o un cambio de plano puede dominar la media y
        ocultar varios segundos realmente sincronizados. Aquí se escoge el
        offset sobre una ventana móvil de dos segundos y después se exige que
        permanezca dentro del mismo límite estricto configurado (por defecto
        ±3 frames) durante todo el tramo aceptado.
        """
        matrix = np.asarray(distances, dtype=np.float32)
        if matrix.ndim != 2 or matrix.size == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int16)
        vshift = int(getattr(self.config, "syncnet_vshift", 10))
        expected_shifts = vshift * 2 + 1
        if matrix.shape[1] != expected_shifts and matrix.shape[0] == expected_shifts:
            matrix = matrix.T
        if matrix.shape[1] != expected_shifts:
            logging.warning("Forma de distancias SyncNet inesperada: %s", matrix.shape)
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int16)

        window = max(3, int(round(self.local_offset_window_seconds * self.crop_fps)) | 1)
        radius = window // 2
        padded = np.pad(matrix, ((radius, radius), (0, 0)), mode="edge")
        cumulative = np.vstack([
            np.zeros((1, matrix.shape[1]), dtype=np.float64),
            np.cumsum(padded, axis=0, dtype=np.float64),
        ])
        rolling = (cumulative[window:] - cumulative[:-window]) / float(window)
        best_indices = np.argmin(rolling, axis=1)
        offsets = (vshift - best_indices).astype(np.int16)

        # Igual que el código oficial, la confianza es la mediana de las
        # distancias alternativas menos la distancia del offset escogido.
        baseline = np.median(rolling, axis=1)
        raw = baseline - matrix[np.arange(len(matrix)), best_indices]
        # Un offset local fuera del rango permitido nunca puede crear un tramo,
        # aunque su confianza sea alta.
        raw[np.abs(offsets) > self.max_abs_offset] = -99.0
        return self._median_filter(raw, kernel_size=9), offsets

    @staticmethod
    def _bridge_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
        result = np.asarray(mask, dtype=bool).copy()
        i = 0
        while i < len(result):
            if result[i]:
                i += 1
                continue
            start = i
            while i < len(result) and not result[i]:
                i += 1
            if start > 0 and i < len(result) and i - start <= max_gap:
                result[start:i] = True
        return result

    def _segments_from_frame_confidence(
        self,
        frame_confidence: np.ndarray,
        duration: float,
        frame_offsets: Optional[np.ndarray] = None,
    ) -> tuple[SyncNetSegment, ...]:
        fps = float(self.crop_fps)
        values = np.asarray(frame_confidence, dtype=np.float32).reshape(-1)
        if values.size == 0 or duration <= 0:
            return ()
        offsets = None
        if frame_offsets is not None:
            candidate_offsets = np.asarray(frame_offsets, dtype=np.int16).reshape(-1)
            if candidate_offsets.size == values.size:
                offsets = candidate_offsets
        original_high = values >= self.framewise_threshold
        if offsets is not None:
            original_high &= np.abs(offsets) <= self.max_abs_offset
        bridged = self._bridge_short_false_gaps(
            original_high,
            max_gap=max(0, int(round(self.framewise_bridge_gap_seconds * fps))),
        )
        min_high_frames = max(1, int(round(self.framewise_min_high_seconds * fps)))
        min_span_frames = max(1, int(round(self.framewise_min_span_seconds * fps)))
        output_target_seconds = min(
            self.framewise_output_target_seconds,
            self.framewise_output_max_seconds,
        )
        segments: list[SyncNetSegment] = []
        i = 0
        while i < len(bridged):
            if not bridged[i]:
                i += 1
                continue
            start_idx = i
            while i < len(bridged) and bridged[i]:
                i += 1
            end_idx = i
            if end_idx - start_idx < min_span_frames:
                continue
            high_count = int(np.count_nonzero(original_high[start_idx:end_idx]))
            if high_count < min_high_frames:
                continue

            # Cada descriptor labial usa cinco frames; +2 centra el instante.
            active_start = max(0.0, (start_idx + 2) / fps)
            active_end = min(float(duration), (end_idx + 2) / fps)
            active_duration = active_end - active_start
            output_windows: list[tuple[float, float]] = []
            if active_duration <= output_target_seconds:
                output_duration = min(
                    max(self.framewise_output_min_seconds, active_duration),
                    output_target_seconds,
                    float(duration),
                )
                center = (active_start + active_end) / 2.0
                output_start = max(
                    0.0,
                    min(center - output_duration / 2.0, duration - output_duration),
                )
                output_windows.append((output_start, min(float(duration), output_start + output_duration)))
            else:
                # Un turno largo aporta varios utterances distintos, sin
                # solaparlos ni repetir siempre el centro del turno.
                cursor = active_start
                while cursor + self.framewise_output_min_seconds <= active_end + 1e-6:
                    output_end = min(active_end, cursor + output_target_seconds)
                    if output_end - cursor < self.framewise_output_min_seconds:
                        break
                    output_windows.append((cursor, output_end))
                    cursor = output_end

            for output_start, output_end in output_windows:
                out_first = max(0, int(np.floor(output_start * fps)) - 2)
                out_last = min(len(values), int(np.ceil(output_end * fps)) - 2)
                if out_last <= out_first:
                    continue
                high_mask = original_high[out_first:out_last]
                high_ratio = float(np.mean(high_mask))
                if high_ratio < self.framewise_min_high_ratio:
                    continue
                high_values = values[out_first:out_last][high_mask]
                confidence = float(np.median(high_values)) if high_values.size else -99.0
                segment_offset = 0
                if offsets is not None:
                    high_offsets = offsets[out_first:out_last][high_mask]
                    if high_offsets.size == 0:
                        continue
                    unique_offsets, offset_counts = np.unique(high_offsets, return_counts=True)
                    best_offset_index = int(np.argmax(offset_counts))
                    segment_offset = int(unique_offsets[best_offset_index])
                    stability = float(offset_counts[best_offset_index] / high_offsets.size)
                    if stability < self.local_offset_min_stability_ratio:
                        continue
                segments.append(SyncNetSegment(
                    start=float(output_start),
                    end=float(output_end),
                    confidence=confidence,
                    high_ratio=high_ratio,
                    offset_frames=segment_offset,
                ))
        return tuple(segments)

    def _crop_box(self, bbox: np.ndarray, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
        x1, y1, x2, y2 = map(float, bbox)
        if x2 <= x1 or y2 <= y1:
            return None
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1) * (1.0 + 2.0 * self.crop_margin)
        x1 = max(0, int(round(cx - side / 2.0)))
        y1 = max(0, int(round(cy - side / 2.0)))
        x2 = min(width, int(round(cx + side / 2.0)))
        y2 = min(height, int(round(cy + side / 2.0)))
        if x2 - x1 < 32 or y2 - y1 < 32:
            return None
        return x1, y1, x2, y2

    def _make_track_clip(
        self,
        video_path: Path,
        global_start: float,
        duration: float,
        local_start: float,
        track: Dict,
        frame_paths: Sequence[Path],
        detect_fps: float,
        key: str,
    ) -> Path:
        if not frame_paths:
            raise RuntimeError("No hay frames de detección para escalar el track facial")
        detection_frame = cv2.imread(str(frame_paths[0]))
        if detection_frame is None:
            raise RuntimeError(f"No se puede leer {frame_paths[0]}")
        detect_h, detect_w = detection_frame.shape[:2]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"No se puede abrir el vídeo {video_path}")
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if source_w <= 0 or source_h <= 0:
            cap.release()
            raise RuntimeError(f"Dimensiones inválidas en {video_path}")

        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        silent_path = self.temp_dir / f"{safe_key}_silent.mp4"
        av_path = self.temp_dir / f"{safe_key}_syncnet.mp4"
        writer = cv2.VideoWriter(
            str(silent_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.crop_fps,
            (self.crop_size, self.crop_size),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"No se puede crear el recorte temporal {silent_path}")

        first_source_frame = max(0, int(math.floor(float(global_start) * source_fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_source_frame)
        current_source_frame = first_source_frame - 1
        current_image = None
        written = 0
        total = max(1, int(math.floor(float(duration) * self.crop_fps)))
        scale = np.asarray([source_w / detect_w, source_h / detect_h] * 2, dtype=np.float32)

        try:
            for output_index in range(total):
                relative_time = output_index / self.crop_fps
                wanted_source_frame = int(round((float(global_start) + relative_time) * source_fps))
                while current_source_frame < wanted_source_frame:
                    ok, image = cap.read()
                    current_source_frame += 1
                    if not ok:
                        current_image = None
                        break
                    current_image = image
                if current_image is None:
                    continue
                bbox = self._interpolated_bbox(
                    track,
                    local_time=float(local_start) + relative_time,
                    detect_fps=detect_fps,
                )
                if bbox is None:
                    continue
                crop_box = self._crop_box(bbox * scale, source_w, source_h)
                if crop_box is None:
                    continue
                x1, y1, x2, y2 = crop_box
                crop = current_image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA)
                writer.write(crop)
                written += 1
        finally:
            writer.release()
            cap.release()

        if written < max(10, int(total * 0.80)):
            silent_path.unlink(missing_ok=True)
            raise RuntimeError(f"Track facial incompleto para SyncNet: {written}/{total} frames")

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(silent_path),
            "-ss", f"{float(global_start):.3f}", "-i", str(video_path),
            "-t", f"{float(duration):.3f}",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-ar", "16000", "-ac", "1", "-shortest",
            str(av_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"No se pudo añadir audio al recorte SyncNet: {(exc.stderr or '')[-1000:]}") from exc
        finally:
            silent_path.unlink(missing_ok=True)
        return av_path

    def verify_candidate(
        self,
        *,
        video_path: Path,
        global_start: float,
        duration: float,
        local_start: float,
        track: Dict,
        frame_paths: Sequence[Path],
        detect_fps: float,
        key: str,
    ) -> SyncNetResult:
        if not self.enabled:
            return SyncNetResult(not self.required, 0.0, 999, float("inf"), "disabled")
        if self._pipeline is None:
            self.validate_ready()
        clip_path: Optional[Path] = None
        try:
            clip_path = self._make_track_clip(
                video_path=Path(video_path),
                global_start=global_start,
                duration=duration,
                local_start=local_start,
                track=track,
                frame_paths=frame_paths,
                detect_fps=detect_fps,
                key=key,
            )
            reference = clip_path.stem
            options = SimpleNamespace(
                tmp_dir=str(self.temp_dir / "official_eval"),
                reference=reference,
                batch_size=int(getattr(self.config, "syncnet_batch_size", 20)),
                vshift=int(getattr(self.config, "syncnet_vshift", 10)),
            )
            Path(options.tmp_dir).mkdir(parents=True, exist_ok=True)
            offset_raw, confidence_raw, distances_raw = self._pipeline.evaluate(options, str(clip_path))
            confidence = float(np.asarray(confidence_raw).reshape(-1)[0])
            offset = int(np.asarray(offset_raw).reshape(-1)[0])
            distances = np.asarray(distances_raw, dtype=float)
            distance = float(np.nanmin(distances)) if distances.size else float("inf")
            if self.framewise_enabled:
                if self.local_offset_enabled:
                    frame_confidence, frame_offsets = self._local_framewise_from_distances(distances)
                    segments = self._segments_from_frame_confidence(
                        frame_confidence,
                        float(duration),
                        frame_offsets=frame_offsets,
                    )
                    accepted = bool(segments)
                    if segments:
                        representative = max(segments, key=lambda item: item.confidence)
                        offset = int(representative.offset_frames)
                    reason = "local_framewise_ok" if accepted else "no_sustained_local_sync"
                else:
                    frame_confidence = self._framewise_from_distances(distances)
                    segments = self._segments_from_frame_confidence(frame_confidence, float(duration))
                    accepted = bool(segments) and abs(offset) <= self.max_abs_offset
                    reason = "framewise_ok" if accepted else "no_sustained_framewise_sync_or_offset"
            else:
                frame_confidence = self._framewise_from_distances(distances)
                accepted = confidence >= self.confidence_threshold and abs(offset) <= self.max_abs_offset
                segments = (
                    (SyncNetSegment(0.0, float(duration), confidence, 1.0),)
                    if accepted else ()
                )
                reason = "ok" if accepted else "low_confidence_or_offset"
            return SyncNetResult(
                accepted,
                confidence,
                offset,
                distance,
                reason,
                segments=segments,
                frame_confidence=tuple(float(value) for value in frame_confidence),
            )
        except Exception as exc:
            if self.required:
                raise
            logging.warning("SyncNet rechazó %s por error: %s", key, exc)
            return SyncNetResult(False, 0.0, 999, float("inf"), f"error:{type(exc).__name__}")
        finally:
            if clip_path is not None and not self.keep_crops:
                clip_path.unlink(missing_ok=True)
                shutil.rmtree(self.temp_dir / "official_eval" / clip_path.stem, ignore_errors=True)
