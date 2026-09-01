#Detección de hablante activo para VoxCeleb-ESP.


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ActiveSpeakerWindow:
    start: float
    end: float
    confidence: float
    audio_score: float
    visual_score: float
    method: str = "av_energy_mouth_motion"


class ActiveSpeakerVerifier:


    def __init__(self, config):
        self.config = config
        self.fps = float(getattr(config, "detect_fps", getattr(config, "fps", 25)) or 25)
        self.threshold = float(getattr(config, "active_proposal_threshold", 0.30))
        self.min_confidence = float(getattr(config, "syncnet_min_confidence", 0.55))

        self.window_seconds = float(getattr(config, "active_window_seconds", 0.64))
        self.hop_seconds = float(getattr(config, "active_hop_seconds", 0.24))
        self.merge_gap_seconds = float(getattr(config, "active_merge_gap_seconds", 0.48))
        self.max_bbox_gap_frames = int(getattr(config, "active_max_bbox_gap_frames", 3))
        self.fallback_to_track_windows = bool(getattr(config, "active_fallback_to_track_windows", True))

        self.min_audio_score = float(getattr(config, "active_min_audio_score", 0.20))
        self.min_visual_score = float(getattr(config, "active_min_visual_score", 0.04))
        self.max_frames_per_window = int(getattr(config, "active_max_frames_per_window", 12))

        print("ActiveSpeakerVerifier inicializado (audio + movimiento de boca)")

    @staticmethod
    def _as_xyxy_bbox(bbox: np.ndarray) -> np.ndarray:
        bbox = np.asarray(bbox, dtype=np.float32).copy()
        if bbox.shape[0] != 4:
            raise ValueError(f"bbox inválida: {bbox}")
        # Si parece formato x,y,w,h lo convertimos a x1,y1,x2,y2.
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            bbox[2] = bbox[0] + max(1.0, bbox[2])
            bbox[3] = bbox[1] + max(1.0, bbox[3])
        return bbox

    def _track_bbox_at(self, track: Dict, frame_idx: int) -> Optional[np.ndarray]:
        frames = list(track.get("frames", []))
        bboxes = list(track.get("bboxes", []))
        if not frames or not bboxes:
            bbox = track.get("bbox")
            return self._as_xyxy_bbox(bbox) if bbox is not None else None

        # Usamos la bbox del frame detectado más cercano.
        pos = int(np.argmin(np.abs(np.asarray(frames) - int(frame_idx))))
        if abs(int(frames[pos]) - int(frame_idx)) > self.max_bbox_gap_frames:
            return None
        pos = min(pos, len(bboxes) - 1)
        try:
            return self._as_xyxy_bbox(np.asarray(bboxes[pos]))
        except Exception:
            return None

    @staticmethod
    def extract_mouth_roi(frame: np.ndarray, bbox: np.ndarray, margin: float = 0.35) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None

        x1, y1, x2, y2 = bbox.astype(int)
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        w, h = x2 - x1, y2 - y1
        mx1 = x1 + int(0.20 * w)
        mx2 = x2 - int(0.20 * w)
        my1 = y1 + int(0.55 * h)
        my2 = y1 + int(0.92 * h)

        mw, mh = mx2 - mx1, my2 - my1
        mx1 = max(0, mx1 - int(mw * margin))
        mx2 = min(w_img, mx2 + int(mw * margin))
        my1 = max(0, my1 - int(mh * margin))
        my2 = min(h_img, my2 + int(mh * margin))

        roi = frame[my1:my2, mx1:mx2]
        if roi.size == 0:
            return None
        roi = cv2.resize(roi, (96, 96), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    @staticmethod
    def _audio_rms(audio: np.ndarray, sr: int, start: float, end: float) -> float:
        s = max(0, int(start * sr))
        e = min(len(audio), int(end * sr))
        if e <= s:
            return 0.0
        chunk = audio[s:e].astype(np.float32)
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        return float(np.sqrt(np.mean(chunk**2) + 1e-12))

    def _audio_score(self, audio: np.ndarray, sr: int, start: float, end: float, global_floor: float) -> float:
        rms = self._audio_rms(audio, sr, start, end)
        # Escala logarítmica suave; 0 significa silencio relativo.
        score = (np.log1p(rms * 1000.0) - np.log1p(global_floor * 1000.0)) / 3.0
        return float(np.clip(score, 0.0, 1.0))

    def _visual_score(self, frame_paths: Sequence[Path], track: Dict, start: float, end: float) -> float:
        first = max(0, int(np.floor(start * self.fps)))
        last = min(len(frame_paths) - 1, int(np.ceil(end * self.fps)))
        if last <= first:
            return 0.0

        indices = np.linspace(first, last, num=min(self.max_frames_per_window, max(2, last - first + 1)))
        prev = None
        diffs: List[float] = []

        for idx_f in indices:
            idx = int(round(idx_f))
            if idx < 0 or idx >= len(frame_paths):
                continue

            frame = cv2.imread(str(frame_paths[idx]))
            if frame is None:
                continue

            bbox = self._track_bbox_at(track, idx)
            if bbox is None:
                continue

            roi = self.extract_mouth_roi(frame, bbox)
            if roi is None:
                continue

            if prev is not None:
                diff = np.mean(cv2.absdiff(prev, roi)) / 255.0
                diffs.append(float(diff))
            prev = roi

        if len(diffs) < 2:
            return 0.0

        # En vídeos reales, valores pequeños ya indican movimiento.
        raw = float(np.median(diffs))
        return float(np.clip(raw / 0.035, 0.0, 1.0))

    def score_window(
        self,
        frame_paths: Sequence[Path],
        audio: np.ndarray,
        audio_sr: int,
        track: Dict,
        start: float,
        end: float,
        global_floor: float,
    ) -> ActiveSpeakerWindow:
        a = self._audio_score(audio, audio_sr, start, end, global_floor)
        v = self._visual_score(frame_paths, track, start, end)

        if v <= 0.0:
            # Audio sin evidencia visual puede pertenecer al presentador, a una voz en off o a ruido. No debe convertirse en candidato del rostro.
            conf = 0.0
            method = "no_visual_rejected"
        else:
            # Requiere simultáneamente audio y boca. La media geométrica penaliza mucho cuando una de las dos señales es débil.
            conf = float(np.sqrt(max(a, 0.0) * max(v, 0.0)))
            method = "av_energy_mouth_motion"

        return ActiveSpeakerWindow(start, end, conf, a, v, method)

    def _global_noise_floor(self, audio: np.ndarray, sr: int) -> float:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        frame = max(1, int(0.20 * sr))
        hop = frame
        rms = []
        for i in range(0, max(1, len(audio) - frame), hop):
            chunk = audio[i:i + frame]
            if len(chunk) == frame:
                rms.append(np.sqrt(np.mean(chunk**2) + 1e-12))
        if not rms:
            return 1e-6
        return float(max(np.percentile(rms, 20), 1e-6))

    def _windows_to_segments(
        self,
        windows: List[ActiveSpeakerWindow],
        track_id: str,
        min_active_duration: float,
    ) -> List[Dict]:
        if not windows:
            return []

        active = [
            w for w in windows
            if w.confidence >= self.threshold
            and w.audio_score >= self.min_audio_score
            and w.visual_score >= self.min_visual_score
            and w.method == "av_energy_mouth_motion"
        ]

        if not active:
            return []

        segments = []
        cur_start = active[0].start
        cur_end = active[0].end
        cur_scores = [active[0].confidence]
        cur_audio = [active[0].audio_score]
        cur_visual = [active[0].visual_score]
        cur_methods = [active[0].method]

        for w in active[1:]:
            if w.start <= cur_end + self.merge_gap_seconds:
                cur_end = max(cur_end, w.end)
                cur_scores.append(w.confidence)
                cur_audio.append(w.audio_score)
                cur_visual.append(w.visual_score)
                cur_methods.append(w.method)
            else:
                if cur_end - cur_start >= min_active_duration:
                    segments.append({
                        "track_id": track_id,
                        "start": float(cur_start),
                        "end": float(cur_end),
                        "duration": float(cur_end - cur_start),
                        "confidence": float(np.mean(cur_scores)),
                        "audio_score": float(np.mean(cur_audio)),
                        "visual_score": float(np.mean(cur_visual)),
                        "method": max(set(cur_methods), key=cur_methods.count),
                    })
                cur_start, cur_end = w.start, w.end
                cur_scores = [w.confidence]
                cur_audio = [w.audio_score]
                cur_visual = [w.visual_score]
                cur_methods = [w.method]

        if cur_end - cur_start >= min_active_duration:
            segments.append({
                "track_id": track_id,
                "start": float(cur_start),
                "end": float(cur_end),
                "duration": float(cur_end - cur_start),
                "confidence": float(np.mean(cur_scores)),
                "audio_score": float(np.mean(cur_audio)),
                "visual_score": float(np.mean(cur_visual)),
                "method": max(set(cur_methods), key=cur_methods.count),
            })

        return segments

    def detect_active_speaker_segments(
        self,
        frame_paths: Sequence[Path],
        audio: np.ndarray,
        audio_sr: int,
        face_tracks: List[Dict],
        min_active_duration: Optional[float] = None,
        step_seconds: Optional[float] = None,
    ) -> List[Dict]:
        min_active_duration = float(
            min_active_duration if min_active_duration is not None
            else getattr(self.config, "segment_min_duration", 2.0)
        )
        hop = float(step_seconds if step_seconds is not None else self.hop_seconds)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        global_floor = self._global_noise_floor(audio, audio_sr)
        all_segments: List[Dict] = []

        for track in face_tracks:
            frames = list(track.get("frames", []))
            if not frames:
                continue

            track_id = str(track.get("track_id", track.get("id", "track")))
            t0 = float(min(frames) / self.fps)
            t1 = float(max(frames) / self.fps)
            if t1 <= t0:
                continue

            windows: List[ActiveSpeakerWindow] = []
            t = t0
            while t + self.window_seconds <= t1 + 1e-6:
                windows.append(
                    self.score_window(
                        frame_paths=frame_paths,
                        audio=audio,
                        audio_sr=audio_sr,
                        track=track,
                        start=t,
                        end=t + self.window_seconds,
                        global_floor=global_floor,
                    )
                )
                t += hop

            segs = self._windows_to_segments(windows, track_id, min_active_duration=min_active_duration)
            if not segs and self.fallback_to_track_windows:
                target = float(getattr(self.config, "segment_target_duration", 5.0))
                t = t0
                while t + min_active_duration <= t1 + 1e-6:
                    end = min(t + target, t1)
                    proposal = self.score_window(
                        frame_paths, audio, audio_sr, track, t, end, global_floor
                    )
                    if (
                        end - t >= min_active_duration
                        and proposal.audio_score >= self.min_audio_score
                        and proposal.visual_score >= max(0.01, self.min_visual_score * 0.5)
                    ):
                        segs.append({
                            "track_id": track_id,
                            "start": float(t),
                            "end": float(end),
                            "duration": float(end - t),
                            "confidence": float(proposal.confidence),
                            "audio_score": float(proposal.audio_score),
                            "visual_score": float(proposal.visual_score),
                            "method": "track_window_proposal",
                            "requires_manual_validation": False,
                        })
                    t += target
            for s in segs:
                s["speaker_id"] = track.get("speaker_id", "unknown")
                s["face_conf"] = float(track.get("face_conf", 0.0))
                s["face_other_conf"] = float(track.get("face_other_conf", -1.0))
                s["face_margin"] = float(track.get("face_margin", 0.0))
                s["face_track_consistency"] = float(track.get("face_track_consistency", 0.0))
                s["avg_det_score"] = float(track.get("avg_det_score", np.mean(track.get("det_scores", [0.0]))))
            all_segments.extend(segs)

        all_segments.sort(key=lambda x: (x["start"], -x["confidence"]))
        print(f"Active speaker: {len(face_tracks)} tracks -> {len(all_segments)} segmentos candidatos")
        return all_segments

    # Compatibilidad con versiones antiguas
    def verify(
        self,
        video_frames: List[np.ndarray],
        frame_times: List[float],
        audio_path: Path,
        face_track: Dict,
        min_duration: float = 2.0,
    ) -> Tuple[bool, float]:
        avg_det = float(np.mean(face_track.get("det_scores", [0.0])))
        conf = min(avg_det, 0.49)
        return conf >= self.threshold and min_duration <= face_track.get("duration", 0), conf

    def filter_active_tracks(
        self,
        tracks: List[Dict],
        video_frames: List[np.ndarray],
        frame_times: List[float],
        audio_path: Path,
    ) -> List[Dict]:
        filtered = []
        for track in tracks:
            if track.get("speaker_id") == "unknown":
                continue
            # Solo aceptamos tracks que ya vengan de una segmentación activa real.
            if track.get("active_speaker_conf", 0.0) >= self.min_confidence:
                filtered.append(track)
        print(f"Active speaker filter: {len(tracks)} -> {len(filtered)} tracks")
        return filtered
