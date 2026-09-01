#Filtros de calidad de audio: VAD, SNR y solapamiento.


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import scipy.signal as scipy_signal

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    reason: str
    duration: float
    vad_ratio: float
    snr_db: float
    rms: float
    peak: float
    clipping_ratio: float
    quality_score: float


class QualityFilter:

    def __init__(self, config):
        self.vad_aggressiveness = int(getattr(config, "vad_aggressiveness", 2))
        self.min_snr_db = float(getattr(config, "min_snr_db", 8.0))
        self.max_overlap_ratio = float(getattr(config, "max_overlap_ratio", 0.10))
        self.min_speech_ratio = float(getattr(config, "min_speech_ratio", 0.55))
        self.min_rms = float(getattr(config, "min_rms", 0.003))
        self.max_clipping_ratio = float(getattr(config, "max_clipping_ratio", 0.02))
        self.frame_ms = int(getattr(config, "vad_frame_ms", 30))
        self.vad = webrtcvad.Vad(self.vad_aggressiveness) if WEBRTC_VAD_AVAILABLE else None

    def check_audio(self, audio: np.ndarray, sr: int) -> QualityResult:
        """Evalúa un fragmento y devuelve la interfaz usada por el pipeline."""
        samples = self._to_float_mono(audio)
        metrics = self.score_chunk(samples, sr)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        clipping_ratio = float(np.mean(np.abs(samples) >= 0.999)) if samples.size else 0.0

        failures = []
        if metrics["duration"] <= 0.0:
            failures.append("audio vacío")
        if metrics["rms"] < self.min_rms:
            failures.append("RMS bajo")
        if metrics["vad_ratio"] < self.min_speech_ratio:
            failures.append("poca voz")
        if metrics["snr_db"] < self.min_snr_db:
            failures.append("SNR bajo")
        if clipping_ratio > self.max_clipping_ratio:
            failures.append("clipping")

        speech_score = np.clip(
            (metrics["vad_ratio"] - self.min_speech_ratio) /
            max(1e-6, 1.0 - self.min_speech_ratio), 0.0, 1.0
        )
        snr_score = np.clip((metrics["snr_db"] - self.min_snr_db) / 20.0, 0.0, 1.0)
        level_score = np.clip(metrics["rms"] / max(self.min_rms * 4.0, 1e-6), 0.0, 1.0)
        clip_score = 1.0 - np.clip(clipping_ratio / max(self.max_clipping_ratio, 1e-6), 0.0, 1.0)
        quality_score = float(0.40 * speech_score + 0.30 * snr_score + 0.20 * level_score + 0.10 * clip_score)

        return QualityResult(
            ok=not failures,
            reason="ok" if not failures else ", ".join(failures),
            duration=float(metrics["duration"]),
            vad_ratio=float(metrics["vad_ratio"]),
            snr_db=float(metrics["snr_db"]),
            rms=float(metrics["rms"]),
            peak=peak,
            clipping_ratio=clipping_ratio,
            quality_score=quality_score,
        )

    @staticmethod
    def _to_float_mono(audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.5:  # probable PCM no normalizado
            audio = audio / peak
        return audio

    def _resample_to_16k(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        if sr == 16000:
            return audio.astype(np.float32), sr
        target_sr = 16000
        n = max(1, int(len(audio) * target_sr / sr))
        resampled = scipy_signal.resample(audio, n)
        return resampled.astype(np.float32), target_sr

    def _rms_vad(self, audio: np.ndarray, sr: int) -> np.ndarray:
        frame_len = max(1, int(sr * self.frame_ms / 1000))
        step = max(1, frame_len // 2)
        if len(audio) < frame_len:
            return np.ones(len(audio), dtype=bool)

        rms = []
        starts = []
        for i in range(0, len(audio) - frame_len + 1, step):
            frame = audio[i:i + frame_len]
            rms.append(np.sqrt(np.mean(frame**2) + 1e-12))
            starts.append(i)

        rms = np.asarray(rms, dtype=np.float32)
        # Umbral adaptativo con mínimo absoluto para evitar marcar ruido como voz.
        threshold = max(float(np.percentile(rms, 35)), self.min_rms)
        mask = np.zeros(len(audio), dtype=bool)
        for i, r in zip(starts, rms):
            if r >= threshold:
                mask[i:min(i + frame_len, len(audio))] = True
        return mask

    def apply_vad(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = self._to_float_mono(audio)

        if self.vad is None:
            return self._rms_vad(audio, sr)

        audio_16k, sr_16k = self._resample_to_16k(audio, sr)
        frame_len = int(sr_16k * self.frame_ms / 1000)
        hop = frame_len // 2
        if len(audio_16k) < frame_len:
            return np.ones(len(audio), dtype=bool)

        mask_16k = np.zeros(len(audio_16k), dtype=bool)
        for i in range(0, len(audio_16k) - frame_len + 1, hop):
            frame = audio_16k[i:i + frame_len]
            pcm = np.clip(frame, -1.0, 1.0)
            pcm = (pcm * 32767).astype(np.int16).tobytes()
            try:
                is_speech = self.vad.is_speech(pcm, sr_16k)
            except Exception:
                is_speech = True
            if is_speech:
                mask_16k[i:i + frame_len] = True

        if len(mask_16k) != len(audio):
            mask = scipy_signal.resample(mask_16k.astype(np.float32), len(audio)) > 0.5
        else:
            mask = mask_16k
        return mask

    def estimate_snr(self, audio: np.ndarray, sr: int, vad_mask: Optional[np.ndarray] = None) -> float:
        audio = self._to_float_mono(audio)
        if vad_mask is None:
            vad_mask = self.apply_vad(audio, sr)

        if len(vad_mask) != len(audio):
            vad_mask = scipy_signal.resample(vad_mask.astype(np.float32), len(audio)) > 0.5

        if not np.any(vad_mask):
            return -99.0
        if not np.any(~vad_mask):
            # Sin zona clara de ruido; usamos un valor conservador aceptable.
            return 30.0

        speech_energy = float(np.mean(audio[vad_mask] ** 2) + 1e-12)
        noise_energy = float(np.mean(audio[~vad_mask] ** 2) + 1e-12)
        return float(10 * np.log10(speech_energy / noise_energy))

    @staticmethod
    def _overlap_ratio(seg_start: float, seg_end: float, diar_df: Optional[pd.DataFrame]) -> float:
        if diar_df is None or diar_df.empty:
            return 0.0
        required = {"start", "end"}
        if not required.issubset(set(diar_df.columns)):
            return 0.0

        total = max(0.0, seg_end - seg_start)
        if total <= 0:
            return 1.0

        events = []
        for _, row in diar_df.iterrows():
            s = max(seg_start, float(row["start"]))
            e = min(seg_end, float(row["end"]))
            if e > s:
                events.append((s, +1))
                events.append((e, -1))
        if not events:
            return 0.0

        events.sort()
        overlap = 0.0
        active = 0
        prev = events[0][0]
        for t, delta in events:
            if t > prev and active > 1:
                overlap += t - prev
            active += delta
            prev = t
        return float(overlap / total)

    def score_chunk(
        self,
        audio: np.ndarray,
        sr: int,
        diar_df: Optional[pd.DataFrame] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> dict:
        audio = self._to_float_mono(audio)
        duration = len(audio) / float(sr) if sr else 0.0
        rms = float(np.sqrt(np.mean(audio**2) + 1e-12)) if len(audio) else 0.0
        vad_mask = self.apply_vad(audio, sr) if len(audio) else np.array([], dtype=bool)
        vad_ratio = float(np.mean(vad_mask)) if len(vad_mask) else 0.0
        snr_db = self.estimate_snr(audio, sr, vad_mask) if len(audio) else -99.0
        overlap_ratio = self._overlap_ratio(start, end, diar_df) if start is not None and end is not None else 0.0

        accepted = (
            duration > 0.0
            and rms >= self.min_rms
            and vad_ratio >= self.min_speech_ratio
            and snr_db >= self.min_snr_db
            and overlap_ratio <= self.max_overlap_ratio
        )

        return {
            "duration": float(duration),
            "rms": rms,
            "vad_ratio": vad_ratio,
            "snr_db": float(snr_db),
            "overlap_ratio": float(overlap_ratio),
            "accepted": bool(accepted),
        }

    def filter_segments_array(
        self,
        segments_df: pd.DataFrame,
        audio: np.ndarray,
        sr: int,
        diar_df: Optional[pd.DataFrame] = None,
        debug: bool = False,
    ) -> pd.DataFrame:
        if segments_df.empty:
            return segments_df

        full_audio = self._to_float_mono(audio)
        filtered = []
        counters = {"short": 0, "rms": 0, "vad": 0, "snr": 0, "overlap": 0}

        for _, seg in segments_df.iterrows():
            start = float(seg["start"])
            end = float(seg["end"])
            s = max(0, int(start * sr))
            e = min(len(full_audio), int(end * sr))
            chunk = full_audio[s:e]

            metrics = self.score_chunk(chunk, sr, diar_df=diar_df, start=start, end=end)
            seg_id = seg.get("id", seg.get("utterance_id", "segmento"))

            if metrics["duration"] < 0.5:
                counters["short"] += 1
                continue
            if metrics["rms"] < self.min_rms:
                counters["rms"] += 1
                continue
            if metrics["vad_ratio"] < self.min_speech_ratio:
                counters["vad"] += 1
                if debug:
                    print(f"  VAD fail: {seg_id} ratio={metrics['vad_ratio']:.2f}")
                continue
            if metrics["snr_db"] < self.min_snr_db:
                counters["snr"] += 1
                continue
            if metrics["overlap_ratio"] > self.max_overlap_ratio:
                counters["overlap"] += 1
                continue

            out = seg.to_dict()
            out.update({k: v for k, v in metrics.items() if k != "accepted"})
            filtered.append(out)

        result = pd.DataFrame(filtered)
        print(
            f"Quality filter: {len(segments_df)} -> {len(result)} segmentos "
            f"(short:{counters['short']} rms:{counters['rms']} "
            f"vad:{counters['vad']} snr:{counters['snr']} overlap:{counters['overlap']})"
        )
        return result

    def filter_segments(
        self,
        segments_df: pd.DataFrame,
        audio_path: Path,
        diar_df: Optional[pd.DataFrame] = None,
        debug: bool = False,
    ) -> pd.DataFrame:
        if segments_df.empty:
            return segments_df
        sr, data = wavfile.read(str(audio_path))
        return self.filter_segments_array(segments_df, data, sr, diar_df=diar_df, debug=debug)
