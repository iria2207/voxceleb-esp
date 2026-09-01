#Tracking de caras por IoU, siguiendo la lógica VoxCeleb-style.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from .utils import compute_iou, l2_normalize


class FaceTracker:
    def __init__(self, config):
        self.iou_threshold = float(getattr(config, "iou_threshold", 0.45))
        self.min_association_iou = float(getattr(config, "track_min_association_iou", 0.15))
        self.min_association_similarity = float(getattr(config, "track_min_embedding_similarity", 0.30))
        self.min_track_frames = int(getattr(config, "min_track_frames", 10))
        self.max_lost_frames = int(getattr(config, "max_lost_frames", 5))
        self.fps = float(getattr(config, "detect_fps", getattr(config, "fps", 2)))
        self.quiet_progress = bool(getattr(config, "quiet_progress", True))

    def track(self, frame_paths: List[Path], detector, video_id: str) -> List[Dict[str, Any]]:
        tracks: Dict[str, Dict[str, Any]] = {}
        active_tracks: Dict[str, int] = {}

        disable_progress = bool(getattr(self, "quiet_progress", True))
        for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc=f"Tracking {video_id}", leave=False, disable=disable_progress)):
            faces = detector.detect(frame_path)
            matched_track_ids = set()

            for face in faces:
                bbox = detector.get_bbox(face)
                emb = detector.get_embedding(face)
                best_score, best_iou, best_tid = -1.0, 0.0, None

                for tid in list(active_tracks.keys()):
                    if tid in matched_track_ids:
                        continue
                    last_bbox = tracks[tid]["bboxes"][-1]
                    iou = compute_iou(bbox, last_bbox)
                    previous_embeddings = tracks[tid].get("embeddings", [])
                    similarity = float(np.dot(l2_normalize(emb), l2_normalize(previous_embeddings[-1]))) \
                        if emb is not None and previous_embeddings else 1.0
                    # El IoU puro cambia identidades cuando dos caras se cruzan.
                    # Exigimos compatibilidad ArcFace y permitimos algo más de
                    # movimiento entre frames cuando la identidad es coherente.
                    compatible = (
                        similarity >= self.min_association_similarity
                        and (iou >= self.min_association_iou or iou >= self.iou_threshold)
                    )
                    score = iou + 0.35 * max(0.0, similarity)
                    if compatible and score > best_score:
                        best_score, best_iou, best_tid = score, iou, tid

                if best_tid is not None:
                    tid = best_tid
                else:
                    tid = f"{video_id}_track_{len(tracks) + 1:04d}"
                    tracks[tid] = {
                        "id": tid,
                        "track_id": tid,
                        "video_id": video_id,
                        "bboxes": [],
                        "embeddings": [],
                        "embedding_scores": [],
                        "frames": [],
                        "det_scores": [],
                        "landmarks": [],
                    }

                tracks[tid]["bboxes"].append(np.asarray(bbox, dtype=np.float32))
                if emb is not None:
                    tracks[tid]["embeddings"].append(np.asarray(emb, dtype=np.float32))
                    tracks[tid]["embedding_scores"].append(float(face.get("det_score", 0.0)))
                tracks[tid]["frames"].append(int(frame_idx))
                tracks[tid]["det_scores"].append(float(face.get("det_score", 0.0)))
                tracks[tid]["landmarks"].append(face.get("landmarks"))
                active_tracks[tid] = frame_idx
                matched_track_ids.add(tid)

            for tid in list(active_tracks.keys()):
                if frame_idx - active_tracks[tid] > self.max_lost_frames:
                    del active_tracks[tid]

        valid_tracks: List[Dict[str, Any]] = []
        for tid, data in tracks.items():
            n_frames = len(data.get("frames", []))
            if n_frames < self.min_track_frames:
                continue
            if not data.get("embeddings"):
                continue

            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            scores = np.asarray(data.get("embedding_scores", []), dtype=np.float32)
            if scores.size == 0 or float(scores.sum()) <= 0:
                weights = np.ones(len(embeddings), dtype=np.float32) / max(1, len(embeddings))
            else:
                weights = scores
                weights = weights / (weights.sum() + 1e-8)

            similarities = embeddings @ embeddings.T
            medoid_index = int(np.argmax(np.median(similarities, axis=1)))
            inlier_mask = similarities[medoid_index] >= 0.45
            if not np.any(inlier_mask):
                continue
            inlier_embeddings = embeddings[inlier_mask]
            inlier_weights = weights[inlier_mask]
            inlier_weights = inlier_weights / (inlier_weights.sum() + 1e-8)
            avg_emb = l2_normalize(np.average(inlier_embeddings, axis=0, weights=inlier_weights))
            frames = data["frames"]
            data["avg_emb"] = avg_emb
            data["start_frame"] = int(frames[0])
            data["end_frame"] = int(frames[-1])
            data["frame_start"] = int(frames[0])
            data["frame_end"] = int(frames[-1])
            data["start_time"] = float(frames[0] / self.fps)
            data["end_time"] = float(frames[-1] / self.fps)
            data["duration"] = float(max(0.0, data["end_time"] - data["start_time"]))
            data["avg_det_score"] = float(np.mean(data["det_scores"])) if data["det_scores"] else 0.0
            data["embedding_consistency"] = float(np.mean(inlier_mask))
            data["track_coverage"] = float(n_frames / max(1, frames[-1] - frames[0] + 1))
            valid_tracks.append(data)

        logging.info("📊 Tracking %s: %d tracks -> %d válidos", video_id, len(tracks), len(valid_tracks))
        return valid_tracks
