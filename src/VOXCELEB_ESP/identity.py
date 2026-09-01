#Gestión de identidades faciales para VoxCeleb-ESP.


from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class IdentityManager:
    def __init__(self, detector, celebrities, config):
        self.detector = detector
        self.config = config
        self.celebrities = celebrities
        self.threshold = float(getattr(config, "face_template_threshold", 0.65))
        self.margin = float(getattr(config, "face_template_margin", 0.08))
        self.min_track_consistency = float(getattr(config, "face_min_track_consistency", 0.75))
        self.min_template_images = int(
            getattr(config, "face_template_min_refs", getattr(config, "min_template_images", 1))
        )
        self.templates: Dict[str, np.ndarray] = {}
        self.template_counts: Dict[str, int] = {}
        self.template_vectors: Dict[str, List[np.ndarray]] = {}

        if bool(getattr(config, "use_face_classifier", False)):
            print("use_face_classifier=true, pero no se ha proporcionado un clasificador entrenado.")
            print("Se usa verificación por templates para evitar predicciones aleatorias.")

        gallery = list(celebrities)
        known = {str(getattr(c, "id", "")) for c in gallery}
        references_dir = Path(getattr(config, "references_dir", "references"))
        if references_dir.is_dir():
            grouped: Dict[str, List[Path]] = {}
            for path in sorted(references_dir.iterdir()):
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                speaker_id = path.stem.split("_", 1)[0]
                if speaker_id:
                    grouped.setdefault(speaker_id, []).append(path)
            for speaker_id, paths in grouped.items():
                if speaker_id not in known:
                    gallery.append(SimpleNamespace(id=speaker_id, reference_images=paths))
        self._build_templates(gallery)

    @staticmethod
    def _norm(emb: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(emb)
        if not np.isfinite(n) or n < 1e-8:
            return None
        return emb / n

    def _choose_reference_face(self, faces: List[Dict]) -> Optional[Dict]:
        if not faces:
            return None
        # Elegimos la cara más grande y con mejor score.
        def score(face):
            bbox = np.asarray(face.get("bbox", [0, 0, 0, 0]), dtype=np.float32)
            area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
            return area * float(face.get("det_score", 0.0))
        return max(faces, key=score)

    def _build_templates(self, celebrities) -> None:
        for celeb in celebrities:
            embeddings = []
            for img_path in getattr(celeb, "reference_images", []):
                p = Path(img_path)
                faces = self.detector.detect(p)
                face = self._choose_reference_face(faces)
                if face is None:
                    print(f"⚠️ Sin cara válida en referencia: {p}")
                    continue

                emb = self._norm(self.detector.get_embedding(face))
                if emb is not None:
                    embeddings.append(emb)

            if len(embeddings) >= self.min_template_images:
                matrix = np.vstack(embeddings)
                similarities = matrix @ matrix.T
                medoid = int(np.argmax(np.median(similarities, axis=1)))
                inliers = matrix[similarities[medoid] >= 0.45]
                template = self._norm(np.mean(inliers, axis=0))
                if template is not None:
                    self.templates[celeb.id] = template
                    self.template_vectors[celeb.id] = [self._norm(v) for v in inliers]
                    self.template_counts[celeb.id] = len(inliers)
                    print(f"Template {celeb.id} creado con {len(inliers)} imágenes")
            else:
                print(f"No se pudo crear template robusto para {getattr(celeb, 'id', 'unknown')}")

    def identify_track(self, track_emb: np.ndarray) -> Tuple[str, float, float, float]:
        emb = self._norm(track_emb)
        if emb is None or not self.templates:
            return "unknown", 0.0, -1.0, 0.0

        best_id = "unknown"
        best_sim = -1.0
        scores = []
        for spk_id, template in self.templates.items():
            sim = float(cosine_similarity([emb], [template])[0, 0])
            scores.append((sim, spk_id))
            if sim > best_sim:
                best_sim = sim
                best_id = spk_id

        scores.sort(reverse=True)
        second_sim = scores[1][0] if len(scores) > 1 else -1.0
        margin = best_sim - second_sim if len(scores) > 1 else best_sim
        if best_sim >= self.threshold and margin >= self.margin:
            return best_id, best_sim, second_sim, margin
        return "unknown", best_sim, second_sim, margin

    def assign_identities(self, tracks: List[Dict]) -> List[Dict]:
        for track in tracks:
            spk_id, conf, other_conf, margin = self.identify_track(track.get("avg_emb"))
            consistency = float(track.get("embedding_consistency", 0.0))
            if consistency < self.min_track_consistency:
                spk_id = "unknown"
            track["speaker_id"] = spk_id
            track["face_conf"] = float(conf)
            track["face_other_conf"] = float(other_conf)
            track["face_margin"] = float(margin)
            track["face_track_consistency"] = consistency
            track["is_identified"] = spk_id != "unknown"
        return tracks
