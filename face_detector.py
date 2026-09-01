#Detección facial y extracción de embeddings usando InsightFace/ArcFace.

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceDetector:
    """Detector facial con embeddings ArcFace/InsightFace."""

    def __init__(self, config):
        self.config = config
        self.det_size = tuple(getattr(config, "face_det_size", (640, 640)))
        self.min_face_size = int(getattr(config, "min_face_size", 80))

        device_str = str(getattr(config, "device", "cpu"))
        use_cuda = "cuda" in device_str.lower()
        ctx_id = 0 if use_cuda else -1

        models_dir = Path(getattr(config, "models_dir", Path("models"))).expanduser()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]

        try:
            self.app = FaceAnalysis(
                name="buffalo_l",
                root=str(models_dir / "insightface"),
                providers=providers,
                # El pipeline solo necesita la caja facial y el embedding de
                # identidad. Evitamos mantener en VRAM edad/genero y los
                # modelos de landmarks densos para dejar espacio a SyncNet.
                allowed_modules=["detection", "recognition"],
            )
            self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)
            print(f"✅ FaceDetector inicializado con InsightFace ({device_str})")
        except Exception as e:
            print(f"❌ Error inicializando FaceDetector: {e}")
            raise

    def _load_image(self, img_path: Path) -> Optional[np.ndarray]:
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))
        if img is not None:
            return img

        try:
            from PIL import Image
            img_pil = Image.open(str(img_path)).convert("RGB")
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"⚠️ No se pudo leer imagen: {img_path} ({e})")
            return None

    @staticmethod
    def _norm_embedding(emb) -> Optional[np.ndarray]:
        if emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(emb)
        if not np.isfinite(norm) or norm < 1e-8:
            return None
        return emb / norm

    def detect(self, img_path: Path) -> List[Dict[str, Any]]:
        img = self._load_image(Path(img_path))
        if img is None:
            return []

        try:
            faces = self.app.get(img)
            results: List[Dict[str, Any]] = []
            for face in faces:
                bbox = face.bbox.astype(np.float32)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w < self.min_face_size or h < self.min_face_size:
                    continue

                emb = self._norm_embedding(getattr(face, "normed_embedding", None))
                if emb is None:
                    continue

                results.append({
                    "bbox": bbox,
                    "embedding": emb,
                    "landmarks": getattr(face, "kps", None),
                    "det_score": float(getattr(face, "det_score", 0.0)),
                    "gender": getattr(face, "gender", None),
                    "age": getattr(face, "age", None),
                })
            return results
        except Exception as e:
            print(f"❌ Error detectando caras en {img_path}: {e}")
            return []

    def get_embedding(self, face_data: Dict) -> Optional[np.ndarray]:
        return face_data.get("embedding")

    def get_bbox(self, face_data: Dict) -> np.ndarray:
        return np.asarray(face_data["bbox"], dtype=np.float32)

    def crop_face(self, img: np.ndarray, face_data: Dict, margin: float = 0.3) -> Optional[np.ndarray]:
        bbox = self.get_bbox(face_data).astype(int).copy()
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        margin_w, margin_h = int(w * margin), int(h * margin)
        x1 = max(0, bbox[0] - margin_w)
        y1 = max(0, bbox[1] - margin_h)
        x2 = min(img.shape[1], bbox[2] + margin_w)
        y2 = min(img.shape[0], bbox[3] + margin_h)
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2]
