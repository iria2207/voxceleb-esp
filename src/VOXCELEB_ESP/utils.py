#Funciones utilitarias y métricas básicas para el pipeline.

import numpy as np
from pathlib import Path
from typing import List, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import hashlib


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return float(np.dot(emb1, emb2) / (norm1 * norm2))


def compute_euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.linalg.norm(np.array(emb1) - np.array(emb2)))


def parallel_process(items: List[Any], process_func: Callable, max_workers: int = 4) -> List[Any]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_func, item): item for item in items}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error en tarea paralela: {e}")
                results.append(None)
    return results


def calculate_eer(labels: List[int], scores: List[float]) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return float(eer)
    except:
        return float('nan')


def hash_file(filepath: Path, algorithm: str = 'sha256') -> str:
    hash_func = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def sha1_file(filepath: Path) -> str:
    """Calcula el SHA-1 usado para deduplicar audios candidatos."""
    return hash_file(Path(filepath), algorithm="sha1")
