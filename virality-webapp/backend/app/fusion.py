from __future__ import annotations

import numpy as np


def _pad_to_dim(vec: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros(int(dim), dtype=np.float32)
    out[: vec.shape[0]] = vec.astype(np.float32)
    return out


def fuse_vectors(
    strategy: str,
    video_vec: np.ndarray,
    audio_vec: np.ndarray,
    text_vec: np.ndarray,
    text_present: int,
    append_mask: bool = True,
) -> np.ndarray:
    strategy = str(strategy).strip().lower()
    v = np.asarray(video_vec, dtype=np.float32).reshape(-1)
    a = np.asarray(audio_vec, dtype=np.float32).reshape(-1)
    t = np.asarray(text_vec, dtype=np.float32).reshape(-1)

    if strategy == "concat":
        base = np.concatenate([v, a, t], axis=0).astype(np.float32)
    else:
        dim = max(v.shape[0], a.shape[0], t.shape[0])
        vp = _pad_to_dim(v, dim)
        ap = _pad_to_dim(a, dim)
        tp = _pad_to_dim(t, dim)
        if strategy == "sum_pool":
            base = (vp + ap + tp).astype(np.float32)
        elif strategy == "max_pool":
            base = np.maximum(np.maximum(vp, ap), tp).astype(np.float32)
        else:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")

    if append_mask:
        mask = np.array([float(int(text_present))], dtype=np.float32)
        return np.concatenate([base, mask], axis=0).astype(np.float32)
    return base.astype(np.float32)
