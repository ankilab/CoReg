from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


def _hash_array(arr: np.ndarray | torch.Tensor) -> str:
    """
    Compute fast hash of array by hashing first and last chunks.
    For large arrays, we don't hash the entire array (too slow),
    but sample from beginning, middle, and end.
    """
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()

    arr = np.asarray(arr)
    flattened = arr.ravel().astype(np.float32)

    # Sample chunks to avoid hashing huge arrays
    chunk_size = max(1, len(flattened) // 3)
    samples = np.concatenate(
        [
            flattened[:chunk_size],
            flattened[len(flattened) // 2 : len(flattened) // 2 + chunk_size],
            flattened[-chunk_size:],
        ]
    )

    h = hashlib.sha256()
    h.update(samples.tobytes())
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    return h.hexdigest()


def _hash_params(params: Dict[str, Any]) -> str:
    """Hash method parameters."""
    h = hashlib.sha256()

    # Sort keys for consistent hashing
    for key in sorted(params.keys()):
        val = params[key]
        h.update(str(key).encode())

        if isinstance(val, (tuple, list)):
            for v in val:
                h.update(str(v).encode())
        else:
            h.update(str(val).encode())

    return h.hexdigest()


class RegistrationCache:
    """
    Cache registration results based on input video hash and parameters.
    Stores results in memory; cache is cleared when video changes.
    """

    def __init__(self):
        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.current_input_hash: Optional[str] = None

    def get_cache_key(self, method: str, params: Dict[str, Any] | None = None) -> str:
        """Generate cache key from method and parameters."""
        param_hash = _hash_params(params or {})
        return f"{method}_{param_hash}"

    def get(
        self, method: str, params: Dict[str, Any] | None = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Get cached registration result.

        Returns
        -------
        (warped, disp) tuple or None if not cached.
        """
        key = self.get_cache_key(method, params)
        if key in self.cache:
            return self.cache[key]
        return None

    def put(
        self,
        method: str,
        warped: np.ndarray,
        disp: np.ndarray,
        params: Dict[str, Any] | None = None,
    ) -> None:
        """Cache registration result."""
        key = self.get_cache_key(method, params)
        self.cache[key] = (warped, disp)

    def set_input(self, video_data: np.ndarray | torch.Tensor) -> None:
        """
        Register a new input video and clear cache.
        Called when user loads a new video file.
        """
        new_hash = _hash_array(video_data)
        if self.current_input_hash != new_hash:
            self.clear()
        self.current_input_hash = new_hash

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        self.current_input_hash = None

    def get_cache_info(self) -> str:
        """Return human-readable cache info."""
        return f"Cached registrations: {len(self.cache)}"
