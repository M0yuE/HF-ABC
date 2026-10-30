
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def compute_key_metrics(pred: np.ndarray, true: np.ndarray, channel_indices: Sequence[int]) -> Dict[str, float]:
    if pred.ndim != 3 or true.ndim != 3:
        raise ValueError("Expected pred/true arrays with shape [N, H, C].")
    idx = [int(i) for i in channel_indices]
    pred_sel = pred[:, :, idx]
    true_sel = true[:, :, idx]
    err = pred_sel - true_sel
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    return {
        "key_mse": mse,
        "key_mae": mae,
        "key_rmse": rmse,
        "n_key_channels": len(idx),
    }


def load_manifest(path: str | Path) -> Dict[str, object]:
    # Support both plain UTF-8 and UTF-8 files that were saved with BOM on Windows.
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def summarize_result_dir(result_dir: str | Path, manifest_path: str | Path | None = None) -> Dict[str, float]:
    result_dir = Path(result_dir)
    pred = np.load(result_dir / 'pred.npy')
    true = np.load(result_dir / 'true.npy')
    out: Dict[str, float] = {}
    if manifest_path is not None:
        manifest = load_manifest(manifest_path)
        key_channels = manifest.get('key_channels', [])
        feature_columns = manifest.get('feature_columns', [])
        if key_channels:
            indices: List[int] = []
            for item in key_channels:
                if isinstance(item, int):
                    indices.append(int(item))
                elif item in feature_columns:
                    indices.append(feature_columns.index(item))
            if indices:
                out.update(compute_key_metrics(pred, true, indices))
    return out
