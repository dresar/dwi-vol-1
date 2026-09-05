from __future__ import annotations

import numpy as np


def top_contributions(
    *,
    feature_names: list[str],
    values: np.ndarray,
    shap_values_1d: np.ndarray,
    top_k: int = 10,
) -> list[dict]:
    fn = list(feature_names)
    vals = np.asarray(values)
    sv = np.asarray(shap_values_1d)
    if vals.ndim > 1:
        vals = vals.reshape(-1)
    if sv.ndim == 2:
        if sv.shape[0] == len(fn):
            sv = sv.mean(axis=1)
        elif sv.shape[1] == len(fn):
            sv = sv.mean(axis=0)
        else:
            sv = sv.reshape(-1)
    elif sv.ndim > 2:
        sv = sv.reshape(-1)
    n = min(len(fn), len(vals), len(sv))
    pairs = []
    for i in range(n):
        name = fn[i]
        v = vals[i]
        s = float(sv[i])
        v_out = float(v) if isinstance(v, (np.floating, np.integer)) else str(v)
        pairs.append({"feature": name, "value": v_out, "shap_value": s})
    pairs.sort(key=lambda x: abs(float(x["shap_value"])), reverse=True)
    return pairs[: max(1, int(top_k))]
