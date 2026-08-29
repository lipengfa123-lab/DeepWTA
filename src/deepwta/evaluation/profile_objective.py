from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


TACTICAL_PROFILES = {
    "balanced": {
        "ammo": 0.32,
        "crossing": 0.28,
        "distance": 0.22,
        "interception": 0.18,
    },
    "conflict_averse": {
        "ammo": 0.20,
        "crossing": 0.45,
        "distance": 0.18,
        "interception": 0.17,
    },
    "resource_efficient": {
        "ammo": 0.45,
        "crossing": 0.18,
        "distance": 0.22,
        "interception": 0.15,
    },
    "time_risk_balanced": {
        "ammo": 0.20,
        "crossing": 0.18,
        "distance": 0.32,
        "interception": 0.30,
    },
}

for _name, _weights in TACTICAL_PROFILES.items():
    _total = sum(float(value) for value in _weights.values())
    if abs(_total - 1.0) > 1e-12:
        raise ValueError("profile %s does not sum to 1.0" % _name)


# The four tactical weights should be interpreted as relative preferences, not
# as a license for the numerically largest metric to dominate the scalar score.
# Training used EMA loss normalization. For integer downstream evaluation we
# therefore normalize every operational metric before applying the same profile
# weights.
#
# Ammo uses a scenario-specific, method-independent optimistic raw-ammo lower
# bound. This avoids the denominator gaming found in ammo_norm when the solver
# is free to change schemes: selecting a much larger-demand scheme must not look
# resource-efficient merely because raw_ammo / selected_demand becomes smaller.
#
# The remaining scales are frozen rounded reference magnitudes from the already
# established DeepWTA validation regime. They are intentionally shared by every
# method and every tactical style; they are not re-fit to the method being
# evaluated.
DOWNSTREAM_REFERENCE_SCALES = {
    "crossing_pair_rate": 0.10,
    "avg_distance": 0.515,
    "avg_interception": 0.132,
}

LEGACY_TERM_FIELDS = {
    "ammo": "ammo_norm",
    "crossing": "crossing_pair_rate",
    "distance": "avg_distance",
    "interception": "avg_interception",
}


def tactical_weights(style: str) -> Dict[str, float]:
    key = str(style).strip().lower()
    if key not in TACTICAL_PROFILES:
        raise ValueError(
            "unknown tactical style %r; choose from %s"
            % (style, ",".join(TACTICAL_PROFILES))
        )
    return dict(TACTICAL_PROFILES[key])


def raw_ammo_capacity_lower_bound(
    B,
    U,
    G,
    G_th,
    Q,
    X,
    tol: float = 1e-7,
) -> float:
    """Optimistic method-independent raw-shot lower bound.

    Each target is allowed to use the full original inventory independently,
    so cross-target inventory competition is relaxed. For every legal scheme
    and weapon demand, available bases are consumed in descending survival
    probability. The bound is therefore optimistic and safe as a normalization
    reference; it is not claimed to be the exact global optimum.
    """
    B = np.asarray(B, dtype=float)
    U = np.asarray(U, dtype=float)
    G = np.asarray(G, dtype=float)
    G_th = np.asarray(G_th, dtype=float)
    Q = np.asarray(Q, dtype=float)
    X = np.asarray(X, dtype=float)

    if Q.ndim != 3:
        raise ValueError("Q must have shape [K,N,M]")
    K, N, M = Q.shape
    if B.shape != (K, N):
        raise ValueError("B shape mismatch")
    if U.ndim != 3 or U.shape[1:] != (K, M):
        raise ValueError("U shape mismatch")
    L = U.shape[0]
    if G.shape != (L, M) or G_th.shape != (M,):
        raise ValueError("G/G_th shape mismatch")
    if X.shape != (K, N, M):
        raise ValueError("X shape mismatch")

    cap = np.maximum(np.floor(B + tol), 0.0).astype(np.int64)
    total = 0.0

    for m in range(M):
        valid = np.flatnonzero(G[:, m] >= G_th[m] - tol)
        if valid.size == 0:
            return float("inf")
        target_best = float("inf")

        for scheme in valid.tolist():
            scheme_shots = 0.0
            feasible = True
            for k in range(K):
                remaining = float(max(U[scheme, k, m], 0.0))
                if remaining <= tol:
                    continue

                survival = 1.0 - Q[k, :, m]
                bases = np.flatnonzero(
                    (X[k, :, m] > 0.5)
                    & (cap[k] > 0)
                    & (survival > tol)
                )
                if bases.size == 0:
                    feasible = False
                    break
                order = bases[np.argsort(-survival[bases], kind="stable")]

                for n in order.tolist():
                    if remaining <= tol:
                        break
                    s = float(survival[n])
                    need = int(np.ceil(max(remaining - tol, 0.0) / max(s, tol)))
                    shots = min(need, int(cap[k, n]))
                    if shots <= 0:
                        continue
                    scheme_shots += float(shots)
                    remaining -= shots * s

                if remaining > tol:
                    feasible = False
                    break

            if feasible:
                target_best = min(target_best, scheme_shots)

        if not np.isfinite(target_best):
            return float("inf")
        total += target_best

    return float(max(total, 1e-12))


def attach_profile_reference_metrics(
    metrics: Mapping[str, float],
    *,
    B,
    U,
    G,
    G_th,
    Q,
    X,
    tol: float = 1e-7,
) -> Dict[str, float]:
    result = dict(metrics)
    lb = raw_ammo_capacity_lower_bound(B, U, G, G_th, Q, X, tol=tol)
    ammo = float(result.get("ammo", float("nan")))
    if not np.isfinite(lb) or lb <= 0.0:
        ammo_ratio = float("inf")
    else:
        ammo_ratio = ammo / lb
    result["raw_ammo_capacity_lb"] = float(lb)
    result["ammo_ratio_to_raw_lb"] = float(ammo_ratio)
    return result


def _legacy_score_breakdown(metrics: Mapping[str, float], style: str) -> Dict[str, float]:
    weights = tactical_weights(style)
    total = 0.0
    out = {}
    for objective_name, metric_name in LEGACY_TERM_FIELDS.items():
        value = float(metrics[metric_name])
        contribution = float(weights[objective_name]) * value
        out["legacy_%s_term" % objective_name] = value
        out["legacy_%s_contribution" % objective_name] = contribution
        total += contribution
    out["legacy_profile_score"] = float(total)
    return out


def profile_score_breakdown(
    metrics: Mapping[str, float],
    style: str,
) -> Dict[str, float]:
    """EMA-like normalized downstream tactical score; lower is better.

    z_ammo = raw_ammo / scenario_raw_ammo_capacity_lb
    z_cross = crossing_pair_rate / 0.10
    z_distance = avg_distance / 0.515
    z_interception = avg_interception / 0.132

    The tactical style weights are then applied directly to the normalized z
    terms. Raw metrics and the old unscaled score are retained for auditing.
    """
    weights = tactical_weights(style)

    required = (
        "ammo_ratio_to_raw_lb",
        "crossing_pair_rate",
        "avg_distance",
        "avg_interception",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise KeyError("missing profile metrics: %s" % ",".join(missing))

    terms = {
        "ammo": float(metrics["ammo_ratio_to_raw_lb"]),
        "crossing": float(metrics["crossing_pair_rate"])
        / DOWNSTREAM_REFERENCE_SCALES["crossing_pair_rate"],
        "distance": float(metrics["avg_distance"])
        / DOWNSTREAM_REFERENCE_SCALES["avg_distance"],
        "interception": float(metrics["avg_interception"])
        / DOWNSTREAM_REFERENCE_SCALES["avg_interception"],
    }

    contributions = {
        name: float(weights[name]) * float(value)
        for name, value in terms.items()
    }
    score = float(sum(contributions.values()))

    result = {
        "profile_score": score,
        "profile_score_version": "operational-normalized-v3",
        "profile_ammo_term": float(terms["ammo"]),
        "profile_crossing_term": float(terms["crossing"]),
        "profile_distance_term": float(terms["distance"]),
        "profile_interception_term": float(terms["interception"]),
        "profile_ammo_contribution": float(contributions["ammo"]),
        "profile_crossing_contribution": float(contributions["crossing"]),
        "profile_distance_contribution": float(contributions["distance"]),
        "profile_interception_contribution": float(contributions["interception"]),
        "profile_weight_ammo": float(weights["ammo"]),
        "profile_weight_crossing": float(weights["crossing"]),
        "profile_weight_distance": float(weights["distance"]),
        "profile_weight_interception": float(weights["interception"]),
        "profile_scale_crossing_pair_rate": float(
            DOWNSTREAM_REFERENCE_SCALES["crossing_pair_rate"]
        ),
        "profile_scale_avg_distance": float(
            DOWNSTREAM_REFERENCE_SCALES["avg_distance"]
        ),
        "profile_scale_avg_interception": float(
            DOWNSTREAM_REFERENCE_SCALES["avg_interception"]
        ),
    }
    if all(name in metrics for name in LEGACY_TERM_FIELDS.values()):
        result.update(_legacy_score_breakdown(metrics, style))
    return result


def profile_score(metrics: Mapping[str, float], style: str) -> float:
    return float(profile_score_breakdown(metrics, style)["profile_score"])
