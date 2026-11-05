"""
Cryptographic Verification Metrics & ISO/IEC 24745 Compliance.

Provides formal evaluations of:
1. System Unlinkability (D_sys) across distinct merchant tokens.
2. Irreversibility against Stolen-Token Inversion Attacks.

Copyright (c) 2026 Sovereign Biometrics Architecture Team.
Licensed under the Apache License, Version 2.0.
"""

import numpy as np
from typing import List, Tuple
from ..constants import (
    MAX_PERMISSIBLE_D_SYS,
    STOLEN_TOKEN_COSINE_CEILING,
)


def compute_system_unlinkability(
    mated_scores: np.ndarray,
    non_mated_scores: np.ndarray,
    num_bins: int = 100,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes ISO/IEC 24745 System Unlinkability metric D_sys.

    Mathematical Formulation:
        Let mated_scores be the normalized Hamming distances between distinct
        merchant tokens derived from the SAME subject.
        Let non_mated_scores be distances between tokens derived from DIFFERENT subjects.

        Estimate probability density functions: p(s | mated) and p(s | non_mated).
        The local linkability score is:
            D(s) = max(0, p(s | mated) - p(s | non_mated)) / (p(s | mated) + p(s | non_mated))
        The system-wide linkability metric is:
            D_sys = 2 * integral_{0}^{1} D(s) * p(s | mated) ds

        Benchmark Target: D_sys <= 0.011 (guaranteeing provable unlinkability).
    """
    assert len(mated_scores) > 0 and len(non_mated_scores) > 0

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    p_mated, _ = np.histogram(mated_scores, bins=bins, density=True)
    p_non_mated, _ = np.histogram(non_mated_scores, bins=bins, density=True)

    # Normalize to proper probability mass
    p_mated = p_mated / (np.sum(p_mated) + 1e-12)
    p_non_mated = p_non_mated / (np.sum(p_non_mated) + 1e-12)

    # Local linkability curve
    denominator = p_mated + p_non_mated + 1e-12
    d_local = np.maximum(0.0, (p_mated - p_non_mated) / denominator)

    # System-wide integral
    d_sys = float(2.0 * np.sum(d_local * p_mated))

    return d_sys, bin_centers, d_local


def simulate_stolen_token_linear_inversion(
    target_embedding: np.ndarray,
    ordinal_indices: np.ndarray,
    projection_matrices: np.ndarray,
) -> float:
    """
    Simulates a stolen-token attack where an adversary possesses:
    1. The ordinal hash indices t_k = argmax_{j in S_k} (w_{k, j}^T * z).
    2. The projection matrices W_k.

    The attacker's best linear approximation of z is the unweighted sum of the
    winning projection vectors:
        z_hat = sum_{k=1}^K w_{k, t_k}
        z_hat_norm = z_hat / ||z_hat||_2

    Returns:
        Cosine recovery similarity cos(z, z_hat).
        Must be < 0.28 to guarantee complete cryptographic failure of linear solvers.
    """
    assert target_embedding.ndim == 1
    assert len(ordinal_indices) == len(projection_matrices)

    # Attacker aggregates the winning vectors
    reconstructed_vector = np.zeros_like(target_embedding, dtype=np.float64)
    for k, t_k in enumerate(ordinal_indices):
        reconstructed_vector += projection_matrices[k, t_k]

    # Normalize reconstruction
    recon_norm = np.linalg.norm(reconstructed_vector) + 1e-12
    z_hat = (reconstructed_vector / recon_norm).astype(np.float64)

    # Target unit vector
    target_norm = np.linalg.norm(target_embedding) + 1e-12
    z_true = (target_embedding / target_norm).astype(np.float64)

    # Cosine similarity
    cosine_sim = float(np.dot(z_true, z_hat))
    return cosine_sim

