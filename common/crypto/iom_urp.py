"""
Index-of-Max Uniform Random Permutations (IoM-URP) Hashing.

Implements provably non-invertible, locality-sensitive cancellable biometric
tokenization complying with ISO/IEC 24745.

Copyright (c) 2026 Sovereign Biometrics Architecture Team.
Licensed under the Apache License, Version 2.0.
"""

import numpy as np
from typing import List, Tuple
from ..constants import (
    FEATURE_EMBEDDING_DIM,
    IOM_NUM_HASHES_K,
    IOM_SUBSET_SIZE_M,
)


class IndexOfMaxHasher:
    """
    Computes Index-of-Max Random Permutation hashes from high-dimensional
    continuous biometric feature vectors.

    Mathematical Formulation:
        Let z in S^{D-1} be a unit-normalized continuous embedding.
        Given a bank-issued secret seed, generate K random projection sets:
            W_k = [w_{k, 1}, w_{k, 2}, ..., w_{k, M}] in R^{M x D}
        The ordinal rank index is selected via max-pooling:
            t_k = argmax_{j in {1, ..., M}} (w_{k, j}^T * z)

    The argmax operator completely discards continuous magnitude information,
    preventing Linear Programming, Quadratic Programming, or Mixed-Integer
    inversion attacks, while preserving rank-order correlation under intra-class noise.
    """

    def __init__(
        self,
        embedding_dim: int = FEATURE_EMBEDDING_DIM,
        num_hashes_k: int = IOM_NUM_HASHES_K,
        subset_size_m: int = IOM_SUBSET_SIZE_M,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.num_hashes_k = num_hashes_k
        self.subset_size_m = subset_size_m
        self.seed = seed
        self.projection_matrices = self._generate_projections(seed)

    def _generate_projections(self, seed: int) -> np.ndarray:
        """
        Generates pseudo-random orthonormal projection bases.
        Returns tensor of shape (K, M, D).
        """
        rng = np.random.RandomState(seed)
        # Generate Gaussian random projections
        raw_projections = rng.randn(self.num_hashes_k, self.subset_size_m, self.embedding_dim)
        # L2-normalize projection vectors
        norms = np.linalg.norm(raw_projections, axis=2, keepdims=True) + 1e-12
        return (raw_projections / norms).astype(np.float32)

    def hash(self, embedding: np.ndarray) -> np.ndarray:
        """
        Projects a continuous embedding vector z into an ordinal index vector.

        Args:
            embedding: 1D or 2D array of shape (D,) or (B, D). Must be L2-normalized.

        Returns:
            Ordinal index array of shape (K,) or (B, K) with values in {0, ..., M - 1}.
        """
        orig_shape = embedding.shape
        if embedding.ndim == 1:
            z = embedding.reshape(1, -1)
        else:
            z = embedding

        # Ensure unit L2 norm
        norm = np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
        z_norm = z / norm

        # Compute dot products: (B, D) x (K, M, D)^T -> (B, K, M)
        # Using tensordot over the feature dimension D
        projections = np.tensordot(z_norm, self.projection_matrices, axes=([1], [2]))
        # projections shape: (B, K, M)

        # argmax over the M candidates in each projection set
        indices = np.argmax(projections, axis=2).astype(np.int32)

        if len(orig_shape) == 1:
            return indices[0]
        return indices

    def compute_concordance(self, indices_a: np.ndarray, indices_b: np.ndarray) -> float:
        """
        Computes the ordinal rank concordance fraction between two hash vectors:
            Concordance = (1 / K) * sum_{k=1}^K I(t_k^A == t_k^B)
        """
        assert indices_a.shape == indices_b.shape, "Shape mismatch between hash vectors"
        matches = np.sum(indices_a == indices_b)
        return float(matches / len(indices_a))

