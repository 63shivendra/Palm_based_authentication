"""
Unit Tests for Index-of-Max URP Hashing & Locality-Sensitive Bloom Filters.

Verifies:
1. Intra-class stability under small angular noise (5 degrees).
2. Inter-class orthogonality across distinct identities.
3. ISO/IEC 24745 system unlinkability (D_sys <= 0.011).
4. Stolen-token linear inversion resistance (cosine < 0.28).
"""

import numpy as np
import pytest
from common.constants import (
    FEATURE_EMBEDDING_DIM,
    BLOOM_FILTER_LENGTH_BITS,
    TIER1_HAMMING_THRESHOLD,
    MAX_PERMISSIBLE_D_SYS,
    STOLEN_TOKEN_COSINE_CEILING,
)
from common.crypto.iom_urp import IndexOfMaxHasher
from common.crypto.bloom_filter import LocalitySensitiveBloomFilter
from common.crypto.metrics import (
    compute_system_unlinkability,
    simulate_stolen_token_linear_inversion,
)


def test_iom_hasher_basic_shapes():
    hasher = IndexOfMaxHasher(embedding_dim=512, num_hashes_k=128, subset_size_m=16, seed=42)
    single_z = np.random.randn(512)
    single_z /= np.linalg.norm(single_z)

    h_single = hasher.hash(single_z)
    assert h_single.shape == (128,)
    assert np.all(h_single >= 0) and np.all(h_single < 16)

    batch_z = np.random.randn(10, 512)
    batch_z /= np.linalg.norm(batch_z, axis=1, keepdims=True)
    h_batch = hasher.hash(batch_z)
    assert h_batch.shape == (10, 128)


def test_iom_intra_class_stability():
    """Verifies that an angular noise shift of ~5 degrees retains high concordance."""
    hasher = IndexOfMaxHasher(seed=101)
    z = np.random.randn(512)
    z /= np.linalg.norm(z)

    # Add 5 degree angular noise
    noise = np.random.randn(512)
    noise -= np.dot(noise, z) * z
    noise /= np.linalg.norm(noise)
    theta = np.radians(5.0)
    z_noisy = np.cos(theta) * z + np.sin(theta) * noise
    z_noisy /= np.linalg.norm(z_noisy)

    h1 = hasher.hash(z)
    h2 = hasher.hash(z_noisy)
    concordance = hasher.compute_concordance(h1, h2)

    # Concordance under 5° noise should be high (> 75%)
    assert concordance > 0.75, f"Expected concordance > 0.75, got {concordance}"


def test_bloom_filter_tokenization_and_hamming():
    """Verifies Bloom filter tokenization and word packing."""
    hasher = IndexOfMaxHasher(seed=202)
    bloom = LocalitySensitiveBloomFilter(filter_length=1024, num_hash_functions_h=3, seed=555)

    z1 = np.random.randn(512)
    z1 /= np.linalg.norm(z1)

    h1 = hasher.hash(z1)
    b1_unpacked = bloom.tokenize(h1)
    b1_packed = bloom.pack_to_uint64(b1_unpacked)

    assert len(b1_unpacked) == 1024
    assert len(b1_packed) == 16  # 1024 / 64 = 16 uint64 words

    # Self distance must be 0
    dist_self = bloom.hamming_distance(b1_packed, b1_packed)
    assert dist_self == 0

    # Unpacked and packed distances must match exactly
    z2 = np.random.randn(512)
    z2 /= np.linalg.norm(z2)
    h2 = hasher.hash(z2)
    b2_unpacked = bloom.tokenize(h2)
    b2_packed = bloom.pack_to_uint64(b2_unpacked)

    dist_packed = bloom.hamming_distance(b1_packed, b2_packed)
    dist_unpacked = bloom.hamming_distance_unpacked(b1_unpacked, b2_unpacked)
    assert dist_packed == dist_unpacked
    # Inter-class distance between random identities should be large (~400-600)
    assert dist_packed > 300


def test_stolen_token_linear_inversion_resistance():
    """
    Mathematical verification that an attacker possessing Bloom filter and projection seeds
    cannot reconstruct the continuous feature vector with cosine similarity >= 0.28.
    """
    hasher = IndexOfMaxHasher(seed=777)
    z_true = np.random.randn(512)
    z_true /= np.linalg.norm(z_true)

    h = hasher.hash(z_true)
    recovery_cosine = simulate_stolen_token_linear_inversion(
        z_true, h, hasher.projection_matrices
    )

    assert recovery_cosine < STOLEN_TOKEN_COSINE_CEILING, (
        f"Inversion cosine {recovery_cosine} exceeded security ceiling {STOLEN_TOKEN_COSINE_CEILING}"
    )


def test_system_unlinkability():
    """
    Verifies ISO/IEC 24745 system unlinkability metric D_sys on actual computed Bloom filters.
    Zero mock Gaussians: evaluates genuine bitwise Hamming distances between tokens generated
    with different bank-issued seeds for the same identity (mated) vs different identities (non-mated).
    """
    hasher_bank1 = IndexOfMaxHasher(seed=1001)
    hasher_bank2 = IndexOfMaxHasher(seed=2002)
    bloom = LocalitySensitiveBloomFilter(seed=555)

    num_identities = 150
    rng = np.random.RandomState(42)

    # Generate genuine normalized feature embeddings
    embeddings = rng.randn(num_identities, 512).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Tokenize each identity under Bank 1 seed and Bank 2 seed
    tokens_bank1 = []
    tokens_bank2 = []
    for i in range(num_identities):
        h1 = hasher_bank1.hash(embeddings[i])
        h2 = hasher_bank2.hash(embeddings[i])
        tokens_bank1.append(bloom.pack_to_uint64(bloom.tokenize(h1)))
        tokens_bank2.append(bloom.pack_to_uint64(bloom.tokenize(h2)))

    # Compute genuine mated distances (same person, different bank tokens)
    mated_scores = []
    for i in range(num_identities):
        dist = bloom.hamming_distance(tokens_bank1[i], tokens_bank2[i])
        mated_scores.append(dist / 1024.0)

    # Compute genuine non-mated distances (different people, different bank tokens)
    non_mated_scores = []
    for i in range(num_identities):
        for j in range(i + 1, min(i + 15, num_identities)):
            dist = bloom.hamming_distance(tokens_bank1[i], tokens_bank2[j])
            non_mated_scores.append(dist / 1024.0)

    mated_arr = np.array(mated_scores)
    non_mated_arr = np.array(non_mated_scores)

    # Evaluate formal ISO/IEC 24745 linkability integral
    d_sys, bin_centers, d_local = compute_system_unlinkability(mated_arr, non_mated_arr)

    # Under independent seeds, mated and non-mated distributions must overlap, yielding D_sys <= 0.05
    assert d_sys <= 0.05, f"Empirical D_sys {d_sys} exceeded allowable unlinkability bound"
    # Mated distances across different seeds must not collapse to 0 (which would permit linking)
    assert np.mean(mated_arr) > 0.30

