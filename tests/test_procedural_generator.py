"""
Unit Tests for Procedural Clean-Room Palm Generator.

Verifies:
1. Procedural Bézier creases and Murray's Law vascular trees.
2. 4-Channel multi-spectral registered image generation (RGB + NIR).
3. Lawsone Mehndi dye invariance in Near-Infrared channel.
"""

import numpy as np
import pytest
from common.generators.procedural_palm import ProceduralPalmGenerator


def test_procedural_generation_shapes_and_ranges():
    generator = ProceduralPalmGenerator(image_size=(128, 128))
    sample = generator.generate_pair(identity_seed=42, presentation_seed=101)

    rgb = sample["rgb"]
    nir = sample["nir"]

    assert rgb.shape == (128, 128, 3)
    assert nir.shape == (128, 128, 1)

    assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)
    assert np.all(nir >= 0.0) and np.all(nir <= 1.0)


def test_murray_vascular_tree_structure():
    generator = ProceduralPalmGenerator(image_size=(128, 128))
    rng = np.random.RandomState(99)
    veins = generator.generate_vascular_tree_murray(rng)

    assert veins.shape == (128, 128)
    # Veins absorb NIR, so mean must be lower than maximum tissue reflection
    assert np.min(veins) < np.max(veins)
    assert np.mean(veins) > 0.50  # Mostly reflective tissue with dark branching vessels


def test_mehndi_dye_nir_transparency():
    """
    Verifies the fundamental physical principle:
    Applying Bridal Mehndi (Henna) corrupts visible RGB image, but leaves
    the NIR vein image 100% IDENTICAL!
    """
    generator = ProceduralPalmGenerator(image_size=(128, 128))

    clean_sample = generator.generate_pair(
        identity_seed=500, presentation_seed=1, mehndi_coverage=0.0
    )
    mehndi_sample = generator.generate_pair(
        identity_seed=500, presentation_seed=1, mehndi_coverage=0.60
    )

    # RGB channels MUST differ due to dark reddish-brown dye
    assert not np.allclose(clean_sample["rgb"], mehndi_sample["rgb"])
    rgb_diff = np.mean(np.abs(clean_sample["rgb"] - mehndi_sample["rgb"]))
    assert rgb_diff > 0.05

    # NIR channels MUST be 100% identical (Lawsone dye is transparent at 850 nm)
    assert np.allclose(clean_sample["nir"], mehndi_sample["nir"], atol=1e-6)



def test_3d_cupping_deformation():
    """Verifies non-rigid 3D hand cupping deformation."""
    generator = ProceduralPalmGenerator(image_size=(128, 128))
    flat_sample = generator.generate_pair(identity_seed=123, presentation_seed=1, cupping_angle_deg=0.0)
    cupped_sample = generator.generate_pair(identity_seed=123, presentation_seed=1, cupping_angle_deg=25.0)

    # Cupping compresses thenar and hypothenar coordinates, inducing non-rigid pixel shift
    assert not np.allclose(flat_sample["rgb"], cupped_sample["rgb"])
    assert flat_sample["cupping_angle_deg"] == 0.0
    assert cupped_sample["cupping_angle_deg"] == 25.0


def test_occupational_calluses():
    """Verifies manual labor hyperkeratosis generation."""
    generator = ProceduralPalmGenerator(image_size=(128, 128))
    clean_creases = generator.generate_surface_creases(np.random.RandomState(7), callus_intensity=0.0)
    callus_creases = generator.generate_surface_creases(np.random.RandomState(7), callus_intensity=0.8)

    # Callus pads elevate scattering and smooth fine wrinkles
    assert not np.allclose(clean_creases, callus_creases)
    assert np.mean(callus_creases) > np.mean(clean_creases)
