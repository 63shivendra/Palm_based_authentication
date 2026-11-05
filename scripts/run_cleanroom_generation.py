"""
scripts/run_cleanroom_generation.py

Operational utility to synthesize clean-room procedural multi-spectral seed datasets.
100% Pure Synthetic Sim-to-Real Generation:
  - Zero Human Data.
  - Zero DPDP Act 2023 Liability.
  - Generates 4th-order Bézier creases, Murray's law fractal hemodynamics,
    and multi-layer spectrophotometric melanin / Lawsone Mehndi dyes.

Usage:
  python scripts/run_cleanroom_generation.py --count 100 --output_dir ./data/seed_cohort/
"""

import os
import sys
import argparse
import json

# Ensure repository root is on Python sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from common.generators.procedural_palm import CleanRoomPalmGenerator


def generate_cohort(count: int, output_dir: str, seed: int = 42):
    """Generates a batch of synthetic palm identities with multi-spectral image pairs."""
    os.makedirs(output_dir, exist_ok=True)
    rgb_dir = os.path.join(output_dir, "rgb")
    nir_dir = os.path.join(output_dir, "nir")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(nir_dir, exist_ok=True)

    generator = CleanRoomPalmGenerator(base_seed=seed)
    metadata_records = []

    print(f"[*] Starting clean-room generation of {count} synthetic sovereign identities...")
    for i in range(count):
        # Generate paired synthetic palm identity
        palm_data = generator.generate_identity(identity_index=i)

        rgb_img = (palm_data["rgb_image"] * 255.0).astype(np.uint8)
        nir_img = (palm_data["nir_image"] * 255.0).astype(np.uint8)

        rgb_path = os.path.join(rgb_dir, f"palm_{i:06d}_rgb.png")
        nir_path = os.path.join(nir_dir, f"palm_{i:06d}_nir.png")

        Image.fromarray(rgb_img).save(rgb_path)
        Image.fromarray(nir_img).save(nir_path)

        record = {
            "identity_id": f"SYNTH-ID-{i:06d}",
            "rgb_path": rgb_path,
            "nir_path": nir_path,
            "melanin_fraction": palm_data["melanin_fraction"],
            "mehndi_presence": palm_data["has_mehndi"],
            "callus_factor": palm_data["callus_factor"],
            "crease_count": len(palm_data["creases"]),
            "vessel_branch_count": len(palm_data["vessels"]),
        }
        metadata_records.append(record)

        if (i + 1) % 20 == 0 or (i + 1) == count:
            print(f"    -> Generated {i + 1}/{count} identities...")

    meta_file = os.path.join(output_dir, "cohort_manifest.json")
    with open(meta_file, "w") as f:
        json.dump(metadata_records, f, indent=2)

    print(f"[+] Complete. Saved {count} identities to {output_dir}")
    print(f"    Manifest written to: {meta_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndiPalm Clean-Room Procedural Dataset Synthesizer")
    parser.add_argument("--count", type=int, default=50, help="Number of identities to generate")
    parser.add_argument("--output_dir", type=str, default="data/seed_cohort", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Master PRNG seed")
    args = parser.parse_args()

    generate_cohort(count=args.count, output_dir=args.output_dir, seed=args.seed)

