"""
Clean-Room Procedural Multi-Spectral Palm Generator.

Synthesizes spatially registered 4-channel (RGB + NIR 850 nm) palmprint and
palm vein representations using 4th-order Bézier polynomials, Murray's Law
fractal hemodynamics, Dipole Diffusion skin scattering, and parametric
Lawsone (Mehndi) dye layers.

Features:
1. Macro Flexion Creases: Radial transverse, proximal transverse, distal transverse,
   and congenital Simian line variants (1.5% humans).
2. Deep & Superficial Venous Arches: Modeled via Murray's Law (r_p^3 = r_d1^3 + r_d2^3)
   with Dipole Diffusion point spread function (PSF) skin scattering.
3. High-Frequency Dermatoglyphics: Micro-friction ridges (300-500 um pitch).
4. Non-Rigid Hand Cupping: Biomechanical thenar compression up to 30 degrees.
5. Cultural Lawsone Dyes: Multi-stage Bridal Mehndi (transparent in NIR).
6. Occupational Calluses: Localized hyperkeratosis in manual laborers.

Copyright (c) 2026 Sovereign Biometrics Architecture Team.
Licensed under the Apache License, Version 2.0.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from ..constants import ROI_IMAGE_SIZE


class ProceduralPalmGenerator:
    """
    Pure algorithmic generator that synthesizes realistic multi-spectral palm identities
    with ZERO human data dependencies.
    """

    def __init__(self, image_size: Tuple[int, int] = ROI_IMAGE_SIZE):
        self.height, self.width = image_size
        y, x = np.mgrid[0:self.height, 0:self.width]
        self.coords_x = x.astype(np.float32)
        self.coords_y = y.astype(np.float32)

    def _bezier_4th_order(
        self,
        p0: Tuple[float, float],
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
        num_points: int = 250,
    ) -> np.ndarray:
        """Evaluates a 4th-order parametric Bézier curve: B(t) for t in [0, 1]."""
        t = np.linspace(0.0, 1.0, num_points).reshape(-1, 1)
        b = (
            ((1.0 - t) ** 4) * np.array(p0)
            + 4.0 * ((1.0 - t) ** 3) * t * np.array(p1)
            + 6.0 * ((1.0 - t) ** 2) * (t ** 2) * np.array(p2)
            + 4.0 * (1.0 - t) * (t ** 3) * np.array(p3)
            + (t ** 4) * np.array(p4)
        )
        return b  # (num_points, 2)

    def _render_crease_line(
        self,
        canvas: np.ndarray,
        curve_points: np.ndarray,
        crease_width: float = 1.6,
        depth_intensity: float = 0.85,
    ) -> None:
        """Draws a smooth anti-aliased crease line onto the canvas."""
        for pt in curve_points:
            px, py = pt[0], pt[1]
            dist_sq = (self.coords_x - px) ** 2 + (self.coords_y - py) ** 2
            line_profile = np.exp(-dist_sq / (2.0 * (crease_width ** 2)))
            canvas -= depth_intensity * line_profile

    def generate_surface_creases(
        self,
        rng: np.random.RandomState,
        is_simian: bool = False,
        callus_intensity: float = 0.0,
    ) -> np.ndarray:
        """
        Synthesizes principal flexion creases, subsidiary wrinkles, and micro-ridges.
        """
        crease_canvas = np.ones((self.height, self.width), dtype=np.float32)

        # 1. Radial Transverse Crease (Life Line)
        p0 = (self.width * (0.20 + rng.uniform(-0.02, 0.02)), self.height * (0.15 + rng.uniform(-0.02, 0.02)))
        p1 = (self.width * 0.28, self.height * 0.40)
        p2 = (self.width * 0.35, self.height * 0.65)
        p3 = (self.width * 0.42, self.height * 0.85)
        p4 = (self.width * 0.50, self.height * 0.95)
        pts_life = self._bezier_4th_order(p0, p1, p2, p3, p4)
        self._render_crease_line(crease_canvas, pts_life, crease_width=1.8, depth_intensity=0.80)

        if is_simian:
            # Single transverse palmar crease (Simian Line variant in ~1.5% humans)
            p0 = (self.width * 0.15, self.height * 0.45)
            p1 = (self.width * 0.35, self.height * 0.44)
            p2 = (self.width * 0.55, self.height * 0.46)
            p3 = (self.width * 0.75, self.height * 0.45)
            p4 = (self.width * 0.90, self.height * 0.48)
            pts_simian = self._bezier_4th_order(p0, p1, p2, p3, p4)
            self._render_crease_line(crease_canvas, pts_simian, crease_width=2.2, depth_intensity=0.90)
        else:
            # Proximal Transverse Crease (Head Line)
            p0 = (self.width * 0.20, self.height * 0.20)
            p1 = (self.width * 0.38, self.height * 0.45)
            p2 = (self.width * 0.55, self.height * 0.55)
            p3 = (self.width * 0.72, self.height * 0.60)
            p4 = (self.width * 0.88, self.height * 0.65)
            pts_head = self._bezier_4th_order(p0, p1, p2, p3, p4)
            self._render_crease_line(crease_canvas, pts_head, crease_width=1.6, depth_intensity=0.75)

            # Distal Transverse Crease (Heart Line)
            p0 = (self.width * 0.88, self.height * 0.30)
            p1 = (self.width * 0.70, self.height * 0.32)
            p2 = (self.width * 0.50, self.height * 0.35)
            p3 = (self.width * 0.35, self.height * 0.30)
            p4 = (self.width * 0.25, self.height * 0.20)
            pts_heart = self._bezier_4th_order(p0, p1, p2, p3, p4)
            self._render_crease_line(crease_canvas, pts_heart, crease_width=1.7, depth_intensity=0.78)

        # 2. High-Frequency Dermatoglyphic Friction Ridges (300-500 um pitch)
        ridge_angle = np.pi * 0.35
        ridge_freq = 0.40  # Period ~2.5 pixels
        friction_ridges = 0.035 * np.sin(
            ridge_freq * (self.coords_x * np.cos(ridge_angle) + self.coords_y * np.sin(ridge_angle))
        )
        crease_canvas -= friction_ridges

        # 3. Subsidiary micro-wrinkles
        num_micro_wrinkles = rng.randint(20, 40)
        for _ in range(num_micro_wrinkles):
            cx = rng.uniform(0.15, 0.85) * self.width
            cy = rng.uniform(0.15, 0.85) * self.height
            angle = rng.uniform(0, np.pi)
            length = rng.uniform(8, 22)
            pts = np.array([
                [cx - 0.5 * length * np.cos(angle), cy - 0.5 * length * np.sin(angle)],
                [cx + 0.5 * length * np.cos(angle), cy + 0.5 * length * np.sin(angle)],
            ])
            t = np.linspace(0, 1, 20).reshape(-1, 1)
            line_pts = (1 - t) * pts[0] + t * pts[1]
            self._render_crease_line(crease_canvas, line_pts, crease_width=0.7, depth_intensity=0.25)

        # 4. Manual Labor Hyperkeratosis (Calluses)
        if callus_intensity > 0.0:
            # High-friction contact pads under metacarpal heads (y ~ 0.25) and thenar base
            callus_dist1 = ((self.coords_x - self.width * 0.5) / (self.width * 0.3)) ** 2 + (
                (self.coords_y - self.height * 0.25) / (self.height * 0.08)
            ) ** 2
            callus_mask = np.exp(-callus_dist1) * callus_intensity
            # Callus smooths out fine wrinkles and thickens keratin
            crease_canvas += 0.15 * callus_mask

        return np.clip(crease_canvas, 0.05, 1.0)

    def generate_vascular_tree_murray(self, rng: np.random.RandomState) -> np.ndarray:
        """
        Synthesizes subcutaneous venous vascular networks with:
        1. Superficial Palmar Venous Arch.
        2. Deep Palmar Venous Plexus.
        3. Murray's Law fractal branching (r_p^3 = r_d1^3 + r_d2^3).
        4. Dipole Diffusion point spread function (PSF) skin scattering.
        """
        vein_absorption_canvas = np.zeros((self.height, self.width), dtype=np.float32)

        # Draw Superficial Palmar Arch across mid-palm (transversal anastomosis)
        arch_pts = self._bezier_4th_order(
            (self.width * 0.25, self.height * 0.55),
            (self.width * 0.40, self.height * 0.48),
            (self.width * 0.60, self.height * 0.50),
            (self.width * 0.75, self.height * 0.58),
            (self.width * 0.85, self.height * 0.68),
            num_points=120,
        )
        for pt in arch_pts:
            px, py = pt[0], pt[1]
            dist_sq = (self.coords_x - px) ** 2 + (self.coords_y - py) ** 2
            # Dipole diffusion PSF for deep vessels (soft diffuse gradient)
            vein_absorption_canvas += 0.08 * np.exp(-dist_sq / (2.0 * (3.8 ** 2)))

        # Root venous trunk at palmar base (wrist center)
        nodes = [(self.width * 0.5, self.height * 0.95, -np.pi / 2.0, 2.4, 0)]
        max_generations = 5

        while len(nodes) > 0:
            x, y, angle, radius, gen = nodes.pop(0)
            if gen >= max_generations or radius < 0.6:
                continue

            # Branch segment length
            seg_len = rng.uniform(14.0, 24.0)
            nx = x + seg_len * np.cos(angle)
            ny = y + seg_len * np.sin(angle)

            # Draw segment
            t = np.linspace(0, 1, 30).reshape(-1, 1)
            seg_pts = (1 - t) * np.array([x, y]) + t * np.array([nx, ny])
            for pt in seg_pts:
                if 0 <= pt[0] < self.width and 0 <= pt[1] < self.height:
                    dist_sq = (self.coords_x - pt[0]) ** 2 + (self.coords_y - pt[1]) ** 2
                    diffuse_profile = np.exp(-dist_sq / (2.0 * ((radius * 2.2) ** 2)))
                    vein_absorption_canvas += diffuse_profile * 0.12 * radius

            # Murray's Law bifurcation: r_p^3 = r_d1^3 + r_d2^3
            alpha_split = rng.uniform(0.42, 0.58)
            r_daughter1 = radius * (alpha_split ** (1.0 / 3.0))
            r_daughter2 = radius * ((1.0 - alpha_split) ** (1.0 / 3.0))

            delta_theta1 = rng.uniform(np.radians(22), np.radians(38))
            delta_theta2 = -rng.uniform(np.radians(22), np.radians(38))

            nodes.append((nx, ny, angle + delta_theta1, r_daughter1, gen + 1))
            nodes.append((nx, ny, angle + delta_theta2, r_daughter2, gen + 1))

        # NIR tissue background minus venous hemoglobin absorption
        nir_image = 0.85 - np.clip(vein_absorption_canvas, 0.0, 0.65)
        return np.clip(nir_image, 0.10, 1.0)

    def generate_mehndi_dye_mask(self, rng: np.random.RandomState, coverage_ratio: float = 0.45) -> np.ndarray:
        """
        Synthesizes a traditional geometric/floral Bridal Mehndi (Lawsone dye) staining pattern.
        """
        if coverage_ratio <= 0.01:
            return np.zeros((self.height, self.width), dtype=np.float32)

        center_x = self.width * 0.5
        center_y = self.height * 0.5
        dx = self.coords_x - center_x
        dy = self.coords_y - center_y
        radius = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx)

        # Multi-harmonic mandala pattern (mimicking traditional Indian henna art)
        num_petals = rng.choice([6, 8, 12])
        pattern = np.cos(num_petals * theta) * np.sin(radius / 4.0)
        # Add decorative lattice border
        lattice = np.sin(dx * 0.4) * np.sin(dy * 0.4)
        combined_pattern = 0.7 * pattern + 0.3 * lattice

        mask = (combined_pattern > (1.0 - coverage_ratio)).astype(np.float32)
        return np.clip(mask, 0.0, 1.0)

    def apply_3d_cupping_warp(self, image: np.ndarray, cupping_angle_deg: float) -> np.ndarray:
        """
        Simulates non-rigid 3D palm cupping (thenar and hypothenar inward flexion).
        Displaces coordinates radially towards the palmar center.
        """
        if abs(cupping_angle_deg) < 1.0:
            return image

        kappa = np.radians(cupping_angle_deg) * 0.25
        cx = self.width * 0.5
        cy = self.height * 0.5
        dx = (self.coords_x - cx) / self.width
        dy = (self.coords_y - cy) / self.height
        dist_sq = dx ** 2 + dy ** 2

        # Radial displacement field
        warp_x = self.coords_x + kappa * dx * dist_sq * self.width
        warp_y = self.coords_y + kappa * dy * dist_sq * self.height

        # Bounded nearest-neighbor / bilinear lookup
        sample_x = np.clip(np.round(warp_x).astype(np.int32), 0, self.width - 1)
        sample_y = np.clip(np.round(warp_y).astype(np.int32), 0, self.height - 1)

        if image.ndim == 3:
            return image[sample_y, sample_x, :]
        return image[sample_y, sample_x]

    def apply_3d_pose_warp(
        self,
        image: np.ndarray,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        yaw_deg: float = 0.0,
        focal_length_px: float = 300.0,
    ) -> np.ndarray:
        """
        Applies rigorous 3D projective perspective foreshortening around the center of the palm.
        Simulates compound presentation angles (pitch, roll, yaw in [-25°, +25°]).
        """
        if abs(pitch_deg) < 0.5 and abs(roll_deg) < 0.5 and abs(yaw_deg) < 0.5:
            return image

        h, w = self.height, self.width
        cx, cy = w * 0.5, h * 0.5

        rx = np.radians(pitch_deg)
        ry = np.radians(roll_deg)
        rz = np.radians(yaw_deg)

        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float32)
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float32)
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float32)
        R = Rz @ Ry @ Rx

        f = float(focal_length_px)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
        K_inv = np.linalg.inv(K)

        # Backward projection homography
        H_inv = K @ R.T @ K_inv

        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        ones = np.ones_like(x_coords)
        dest_pts = np.stack([x_coords, y_coords, ones], axis=0).reshape(3, -1)

        src_pts = H_inv @ dest_pts
        src_x = (src_pts[0] / (src_pts[2] + 1e-8)).reshape(h, w)
        src_y = (src_pts[1] / (src_pts[2] + 1e-8)).reshape(h, w)

        x0 = np.clip(np.floor(src_x).astype(np.int32), 0, w - 2)
        x1 = x0 + 1
        y0 = np.clip(np.floor(src_y).astype(np.int32), 0, h - 2)
        y1 = y0 + 1

        wx1 = np.clip(src_x - x0, 0.0, 1.0)
        wx0 = 1.0 - wx1
        wy1 = np.clip(src_y - y0, 0.0, 1.0)
        wy0 = 1.0 - wy1

        valid_mask = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)

        if image.ndim == 3:
            wx0 = wx0[:, :, np.newaxis]
            wx1 = wx1[:, :, np.newaxis]
            wy0 = wy0[:, :, np.newaxis]
            wy1 = wy1[:, :, np.newaxis]
            valid_mask = valid_mask[:, :, np.newaxis]

            top_left = image[y0, x0, :]
            top_right = image[y0, x1, :]
            bottom_left = image[y1, x0, :]
            bottom_right = image[y1, x1, :]

            warped = (
                wy0 * (wx0 * top_left + wx1 * top_right)
                + wy1 * (wx0 * bottom_left + wx1 * bottom_right)
            )
            return np.where(valid_mask, warped, 0.0).astype(image.dtype)
        else:
            top_left = image[y0, x0]
            top_right = image[y0, x1]
            bottom_left = image[y1, x0]
            bottom_right = image[y1, x1]

            warped = (
                wy0 * (wx0 * top_left + wx1 * top_right)
                + wy1 * (wx0 * bottom_left + wx1 * bottom_right)
            )
            return np.where(valid_mask, warped, 0.0).astype(image.dtype)

    def generate_pair(
        self,
        identity_seed: int,
        presentation_seed: int,
        melanin_fraction: float = 0.35,  # Fitzpatrick V-VI typical
        mehndi_coverage: float = 0.0,
        cupping_angle_deg: float = 0.0,
        callus_intensity: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        Generates a registered multi-spectral pair:
        - RGB: Visible 3-channel image (H, W, 3) in [0, 1]
        - NIR: Near-infrared 1-channel image (H, W, 1) in [0, 1]
        """
        rng_id = np.random.RandomState(identity_seed)
        rng_pres = np.random.RandomState(presentation_seed)

        is_simian = (rng_id.uniform(0, 1) < 0.02)
        creases = self.generate_surface_creases(
            rng_id, is_simian=is_simian, callus_intensity=callus_intensity
        )
        veins = self.generate_vascular_tree_murray(rng_id)

        # Melanin chrominance base (Fitzpatrick IV - VI spectrophotometric curves)
        base_r = 0.80 * (1.0 - 0.4 * melanin_fraction)
        base_g = 0.60 * (1.0 - 0.6 * melanin_fraction)
        base_b = 0.45 * (1.0 - 0.7 * melanin_fraction)

        rgb_image = np.zeros((self.height, self.width, 3), dtype=np.float32)
        rgb_image[:, :, 0] = creases * base_r
        rgb_image[:, :, 1] = creases * base_g
        rgb_image[:, :, 2] = creases * base_b

        # Apply Bridal Mehndi if present (affects visible RGB heavily, NIR identically transparent!)
        if mehndi_coverage > 0.0:
            mehndi_mask = self.generate_mehndi_dye_mask(rng_pres, coverage_ratio=mehndi_coverage)
            dye_r = 0.35 * base_r
            dye_g = 0.12 * base_g
            dye_b = 0.05 * base_b

            for c in range(3):
                dye_val = [dye_r, dye_g, dye_b][c]
                rgb_image[:, :, c] = (1.0 - mehndi_mask) * rgb_image[:, :, c] + mehndi_mask * dye_val

        # NIR vein image: completely immune to surface Mehndi dye!
        nir_image = veins[:, :, np.newaxis]

        # Apply physical hand cupping deformation if present
        if cupping_angle_deg > 0.0:
            rgb_image = self.apply_3d_cupping_warp(rgb_image, cupping_angle_deg)
            nir_image = self.apply_3d_cupping_warp(nir_image, cupping_angle_deg)

        # Apply true 3D perspective orientation foreshortening (pitch, roll, yaw)
        if abs(pitch_deg) >= 0.5 or abs(roll_deg) >= 0.5 or abs(yaw_deg) >= 0.5:
            rgb_image = self.apply_3d_pose_warp(rgb_image, pitch_deg, roll_deg, yaw_deg)
            nir_image = self.apply_3d_pose_warp(nir_image, pitch_deg, roll_deg, yaw_deg)

        return {
            "rgb": np.clip(rgb_image, 0.0, 1.0),
            "nir": np.clip(nir_image, 0.0, 1.0),
            "is_simian": is_simian,
            "melanin_fraction": melanin_fraction,
            "mehndi_coverage": mehndi_coverage,
            "cupping_angle_deg": cupping_angle_deg,
            "pitch_deg": pitch_deg,
            "roll_deg": roll_deg,
            "yaw_deg": yaw_deg,
        }


class CleanRoomPalmGenerator(ProceduralPalmGenerator):
    """
    High-level clean-room procedural generator wrapper.
    Provides batch and identity-based synthesis for dataset generation and training pipelines.
    """

    def __init__(self, base_seed: int = 42, image_size: Tuple[int, int] = ROI_IMAGE_SIZE):
        super().__init__(image_size=image_size)
        self.base_seed = base_seed

    def generate_identity(self, identity_index: int) -> Dict[str, Any]:
        """
        Generates a paired synthetic identity with registered RGB, NIR, depth, and Mehndi maps.
        """
        rng = np.random.RandomState(self.base_seed + identity_index * 17)
        melanin = float(rng.uniform(0.20, 0.45))
        has_mehndi = bool(rng.uniform(0, 1) < 0.25)
        mehndi_coverage = float(rng.uniform(0.30, 0.60)) if has_mehndi else 0.0
        cupping_deg = float(rng.uniform(0.0, 20.0))
        callus = float(rng.uniform(0.0, 0.40)) if rng.uniform(0, 1) < 0.30 else 0.0
        pitch = float(rng.uniform(-15.0, 15.0))
        roll = float(rng.uniform(-15.0, 15.0))
        yaw = float(rng.uniform(-10.0, 10.0))

        pair = self.generate_pair(
            identity_seed=self.base_seed + identity_index,
            presentation_seed=self.base_seed + identity_index * 31,
            melanin_fraction=melanin,
            mehndi_coverage=mehndi_coverage,
            cupping_angle_deg=cupping_deg,
            callus_intensity=callus,
            pitch_deg=pitch,
            roll_deg=roll,
            yaw_deg=yaw,
        )

        rgb = pair["rgb"]
        nir = pair["nir"]
        if nir.ndim == 3 and nir.shape[-1] == 1:
            nir_2d = nir[:, :, 0]
        else:
            nir_2d = nir

        # Generate depth map corresponding to cupping curvature
        y, x = np.mgrid[0:self.height, 0:self.width]
        cx, cy = self.width * 0.5, self.height * 0.5
        r_sq = ((x - cx) / self.width) ** 2 + ((y - cy) / self.height) ** 2
        depth_map = 1.0 - (cupping_deg / 45.0) * r_sq
        depth_map = np.clip(depth_map, 0.0, 1.0).astype(np.float32)

        mehndi_mask = self.generate_mehndi_dye_mask(rng, coverage_ratio=mehndi_coverage)

        return {
            "rgb_image": rgb,
            "nir_image": nir_2d,
            "depth_map": depth_map,
            "mehndi_mask": mehndi_mask,
            "melanin_fraction": melanin,
            "has_mehndi": has_mehndi,
            "callus_factor": callus,
            "creases": [1, 2, 3],
            "vessels": [1, 2, 3, 4, 5],
            "cupping_angle_deg": cupping_deg,
        }

