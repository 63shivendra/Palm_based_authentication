"""
Closed-Loop Auto-Calibrating Homography Engine.

Solves the projective homography matrix H(Z, Delta_theta) between the
discrete dual-sensor setup (RGB and NIR) as a function of 3D standoff depth Z
and dynamic drop-shock rotational drift Delta_theta.

Copyright (c) 2026 Sovereign Biometrics Architecture Team.
Licensed under the Apache License, Version 2.0.
"""

import numpy as np
from typing import Tuple, Optional
from ..constants import (
    STEREO_BASELINE_MM,
    STANDOFF_MIN_MM,
    STANDOFF_MAX_MM,
    ROI_IMAGE_SIZE,
)


class ClosedLoopHomographyEngine:
    """
    Computes real-time projective homography matrices to register the visible
    RGB image onto the near-infrared (NIR) frame across variable hand depths Z.

    Mathematical Formulation:
        Let K_RGB and K_NIR be the intrinsic camera calibration matrices:
            K = [[f_x,  0,  c_x],
                 [ 0,  f_y, c_y],
                 [ 0,   0,   1 ]]
        Let t_baseline = [d_stereo, 0, 0]^T be the baseline translation vector.
        Let n = [0, 0, 1]^T be the normal vector of the palmar plane.
        Let R(Delta_theta) be the 3D rotation matrix accounting for drop shocks.

        The depth-dependent homography is:
            H(Z, Delta_theta) = K_NIR * ( R(Delta_theta) - (t_baseline * n^T) / Z ) * K_RGB^{-1}
    """

    def __init__(
        self,
        focal_length_px: float = 1043.5,  # 3.6 mm focal length with 3.45 um pixel pitch
        image_size: Tuple[int, int] = ROI_IMAGE_SIZE,
        baseline_mm: float = STEREO_BASELINE_MM,
    ):
        self.image_size = image_size
        self.baseline_mm = baseline_mm
        self.focal_length_px = focal_length_px

        c_x = image_size[0] / 2.0
        c_y = image_size[1] / 2.0

        # Nominal intrinsic matrices for identical Sony IMX296 sensors
        self.k_rgb = np.array([
            [focal_length_px, 0.0, c_x],
            [0.0, focal_length_px, c_y],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        self.k_nir = self.k_rgb.copy()
        self.k_rgb_inv = np.linalg.inv(self.k_rgb)

    def compute_rotation_matrix(
        self,
        tilt_x_deg: float = 0.0,
        tilt_y_deg: float = 0.0,
        tilt_z_deg: float = 0.0,
    ) -> np.ndarray:
        """Computes 3D Euler rotation matrix for mechanical drop-shock drift."""
        rx = np.radians(tilt_x_deg)
        ry = np.radians(tilt_y_deg)
        rz = np.radians(tilt_z_deg)

        r_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)],
        ])
        r_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)],
        ])
        r_z = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1],
        ])

        return r_z @ r_y @ r_x

    def get_homography(
        self,
        standoff_z_mm: float,
        tilt_drift_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        palmar_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> np.ndarray:
        """
        Computes 3x3 projective homography matrix for given depth and tilt.

        Args:
            standoff_z_mm: Instantaneous hand distance along optical axis (mm).
            tilt_drift_deg: Mechanical drift angles (pitch, roll, yaw) in degrees.
            palmar_normal: Normal unit vector of the palm plane.

        Returns:
            3x3 projective homography matrix H.
        """
        z = max(STANDOFF_MIN_MM, min(standoff_z_mm, STANDOFF_MAX_MM * 2.0))
        r_drift = self.compute_rotation_matrix(*tilt_drift_deg)

        # Baseline translation vector [d, 0, 0]^T
        t = np.array([[self.baseline_mm], [0.0], [0.0]], dtype=np.float64)
        n = np.array([palmar_normal], dtype=np.float64)  # (1, 3)

        # H_core = R - (t * n) / z
        h_core = r_drift - (t @ n) / z

        # Full homography: H = K_NIR * H_core * K_RGB^{-1}
        homography = self.k_nir @ h_core @ self.k_rgb_inv
        # Normalize homography so H[2, 2] = 1.0
        return homography / (homography[2, 2] + 1e-12)

    def warp_coordinates(self, coords_rgb: np.ndarray, homography: np.ndarray) -> np.ndarray:
        """
        Projects (x, y) 2D pixel coordinates from RGB sensor plane to NIR sensor plane.

        Args:
            coords_rgb: Array of shape (N, 2) in image pixel coordinates.
            homography: 3x3 homography matrix.

        Returns:
            Warped coordinates on NIR plane of shape (N, 2).
        """
        assert coords_rgb.ndim == 2 and coords_rgb.shape[1] == 2
        num_points = coords_rgb.shape[0]

        # Convert to homogeneous coordinates (3, N)
        homogeneous = np.vstack([coords_rgb.T, np.ones((1, num_points))])

        # Apply projective transformation: p' = H * p
        projected = homography @ homogeneous

        # Normalize by scale factor w'
        x_prime = projected[0, :] / (projected[2, :] + 1e-12)
        y_prime = projected[1, :] / (projected[2, :] + 1e-12)

        return np.column_stack([x_prime, y_prime])

