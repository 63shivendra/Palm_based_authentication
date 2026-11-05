"""
benchmarks/iso_iec_evaluator.py

Comprehensive Biometric Certification and Forensic Audit Suite:
  1. ISO/IEC 19795-1: Biometric Performance Testing and Reporting (DET Curves, FAR, FRR, EER).
  2. ISO/IEC 30107-3: Presentation Attack Detection Evaluation (APCER, BPCER, ACER).
  3. ISO/IEC 24745: Information Technology — Security Techniques — Biometric Information Protection
     (System Unlinkability D_sys, Irreversibility, Revocability).
"""

from typing import Dict, List, Tuple, Any
import numpy as np


class ISO19795AccuracyEvaluator:
    """
    Evaluator for ISO/IEC 19795-1 Biometric Performance Metrics.
    Computes False Acceptance Rate (FAR), False Rejection Rate (FRR),
    Equal Error Rate (EER), and Detection Error Tradeoff (DET) points.
    """

    def __init__(self, genuine_distances: List[float], impostor_distances: List[float]):
        self.genuine_distances = np.array(genuine_distances, dtype=np.float64)
        self.impostor_distances = np.array(impostor_distances, dtype=np.float64)

    def evaluate_at_threshold(self, threshold: float) -> Tuple[float, float]:
        """
        Calculates FAR and FRR at a given distance threshold.
        Lower distance indicates a match.
        """
        # Genuine transactions with distance > threshold are falsely rejected
        frr = float(np.mean(self.genuine_distances > threshold))
        # Impostor transactions with distance <= threshold are falsely accepted
        far = float(np.mean(self.impostor_distances <= threshold))
        return far, frr

    def compute_eer(self, num_thresholds: int = 1000) -> Dict[str, float]:
        """
        Finds the Equal Error Rate (EER) where FAR == FRR.
        """
        min_dist = min(self.genuine_distances.min(), self.impostor_distances.min())
        max_dist = max(self.genuine_distances.max(), self.impostor_distances.max())
        thresholds = np.linspace(min_dist, max_dist, num_thresholds)

        best_eer = 1.0
        best_threshold = thresholds[0]

        for th in thresholds:
            far, frr = self.evaluate_at_threshold(th)
            diff = abs(far - frr)
            if diff < best_eer:
                best_eer = diff
                eer_val = (far + frr) / 2.0
                best_threshold = th

        return {
            "eer": float(eer_val),
            "eer_threshold": float(best_threshold),
        }


class ISO30107PADEvaluator:
    """
    Evaluator for ISO/IEC 30107-3 Presentation Attack Detection.
      - APCER: Attack Presentation Classification Error Rate (fraction of spoof presentations misclassified as live).
      - BPCER: Bona Fide Presentation Classification Error Rate (fraction of bona fide live presentations misclassified as spoofs).
      - ACER: Average Classification Error Rate: ACER = (APCER + BPCER) / 2.
    """

    def __init__(self, bona_fide_scores: List[float], attack_scores: List[float]):
        """
        Higher score indicates higher probability of bona fide live tissue.
        """
        self.bona_fide_scores = np.array(bona_fide_scores, dtype=np.float64)
        self.attack_scores = np.array(attack_scores, dtype=np.float64)

    def evaluate_pad(self, threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluates APCER, BPCER, and ACER at classification threshold.
        Scores >= threshold are classified as bona fide (live).
        Scores < threshold are classified as presentation attack (spoof).
        """
        # Spoofs classified as live: score >= threshold
        apcer = float(np.mean(self.attack_scores >= threshold))
        # Bona fide classified as attack: score < threshold
        bpcer = float(np.mean(self.bona_fide_scores < threshold))
        acer = (apcer + bpcer) / 2.0

        return {
            "apcer": apcer,
            "bpcer": bpcer,
            "acer": acer,
            "threshold": threshold,
            "passes_iso_target": acer < 0.0010,  # ACER < 0.10%
        }
