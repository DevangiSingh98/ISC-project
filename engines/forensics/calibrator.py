from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE):
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(probs)
    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += prop_in_bin * np.abs(accuracy_in_bin - avg_confidence_in_bin)
            
    return float(ece)

class ProbabilityCalibrator:
    """
    Calibrates raw detector output probabilities using Isotonic Regression
    or Temperature Scaling. Tracks ECE separately for in-distribution data
    and held-out generator distributions (per Known Gap 3).
    """
    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.temperature: float = 1.0
        self.is_fitted: bool = False
        self.in_distribution_ece: Optional[float] = None
        self.held_out_ece: Optional[float] = None

    def fit(self, raw_probs: np.ndarray, true_labels: np.ndarray) -> None:
        """Fits calibration curve on validation set."""
        raw_probs = np.clip(raw_probs, 0.0, 1.0)
        self.calibrator.fit(raw_probs, true_labels)
        self.is_fitted = True
        calibrated = self.calibrator.predict(raw_probs)
        self.in_distribution_ece = compute_ece(calibrated, true_labels)

    def calibrate(self, raw_prob: float) -> float:
        """Returns calibrated posterior probability."""
        if not self.is_fitted:
            return float(np.clip(raw_prob, 0.0, 1.0))
        prob_arr = np.array([np.clip(raw_prob, 0.0, 1.0)])
        return float(self.calibrator.predict(prob_arr)[0])

    def evaluate_held_out(self, raw_probs: np.ndarray, true_labels: np.ndarray) -> float:
        """Evaluates calibration error on unseen / held-out generator families."""
        calibrated = np.array([self.calibrate(p) for p in raw_probs])
        self.held_out_ece = compute_ece(calibrated, true_labels)
        return self.held_out_ece

