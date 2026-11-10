"""
engines/forensics/__init__.py — Track B AI Forensics Ensemble package
"""
from .manipulation.localizer import ManipulationLocalizer
from .synthetic.detector import AIGeneratedImageDetector, SpectralFFTBranch
from .noise_compression.compression_forensics import CompressionForensicsAnalyzer
from .calibrator import ProbabilityCalibrator, compute_ece

__all__ = [
    "ManipulationLocalizer",
    "AIGeneratedImageDetector",
    "SpectralFFTBranch",
    "CompressionForensicsAnalyzer",
    "ProbabilityCalibrator",
    "compute_ece",
]

