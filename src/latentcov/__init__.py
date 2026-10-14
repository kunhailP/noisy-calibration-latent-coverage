"""Latent coverage from noisy calibration.

Modules
-------
interval       ball-arithmetic (Arb) enclosures of the one-sided constants (from the Biometrika code)
noise_classes  coverage transfer for classes of noise laws (from the Biometrika code)
levels         noisy levels by noise class and latent shape, with their support
rank           conformal ranks of Theorem 2 through one interface
"""
from latentcov.levels import level
from latentcov.rank import rank, rank_for_level
