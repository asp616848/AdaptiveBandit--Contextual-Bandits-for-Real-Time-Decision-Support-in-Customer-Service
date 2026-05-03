"""Minimal local matplotlib shim for Stable-Baselines3 logger imports.

This project only needs the Figure type to satisfy stable_baselines3.common.logger.
Plotting is intentionally disabled in phase13_lumo_evaluation.py when real
matplotlib cannot be imported.
"""

from .figure import Figure

__all__ = ["Figure"]
