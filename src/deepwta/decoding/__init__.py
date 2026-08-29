"""Continuous-to-integer decoding for DeepWTA-Standard."""

from deepwta.mechanisms.integer_decoder import floor_integerize
from deepwta.solver import solve_deepwta

__all__ = [
    "floor_integerize",
    "solve_deepwta",
]
