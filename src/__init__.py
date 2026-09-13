"""
quant-brainteasers-sim
=======================

A collection of quant-interview brainteaser simulations:

- ``betting_game``: The coin betting dilemma (EV maximization vs. Kelly
  Criterion vs. fixed-fraction betting, ergodicity, and risk of ruin).
- ``dice_sequence``: The [3, 4, 5] dice sequence problem (absorbing Markov
  chains, exact linear-algebra solutions, and a generalized KMP-based
  solver for arbitrary target sequences).
"""

from . import betting_game, dice_sequence

__all__ = ["betting_game", "dice_sequence"]

__version__ = "1.0.0"
