"""
betting_game.py
================

Module 1: The Coin Betting Dilemma.

Problem
-------
You start with $100. You flip a fair coin 100 times. On each flip you may
wager an amount ``B`` such that ``0 <= B <= current_wealth``. If the coin
comes up heads you win ``2 * B`` in profit (you keep your original stake
plus the winnings). If it comes up tails you lose the amount ``B``.

Because the payout is 2:1 on a 50/50 bet, the *expected value* of any
single bet of size ``B`` is:

    EV(bet) = 0.5 * (2B) + 0.5 * (-B) = 0.5 * B

Since EV(bet) is strictly increasing in ``B``, an investor who only cares
about maximizing the *expected value* of their terminal wealth should bet
100% of their wealth on every single flip (``f = 1.0``). This is the
"Expected Value Maximization" strategy.

However, betting 100% of your wealth on a coin flip means that a single
tails wipes you out completely (wealth -> $0), and once your wealth hits
zero it stays zero forever (absorbing state). Over 100 flips, the
probability of getting all heads is ``(1/2)^100``, an astronomically small
number. Therefore the probability of *ruin* (ending with $0) for the
all-in strategy is:

    P(ruin) = 1 - (1/2)^100  ≈  99.9999999999999999999999999999992%

This is the classic illustration of the difference between the
**ensemble average** (expected value across many parallel universes /
players) and the **time average** (what happens to a single player's
wealth path over time) -- the core idea behind *ergodicity economics* and
the *Kelly Criterion*.

This module provides:

1. An analytical engine that computes the exact expected value formula
   for the all-in strategy and proves f=1.0 maximizes single-flip EV.
2. A vectorized Monte Carlo simulation engine comparing three strategies:
      - Strategy A: All-in (bet 100% of wealth every flip)
      - Strategy B: Kelly-style fractional betting (bet f* of wealth)
      - Strategy C: Fixed fractional betting (bet a fixed 10% of wealth)
3. Ergodicity / risk-of-ruin analysis utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Analytical Engine
# --------------------------------------------------------------------------- #
class BettingAnalytics:
    """Closed-form / exact analytical results for the coin betting game.

    Parameters
    ----------
    initial_wealth:
        Starting wealth (defaults to $100 as specified in the problem).
    num_flips:
        Number of coin flips (defaults to 100).
    """

    def __init__(self, initial_wealth: float = 100.0, num_flips: int = 100) -> None:
        self.initial_wealth = initial_wealth
        self.num_flips = num_flips

    # -- Single flip EV -------------------------------------------------- #
    @staticmethod
    def single_flip_ev(bet_size: float) -> float:
        """Expected value of wagering ``bet_size`` on one fair coin flip.

        Win (p=0.5):  +2 * bet_size
        Loss (p=0.5): -1 * bet_size

        EV = 0.5*(2*bet_size) + 0.5*(-bet_size) = 0.5 * bet_size
        """
        return 0.5 * (2.0 * bet_size) + 0.5 * (-1.0 * bet_size)

    def optimal_ev_fraction(self) -> float:
        """Return the wealth fraction ``f`` that maximizes single-flip EV.

        Since ``single_flip_ev`` is linear and strictly increasing in
        ``bet_size`` (slope = +0.5), the maximizing fraction is the largest
        feasible fraction of wealth, i.e. ``f = 1.0`` (bet everything).
        """
        # EV(f) = 0.5 * f * wealth  -> derivative wrt f is 0.5 * wealth > 0
        # => EV is monotonically increasing in f on [0, 1] => argmax f = 1.0
        return 1.0

    def exact_all_in_ev(self) -> Fraction:
        """Exact expected terminal wealth for the "bet everything every
        time" strategy, computed with exact rational arithmetic.

        If you bet everything every flip, there are only two possible
        outcomes after ``n`` flips:

          * All heads (probability ``(1/2)^n``): wealth becomes
            ``initial_wealth * 3^n`` (each win multiplies wealth by 3:
            you keep your stake B and gain 2B, i.e. wealth -> 3*wealth).
          * At least one tails (probability ``1 - (1/2)^n``): wealth is 0,
            since the very first tails wipes you out (you bet everything).

        So:

            EV = (1/2)^n * (W0 * 3^n) + (1 - (1/2)^n) * 0
               = W0 * (3/2)^n

        Returns
        -------
        Fraction
            The exact expected terminal wealth, as a Python ``Fraction``
            for arbitrary precision.
        """
        w0 = Fraction(self.initial_wealth).limit_denominator()
        n = self.num_flips
        p_all_heads = Fraction(1, 2) ** n
        payout_all_heads = w0 * (Fraction(3, 1) ** n)
        ev = p_all_heads * payout_all_heads  # the (1 - p) * 0 term vanishes
        return ev

    def exact_all_in_ev_float(self) -> float:
        """Same as :meth:`exact_all_in_ev` but returned as a Python float
        (may overflow to ``inf`` for large ``num_flips`` since 3^100 is
        astronomically large -- use :meth:`exact_all_in_ev` for the exact
        rational value)."""
        return float(self.exact_all_in_ev())

    def probability_of_ruin_all_in(self) -> float:
        """Exact probability that the all-in strategy ends at $0.

        Ruin occurs unless *every single* flip is a win.

            P(ruin) = 1 - (1/2)^n
        """
        return 1.0 - (0.5 ** self.num_flips)

    def probability_of_survival_all_in(self) -> float:
        """P(survive) = (1/2)^n -- probability of an all-heads streak."""
        return 0.5 ** self.num_flips

    def summary(self) -> Dict[str, float]:
        """Return a dictionary summarizing the key analytical results."""
        ev_exact = self.exact_all_in_ev()
        return {
            "initial_wealth": self.initial_wealth,
            "num_flips": self.num_flips,
            "optimal_ev_fraction": self.optimal_ev_fraction(),
            "expected_terminal_wealth_all_in": float(ev_exact),
            "expected_terminal_wealth_all_in_exact": str(ev_exact),
            "probability_of_ruin_all_in": self.probability_of_ruin_all_in(),
            "probability_of_survival_all_in": self.probability_of_survival_all_in(),
        }


# --------------------------------------------------------------------------- #
# Simulation Engine
# --------------------------------------------------------------------------- #
@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation outputs for one strategy."""

    name: str
    final_wealth: np.ndarray  # shape (num_players,)
    wealth_paths: Optional[np.ndarray] = None  # shape (num_players, num_flips+1)

    @property
    def mean_final_wealth(self) -> float:
        return float(np.mean(self.final_wealth))

    @property
    def median_final_wealth(self) -> float:
        return float(np.median(self.final_wealth))

    @property
    def prob_ruin(self) -> float:
        """Fraction of players ending with (numerically) zero wealth."""
        return float(np.mean(self.final_wealth <= 1e-9))

    @property
    def prob_profit(self) -> float:
        """Fraction of players ending above their initial wealth."""
        return float(np.mean(self.final_wealth > 0))

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.final_wealth, q))

    def summary(self) -> Dict[str, float]:
        return {
            "strategy": self.name,
            "mean_final_wealth": self.mean_final_wealth,
            "median_final_wealth": self.median_final_wealth,
            "prob_ruin": self.prob_ruin,
            "p5": self.percentile(5),
            "p25": self.percentile(25),
            "p75": self.percentile(75),
            "p95": self.percentile(95),
        }


class BettingStrategy:
    """Base class for a fixed-fraction betting strategy.

    A fixed-fraction strategy bets ``fraction * current_wealth`` on every
    flip. ``fraction = 1.0`` reproduces the "bet everything" / EV-maximizing
    strategy. ``fraction = 0.0`` means "never bet" (wealth stays constant).
    """

    def __init__(self, name: str, fraction: float) -> None:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")
        self.name = name
        self.fraction = fraction

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"BettingStrategy(name={self.name!r}, fraction={self.fraction})"


class CoinBettingSimulator:
    """Vectorized Monte Carlo simulator for the coin betting game.

    Simulates ``num_players`` independent players, each flipping a fair
    coin ``num_flips`` times and betting according to a fixed-fraction
    strategy.
    """

    def __init__(
        self,
        initial_wealth: float = 100.0,
        num_flips: int = 100,
        seed: Optional[int] = 42,
    ) -> None:
        self.initial_wealth = initial_wealth
        self.num_flips = num_flips
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        strategy: BettingStrategy,
        num_players: int = 100_000,
        store_paths_for: int = 0,
    ) -> SimulationResult:
        """Run a vectorized Monte Carlo simulation for a single strategy.

        Parameters
        ----------
        strategy:
            The betting strategy (fixed fraction of wealth per flip).
        num_players:
            Number of independent simulated players.
        store_paths_for:
            If > 0, store full wealth trajectories for this many sample
            players (useful for plotting). Set to 0 to skip (saves memory
            for large ``num_players``).

        Returns
        -------
        SimulationResult
        """
        f = strategy.fraction
        wealth = np.full(num_players, self.initial_wealth, dtype=np.float64)

        paths: Optional[np.ndarray] = None
        n_store = min(store_paths_for, num_players)
        if n_store > 0:
            paths = np.empty((n_store, self.num_flips + 1), dtype=np.float64)
            paths[:, 0] = wealth[:n_store]

        # Coin flips: True = heads (win), False = tails (loss)
        flips = self.rng.random((self.num_flips, num_players)) < 0.5

        for t in range(self.num_flips):
            bet = f * wealth
            win_mask = flips[t]
            # Win: wealth += 2*bet ; Loss: wealth -= bet
            wealth = np.where(win_mask, wealth + 2.0 * bet, wealth - bet)
            # Guard against floating point dust below zero
            wealth = np.maximum(wealth, 0.0)
            if paths is not None:
                paths[:, t + 1] = wealth[:n_store]

        return SimulationResult(name=strategy.name, final_wealth=wealth, wealth_paths=paths)

    def compare_strategies(
        self,
        strategies: List[BettingStrategy],
        num_players: int = 100_000,
        store_paths_for: int = 200,
    ) -> Dict[str, SimulationResult]:
        """Run all provided strategies and return a dict of results keyed
        by strategy name."""
        results: Dict[str, SimulationResult] = {}
        for strat in strategies:
            results[strat.name] = self.run(
                strat, num_players=num_players, store_paths_for=store_paths_for
            )
        return results


def default_strategies(kelly_fraction: float = 0.25) -> List[BettingStrategy]:
    """Convenience factory returning the three strategies described in the
    problem statement:

      * Strategy A: EV-maximizing "all in" (f=1.0)
      * Strategy B: Kelly-style fractional betting (f=kelly_fraction)
      * Strategy C: Fixed 10% betting (f=0.10)
    """
    return [
        BettingStrategy("Strategy A: All-In (EV Max)", 1.0),
        BettingStrategy(f"Strategy B: Kelly-style (f={kelly_fraction})", kelly_fraction),
        BettingStrategy("Strategy C: Fixed 10%", 0.10),
    ]


# --------------------------------------------------------------------------- #
# Ergodicity Analysis
# --------------------------------------------------------------------------- #
class ErgodicityAnalyzer:
    """Utilities to illustrate the ensemble-average vs. time-average
    distinction (ergodicity economics) for fixed-fraction betting games.

    For a fixed-fraction bet of size ``f`` on a fair 2:1 coin flip, the
    *per-flip growth factor* is:

        * Win  (p=0.5): multiply wealth by (1 + 2f)
        * Loss (p=0.5): multiply wealth by (1 - f)

    The **ensemble average growth factor** (arithmetic mean) is:

        E[growth] = 0.5*(1 + 2f) + 0.5*(1 - f) = 1 + 0.5f

    This is > 1 for any f > 0, i.e. the *expected value* always favors
    betting more. But the **time-average growth rate** (what an individual
    player experiences over many repeated flips) is governed by the
    *geometric* mean, since wealth compounds multiplicatively:

        G(f) = (1 + 2f)^0.5 * (1 - f)^0.5

    Maximizing log(G(f)) over f gives the Kelly-optimal fraction f*, which
    is generally well below 1.0 -- this is the mathematical root of why
    "EV-maximizing" and "growth-maximizing" strategies diverge so sharply.
    """

    @staticmethod
    def ensemble_avg_growth_factor(f: float) -> float:
        """Arithmetic-mean (ensemble average) growth factor per flip."""
        win_factor = 1.0 + 2.0 * f
        loss_factor = 1.0 - f
        return 0.5 * win_factor + 0.5 * loss_factor

    @staticmethod
    def time_avg_growth_factor(f: float) -> float:
        """Geometric-mean (time average) growth factor per flip.

        Returns 0.0 for f=1.0 since (1-f)=0 implies certain eventual ruin.
        """
        win_factor = 1.0 + 2.0 * f
        loss_factor = 1.0 - f
        if loss_factor <= 0:
            return 0.0
        return (win_factor ** 0.5) * (loss_factor ** 0.5)

    @classmethod
    def kelly_optimal_fraction(cls, grid_size: int = 100_001) -> float:
        """Numerically find the fraction f in [0, 1) that maximizes the
        time-average (geometric) growth factor, i.e. the Kelly-optimal bet
        fraction for this specific 2:1 payout coin game.

        Analytically, for a bet that wins amount ``b`` per unit staked with
        probability ``p`` and loses the stake with probability ``q=1-p``,
        the Kelly fraction is ``f* = p - q/b``. Here ``b=2`` (2:1 payout),
        ``p=q=0.5``, so ``f* = 0.5 - 0.5/2 = 0.25``.
        """
        fs = np.linspace(0.0, 0.999999, grid_size)
        growth = (1.0 + 2.0 * fs) ** 0.5 * (1.0 - fs) ** 0.5
        return float(fs[np.argmax(growth)])

    @staticmethod
    def kelly_optimal_fraction_analytical() -> float:
        """Closed form Kelly fraction for this game: f* = p - q/b = 0.25."""
        p, q, b = 0.5, 0.5, 2.0
        return p - q / b

    @staticmethod
    def probability_of_ruin_after_n_flips(f: float, n: int) -> float:
        """Probability of hitting (numerically) zero wealth within ``n``
        flips under fixed-fraction betting with fraction ``f``.

        For f < 1, wealth can only asymptotically approach zero but never
        exactly reach it in a discrete simulation (it approaches zero
        exponentially at rate (1-f) per loss). For f == 1, a single loss
        immediately produces exact ruin. This helper returns the exact
        analytical probability for the f == 1 case and 0.0 (no exact ruin
        possible) for f < 1, reflecting the true mathematical behaviour.
        """
        if f >= 1.0:
            return 1.0 - (0.5 ** n)
        return 0.0
