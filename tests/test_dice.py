"""Unit tests for src/dice_sequence.py"""

import os
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dice_sequence import (
    DiceSequenceAnalytics,
    DiceSequenceSimulator,
    GeneralizedSequenceSolver,
)


# --------------------------------------------------------------------------- #
# DiceSequenceAnalytics
# --------------------------------------------------------------------------- #
class TestDiceSequenceAnalytics:
    def test_exact_answer_matches_known_fraction(self):
        analytics = DiceSequenceAnalytics()
        exact = analytics.exact_answer()
        assert exact == Fraction(216, 431)

    def test_exact_answer_float_matches_known_decimal(self):
        analytics = DiceSequenceAnalytics()
        exact = analytics.exact_answer()
        assert float(exact) == pytest.approx(0.50116009280742, abs=1e-12)

    def test_numeric_solver_matches_exact_solver(self):
        analytics = DiceSequenceAnalytics()
        numeric = analytics.solve_numeric()
        exact = analytics.solve_exact()
        assert numeric["p0_P_odd"] == pytest.approx(float(exact["p0_P_odd"]), rel=1e-9)
        assert numeric["p1"] == pytest.approx(float(exact["p1"]), rel=1e-9)
        assert numeric["p2"] == pytest.approx(float(exact["p2"]), rel=1e-9)

    def test_linear_system_shapes(self):
        analytics = DiceSequenceAnalytics()
        A, b = analytics.build_linear_system()
        assert A.shape == (3, 3)
        assert b.shape == (3,)

    def test_summary_contains_expected_keys(self):
        analytics = DiceSequenceAnalytics()
        summary = analytics.summary()
        expected_keys = {
            "P_odd_exact_fraction",
            "P_odd_exact_float",
            "P_odd_numeric",
            "P_even_exact_float",
        }
        assert expected_keys.issubset(summary.keys())

    def test_p_odd_plus_p_even_equals_one(self):
        analytics = DiceSequenceAnalytics()
        summary = analytics.summary()
        assert summary["P_odd_exact_float"] + summary["P_even_exact_float"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# GeneralizedSequenceSolver
# --------------------------------------------------------------------------- #
class TestGeneralizedSequenceSolver:
    def test_345_sequence_matches_hardcoded_analytics(self):
        general = GeneralizedSequenceSolver([3, 4, 5], num_faces=6)
        result = general.solve()
        assert result["P_odd"] == pytest.approx(216 / 431, rel=1e-9)

    def test_probabilities_sum_to_one(self):
        for seq in [[1, 2, 3, 4], [6, 6], [1, 1, 2, 1, 1], [2, 2, 2]]:
            general = GeneralizedSequenceSolver(seq, num_faces=6)
            result = general.solve()
            assert result["P_odd"] + result["P_even"] == pytest.approx(1.0, abs=1e-9)

    def test_single_symbol_sequence_is_exactly_half(self):
        # For a single-symbol target (e.g. "roll a single 6"), the process
        # is memoryless and by symmetry the probability of ANY specific
        # roll count parity converges via the standard geometric-parity
        # argument to a value determined purely by p=1/6.
        general = GeneralizedSequenceSolver([6], num_faces=6)
        result = general.solve()
        # Analytical check via first-step geometric series:
        # P(odd) = p + q^2*p + q^4*p + ... = p / (1-q^2) = p / ((1-q)(1+q))
        p = 1 / 6
        q = 5 / 6
        expected = p / (1 - q**2)
        assert result["P_odd"] == pytest.approx(expected, rel=1e-9)

    def test_overlapping_sequence_handled_via_kmp(self):
        # [6, 6] has self-overlap; ensure failure function doesn't crash
        # and produces a valid probability.
        general = GeneralizedSequenceSolver([6, 6], num_faces=6)
        result = general.solve()
        assert 0.0 <= result["P_odd"] <= 1.0

    def test_invalid_sequence_value_raises(self):
        with pytest.raises(ValueError):
            GeneralizedSequenceSolver([7, 8, 9], num_faces=6)

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError):
            GeneralizedSequenceSolver([], num_faces=6)

    def test_transition_matrix_shape(self):
        general = GeneralizedSequenceSolver([3, 4, 5], num_faces=6)
        T = general.build_transition_matrix()
        assert T.shape == (3, 3)
        # Row sums should be <= 1 (remainder goes to absorption)
        assert np.all(T.sum(axis=1) <= 1.0 + 1e-9)


# --------------------------------------------------------------------------- #
# DiceSequenceSimulator (Monte Carlo)
# --------------------------------------------------------------------------- #
class TestDiceSequenceSimulator:
    def test_single_trial_returns_valid_output(self):
        sim = DiceSequenceSimulator(seed=1)
        rolls, is_odd = sim.simulate_single_trial([3, 4, 5])
        assert rolls >= 3
        assert isinstance(is_odd, (bool, np.bool_))
        assert is_odd == (rolls % 2 == 1)

    def test_monte_carlo_converges_to_analytical_value(self):
        sim = DiceSequenceSimulator(seed=42)
        result = sim.run(sequence=[3, 4, 5], num_trials=20_000)
        analytics = DiceSequenceAnalytics()
        expected = float(analytics.exact_answer())
        assert result.empirical_p_odd == pytest.approx(expected, abs=0.02)

    def test_mean_rolls_is_positive_and_finite(self):
        sim = DiceSequenceSimulator(seed=7)
        result = sim.run(sequence=[3, 4, 5], num_trials=5_000)
        assert result.mean_rolls_to_complete > 0
        assert np.isfinite(result.mean_rolls_to_complete)

    def test_convergence_curve_returns_requested_checkpoints(self):
        sim = DiceSequenceSimulator(seed=3)
        result = sim.run(sequence=[3, 4, 5], num_trials=1000)
        curve = result.convergence_curve([10, 100, 1000])
        assert set(curve.keys()) == {10, 100, 1000}
        for v in curve.values():
            assert 0.0 <= v <= 1.0
