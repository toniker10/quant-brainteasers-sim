"""Unit tests for src/betting_game.py"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.betting_game import (
    BettingAnalytics,
    BettingStrategy,
    CoinBettingSimulator,
    ErgodicityAnalyzer,
    default_strategies,
)


# --------------------------------------------------------------------------- #
# BettingAnalytics
# --------------------------------------------------------------------------- #
class TestBettingAnalytics:
    def test_single_flip_ev_positive_for_positive_bet(self):
        ev = BettingAnalytics.single_flip_ev(100.0)
        assert ev == pytest.approx(50.0)

    def test_single_flip_ev_zero_for_zero_bet(self):
        assert BettingAnalytics.single_flip_ev(0.0) == 0.0

    def test_optimal_ev_fraction_is_one(self):
        analytics = BettingAnalytics(initial_wealth=100.0, num_flips=100)
        assert analytics.optimal_ev_fraction() == 1.0

    def test_exact_all_in_ev_matches_closed_form(self):
        analytics = BettingAnalytics(initial_wealth=100.0, num_flips=100)
        ev = analytics.exact_all_in_ev()
        expected = (100 * (3**100)) / (2**100)
        assert float(ev) == pytest.approx(expected, rel=1e-9)

    def test_probability_of_ruin_all_in_near_certain(self):
        analytics = BettingAnalytics(initial_wealth=100.0, num_flips=100)
        p_ruin = analytics.probability_of_ruin_all_in()
        assert p_ruin > 0.999999999999999
        assert p_ruin <= 1.0

    def test_probability_of_survival_matches_two_to_the_n(self):
        analytics = BettingAnalytics(initial_wealth=100.0, num_flips=10)
        p_survive = analytics.probability_of_survival_all_in()
        assert p_survive == pytest.approx(1.0 / (2**10))

    def test_summary_contains_expected_keys(self):
        analytics = BettingAnalytics()
        summary = analytics.summary()
        expected_keys = {
            "initial_wealth",
            "num_flips",
            "optimal_ev_fraction",
            "expected_terminal_wealth_all_in",
            "expected_terminal_wealth_all_in_exact",
            "probability_of_ruin_all_in",
            "probability_of_survival_all_in",
        }
        assert expected_keys.issubset(summary.keys())


# --------------------------------------------------------------------------- #
# BettingStrategy
# --------------------------------------------------------------------------- #
class TestBettingStrategy:
    def test_valid_fraction_accepted(self):
        strat = BettingStrategy("test", 0.5)
        assert strat.fraction == 0.5

    @pytest.mark.parametrize("bad_fraction", [-0.1, 1.1, 2.0])
    def test_invalid_fraction_raises(self, bad_fraction):
        with pytest.raises(ValueError):
            BettingStrategy("bad", bad_fraction)


# --------------------------------------------------------------------------- #
# CoinBettingSimulator
# --------------------------------------------------------------------------- #
class TestCoinBettingSimulator:
    def test_zero_fraction_wealth_unchanged(self):
        sim = CoinBettingSimulator(initial_wealth=100.0, num_flips=50, seed=1)
        strat = BettingStrategy("never bet", 0.0)
        result = sim.run(strat, num_players=1000)
        assert np.allclose(result.final_wealth, 100.0)

    def test_all_in_strategy_wealth_is_zero_or_huge(self):
        sim = CoinBettingSimulator(initial_wealth=100.0, num_flips=20, seed=2)
        strat = BettingStrategy("all in", 1.0)
        result = sim.run(strat, num_players=5000)
        wealth = result.final_wealth
        # Every outcome should be either 0 (any loss) or 100 * 3^20 (all wins)
        huge_value = 100.0 * (3.0**20)
        is_zero = np.isclose(wealth, 0.0)
        is_huge = np.isclose(wealth, huge_value, rtol=1e-6)
        assert np.all(is_zero | is_huge)

    def test_all_in_strategy_prob_ruin_matches_analytics(self):
        num_flips = 15
        sim = CoinBettingSimulator(initial_wealth=100.0, num_flips=num_flips, seed=3)
        strat = BettingStrategy("all in", 1.0)
        result = sim.run(strat, num_players=200_000)
        analytics = BettingAnalytics(initial_wealth=100.0, num_flips=num_flips)
        expected_ruin = analytics.probability_of_ruin_all_in()
        # Monte carlo estimate should be close to analytical value
        assert result.prob_ruin == pytest.approx(expected_ruin, abs=0.01)

    def test_wealth_never_negative(self):
        sim = CoinBettingSimulator(initial_wealth=100.0, num_flips=100, seed=4)
        strat = BettingStrategy("fixed 10%", 0.10)
        result = sim.run(strat, num_players=1000)
        assert np.all(result.final_wealth >= 0.0)

    def test_store_paths_shape(self):
        sim = CoinBettingSimulator(initial_wealth=100.0, num_flips=10, seed=5)
        strat = BettingStrategy("kelly", 0.25)
        result = sim.run(strat, num_players=100, store_paths_for=10)
        assert result.wealth_paths is not None
        assert result.wealth_paths.shape == (10, 11)
        assert np.allclose(result.wealth_paths[:, 0], 100.0)

    def test_compare_strategies_returns_all_names(self):
        sim = CoinBettingSimulator(initial_wealth=100.0, num_flips=20, seed=6)
        strategies = default_strategies()
        results = sim.compare_strategies(strategies, num_players=2000, store_paths_for=5)
        assert set(results.keys()) == {s.name for s in strategies}
        for res in results.values():
            assert len(res.final_wealth) == 2000

    def test_summary_keys(self):
        sim = CoinBettingSimulator(initial_wealth=100.0, num_flips=10, seed=7)
        strat = BettingStrategy("fixed 10%", 0.10)
        result = sim.run(strat, num_players=500)
        summary = result.summary()
        expected_keys = {"strategy", "mean_final_wealth", "median_final_wealth", "prob_ruin", "p5", "p25", "p75", "p95"}
        assert expected_keys.issubset(summary.keys())


# --------------------------------------------------------------------------- #
# ErgodicityAnalyzer
# --------------------------------------------------------------------------- #
class TestErgodicityAnalyzer:
    def test_ensemble_avg_growth_increasing_in_f(self):
        g_low = ErgodicityAnalyzer.ensemble_avg_growth_factor(0.1)
        g_high = ErgodicityAnalyzer.ensemble_avg_growth_factor(0.9)
        assert g_high > g_low > 1.0

    def test_time_avg_growth_zero_at_f_equals_one(self):
        assert ErgodicityAnalyzer.time_avg_growth_factor(1.0) == 0.0

    def test_time_avg_growth_peaks_near_kelly_fraction(self):
        f_star = ErgodicityAnalyzer.kelly_optimal_fraction()
        assert f_star == pytest.approx(0.25, abs=0.01)

    def test_kelly_optimal_fraction_analytical_is_quarter(self):
        assert ErgodicityAnalyzer.kelly_optimal_fraction_analytical() == pytest.approx(0.25)

    def test_probability_of_ruin_after_n_flips_all_in(self):
        p = ErgodicityAnalyzer.probability_of_ruin_after_n_flips(1.0, 10)
        assert p == pytest.approx(1 - 0.5**10)

    def test_probability_of_ruin_after_n_flips_fractional_bet(self):
        p = ErgodicityAnalyzer.probability_of_ruin_after_n_flips(0.25, 10)
        assert p == 0.0
