"""Generate publication-quality PNG charts for the README from the actual
simulation/analysis code (not mockups) — run once to populate assets/."""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.betting_game import (
    BettingAnalytics,
    CoinBettingSimulator,
    ErgodicityAnalyzer,
    default_strategies,
)
from src.dice_sequence import DiceSequenceAnalytics, DiceSequenceSimulator

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COLOR_RED = "#d1495b"
COLOR_BLUE = "#2e86ab"
COLOR_TEAL = "#118a7e"
COLOR_GRAY = "#6c757d"

# --------------------------------------------------------------------- #
# 1. Wealth trajectories: All-in vs Kelly (log scale)
# --------------------------------------------------------------------- #
NUM_FLIPS = 100
kelly_f = ErgodicityAnalyzer.kelly_optimal_fraction_analytical()
sim = CoinBettingSimulator(initial_wealth=100.0, num_flips=NUM_FLIPS, seed=7)
strategies = default_strategies(kelly_fraction=kelly_f)
results = sim.compare_strategies(strategies, num_players=20_000, store_paths_for=400)

all_in = results["Strategy A: All-In (EV Max)"]
kelly_name = [k for k in results if k.startswith("Strategy B")][0]
kelly = results[kelly_name]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=False)
for i in range(80):
    axes[0].plot(all_in.wealth_paths[i] + 1e-9, color=COLOR_RED, alpha=0.25, linewidth=0.8)
axes[0].set_yscale("log")
axes[0].set_title("All-In (f = 1.0)", fontweight="bold")
axes[0].set_xlabel("Flip number")
axes[0].set_ylabel("Wealth ($, log scale)")

for i in range(80):
    axes[1].plot(kelly.wealth_paths[i] + 1e-9, color=COLOR_BLUE, alpha=0.35, linewidth=0.8)
axes[1].set_yscale("log")
axes[1].set_title(f"Kelly Fraction (f = {kelly_f:.2f})", fontweight="bold")
axes[1].set_xlabel("Flip number")

fig.suptitle("Sample Wealth Trajectories Over 100 Coin Flips", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("assets/wealth_trajectories.png", bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------- #
# 2. Ensemble vs time-average growth
# --------------------------------------------------------------------- #
f_grid = np.linspace(0.0, 0.999, 500)
ensemble = [ErgodicityAnalyzer.ensemble_avg_growth_factor(f) for f in f_grid]
time_avg = [ErgodicityAnalyzer.time_avg_growth_factor(f) for f in f_grid]

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(f_grid, ensemble, label="Ensemble average (arithmetic mean)", color=COLOR_RED, linewidth=2)
ax.plot(f_grid, time_avg, label="Time average (geometric mean)", color=COLOR_BLUE, linewidth=2)
ax.axvline(kelly_f, color=COLOR_GRAY, linestyle="--", linewidth=1.3, label=f"Kelly optimum f* = {kelly_f:.2f}")
ax.axhline(1.0, color="black", linewidth=0.6)
ax.set_xlabel("Fraction of wealth bet per flip (f)")
ax.set_ylabel("Growth factor per flip")
ax.set_title("Ensemble Average vs. Time Average Growth", fontweight="bold")
ax.legend(frameon=False, fontsize=9.5)
plt.tight_layout()
plt.savefig("assets/ergodicity_growth.png", bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------- #
# 3. Dice: Monte Carlo convergence
# --------------------------------------------------------------------- #
dice_analytics = DiceSequenceAnalytics()
exact_p_odd = float(dice_analytics.exact_answer())

dice_sim = DiceSequenceSimulator(seed=42)
mc_result = dice_sim.run(sequence=[3, 4, 5], num_trials=500_000)

checkpoints = np.unique(np.logspace(2, np.log10(500_000), 60).astype(int))
curve = mc_result.convergence_curve(checkpoints)
xs = sorted(curve.keys())
ys = [curve[x] for x in xs]

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(xs, ys, color=COLOR_BLUE, marker="o", markersize=2.5, linewidth=1.1, label="Empirical P(Odd)")
ax.axhline(exact_p_odd, color=COLOR_RED, linestyle="--", linewidth=1.5,
           label=f"Analytical: 216/431 = {exact_p_odd:.5f}")
ax.set_xscale("log")
ax.set_xlabel("Number of simulated trials (log scale)")
ax.set_ylabel("Empirical P(Odd)")
ax.set_title("Monte Carlo Convergence to Analytical Value", fontweight="bold")
ax.legend(frameon=False, fontsize=9.5)
plt.tight_layout()
plt.savefig("assets/dice_convergence.png", bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------- #
# 4. Dice: rolls-to-completion distribution
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.hist(mc_result.roll_counts, bins=range(3, 200, 3), color=COLOR_TEAL, alpha=0.85)
ax.axvline(mc_result.mean_rolls_to_complete, color=COLOR_RED, linestyle="--", linewidth=1.5,
           label=f"Mean = {mc_result.mean_rolls_to_complete:.1f} rolls")
ax.set_xlabel("Number of rolls until [3, 4, 5] appears")
ax.set_ylabel("Frequency (out of 500,000 trials)")
ax.set_title("Distribution of Rolls-to-Completion", fontweight="bold")
ax.legend(frameon=False, fontsize=9.5)
plt.tight_layout()
plt.savefig("assets/dice_rolls_distribution.png", bbox_inches="tight")
plt.close()

# --------------------------------------------------------------------- #
# 5. Final wealth distributions across strategies
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
colors = [COLOR_RED, COLOR_BLUE, COLOR_TEAL]
for ax, (name, res), color in zip(axes, results.items(), colors):
    wealth = res.final_wealth
    nonzero = wealth[wealth > 1e-6]
    if len(nonzero) > 1:
        ax.hist(np.log10(nonzero + 1), bins=40, color=color, alpha=0.8)
    short_name = name.split(":")[1].strip() if ":" in name else name
    ax.set_title(short_name, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("log10(final wealth + 1)")
    ax.set_ylabel("Count")

fig.suptitle("Distribution of Final Wealth Across 20,000 Simulated Players", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("assets/final_wealth_distributions.png", bbox_inches="tight")
plt.close()

print("All charts saved to assets/")
print("Kelly fraction:", kelly_f)
print("Exact P(Odd):", exact_p_odd)
print("Empirical P(Odd) (500k trials):", mc_result.empirical_p_odd)
