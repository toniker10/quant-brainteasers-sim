"""
dice_sequence.py
=================

Module 2: The 3-4-5 Dice Sequence Problem.

Problem
-------
Roll a fair six-sided die repeatedly until the sequence ``[3, 4, 5]``
appears in three *consecutive* rolls. What is the probability that the
sequence first completes on an *odd*-numbered roll?

Approach
--------
We model the process as an **absorbing Markov chain** over "progress"
states that track how much of the target sequence has been matched so
far:

    State 0: no progress (last roll(s) did not extend a match of "3")
    State 1: last roll was "3" (matched 1 of 3 target symbols)
    State 2: last two rolls were "3, 4" (matched 2 of 3)
    State 3: "3, 4, 5" observed -> ABSORBING (sequence complete)

From each non-absorbing state, a die roll transitions us to a new state
depending on whether the roll extends, restarts, or breaks the partial
match. Because the target sequence [3, 4, 5] contains no self-overlap
(no proper prefix of the sequence is also a suffix of any partial match
in a way that creates ambiguity beyond simply checking "does this roll
equal the next needed symbol, and if not, could it start a new partial
match"), the transitions are:

    From State 0 (need '3'):
        roll == 3 (p=1/6): -> State 1
        roll != 3 (p=5/6): -> State 0

    From State 1 (need '4'):
        roll == 4 (p=1/6): -> State 2
        roll == 3 (p=1/6): -> State 1   (the new '3' restarts progress)
        else      (p=4/6): -> State 0

    From State 2 (need '5'):
        roll == 5 (p=1/6): -> State 3 (ABSORBED)
        roll == 3 (p=1/6): -> State 1   (the new '3' restarts progress)
        else      (p=4/6): -> State 0

To find P(absorption occurs on an ODD roll), we track two probability
vectors -- one for "currently at an even roll count" and one for "odd" --
or equivalently we build a system of linear equations for
``p_state = P(absorbed on an odd total roll count | currently in `state`,
about to make the NEXT roll)`` and solve directly with linear algebra.

Let ``p0, p1, p2`` be the probability that, starting from state 0, 1, or 2
respectively (with zero rolls made so far from that state), the sequence
completes on a roll whose index is odd RELATIVE to a fresh start from that
state. Since each transition consumes exactly one roll (flipping
even/odd parity), we can write:

    p0 = (1/6)*(1 - p1) + (5/6)*(1 - p0)
    p1 = (1/6)*(1 - p2) + (1/6)*(1 - p1) + (4/6)*(1 - p0)
    p2 = (1/6)*(1)      + (1/6)*(1 - p1) + (4/6)*(1 - p0)

Explanation of the "(1 - p_next)" terms: if we make one roll (flipping
parity) and move to a new state, the probability that the *original*
target roll count is odd equals the probability that the *remaining*
roll count (from the new state) is EVEN, i.e. ``1 - p_next`` (since
p_next is defined as the probability of odd completion from that state).
The single case where the roll *immediately* completes the sequence
(rolling a 5 from state 2) contributes probability 1 (that roll itself
is the completing, and its parity is odd relative to this sub-process
consuming exactly 1 roll).

Solving this 3x3 linear system yields the exact answer:

    p0 = P(Odd) = 216/431 ≈ 0.50116009...

This module solves the system both symbolically (via exact fractions)
and numerically (via ``numpy.linalg.solve``), and generalizes to
arbitrary target sequences over an arbitrary number of die faces.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Analytical Solver (hardcoded for the classic 3-4-5 problem)
# --------------------------------------------------------------------------- #
class DiceSequenceAnalytics:
    """Exact analytical solver for the classic "sequence [3,4,5] on an odd
    roll" problem, built from the 3-state absorbing Markov chain described
    in the module docstring.
    """

    NUM_FACES = 6

    def __init__(self) -> None:
        pass

    def build_linear_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build the ``A @ p = b`` linear system for
        ``p = [p0, p1, p2]`` where ``p_i`` is the probability of odd-roll
        completion starting fresh from state ``i``.

        Rearranging each equation of the form::

            p_i = sum_over_transitions( P(transition) * (1 - p_next) )
                  [+ P(immediate completion)]

        into standard linear form ``A @ p = b`` (moving all p terms to the
        left-hand side).

        Returns
        -------
        (A, b): np.ndarray, np.ndarray
            Coefficient matrix (3x3) and right-hand-side vector (3,).
        """
        # Equations (multiplying through by 6 to clear denominators):
        # 6*p0 = 1*(1-p1) + 5*(1-p0)
        # 6*p1 = 1*(1-p2) + 1*(1-p1) + 4*(1-p0)
        # 6*p2 = 1*(1)    + 1*(1-p1) + 4*(1-p0)
        #
        # => 6*p0 + 5*p0 + 1*p1            = 1 + 5        => 11*p0 + p1        = 6
        # => 6*p1 + 4*p0 + 1*p1 + 1*p2     = 1 + 1 + 4    => 4*p0 + 7*p1 + p2  = 6
        # => 6*p2 + 4*p0 + 1*p1            = 1 + 1 + 4    => 4*p0 + p1 + 6*p2  = 6
        A = np.array(
            [
                [11.0, 1.0, 0.0],
                [4.0, 7.0, 1.0],
                [4.0, 1.0, 6.0],
            ]
        )
        b = np.array([6.0, 6.0, 6.0])
        return A, b

    def solve_numeric(self) -> Dict[str, float]:
        """Solve the linear system numerically with ``numpy.linalg.solve``.

        Returns a dict with p0, p1, p2 where p0 is the answer to the
        original problem: P(the [3,4,5] sequence completes on an odd
        roll), starting fresh.
        """
        A, b = self.build_linear_system()
        p0, p1, p2 = np.linalg.solve(A, b)
        return {"p0_P_odd": float(p0), "p1": float(p1), "p2": float(p2)}

    def solve_exact(self) -> Dict[str, Fraction]:
        """Solve the same linear system using exact rational (Fraction)
        arithmetic via Gaussian elimination, to obtain the exact closed
        form fraction (e.g. 216/431).
        """
        # Coefficients as exact Fractions
        A = [
            [Fraction(11), Fraction(1), Fraction(0)],
            [Fraction(4), Fraction(7), Fraction(1)],
            [Fraction(4), Fraction(1), Fraction(6)],
        ]
        b = [Fraction(6), Fraction(6), Fraction(6)]
        p = _solve_linear_system_exact(A, b)
        return {"p0_P_odd": p[0], "p1": p[1], "p2": p[2]}

    def exact_answer(self) -> Fraction:
        """Return the exact fraction 216/431 (probability of odd
        completion) as a Python ``Fraction``."""
        return self.solve_exact()["p0_P_odd"]

    def summary(self) -> Dict[str, float]:
        exact = self.solve_exact()
        numeric = self.solve_numeric()
        return {
            "P_odd_exact_fraction": str(exact["p0_P_odd"]),
            "P_odd_exact_float": float(exact["p0_P_odd"]),
            "P_odd_numeric": numeric["p0_P_odd"],
            "P_even_exact_float": 1.0 - float(exact["p0_P_odd"]),
        }


def _solve_linear_system_exact(
    A: List[List[Fraction]], b: List[Fraction]
) -> List[Fraction]:
    """Solve ``A x = b`` exactly using Gaussian elimination over
    ``fractions.Fraction`` (no floating point rounding anywhere).
    """
    n = len(b)
    # Build augmented matrix
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    # Forward elimination with partial pivoting (exact arithmetic, so
    # pivoting is only needed to avoid dividing by zero).
    for col in range(n):
        pivot_row = None
        for r in range(col, n):
            if M[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            raise ValueError("Singular matrix - no unique solution")
        M[col], M[pivot_row] = M[pivot_row], M[col]

        pivot_val = M[col][col]
        M[col] = [v / pivot_val for v in M[col]]

        for r in range(n):
            if r != col and M[r][col] != 0:
                factor = M[r][col]
                M[r] = [M[r][k] - factor * M[col][k] for k in range(n + 1)]

    return [M[i][n] for i in range(n)]


# --------------------------------------------------------------------------- #
# Generalized Sequence Solver
# --------------------------------------------------------------------------- #
class GeneralizedSequenceSolver:
    """Solve "P(sequence completes on an odd roll)" for an *arbitrary*
    target sequence of die faces, on a die with an arbitrary number of
    faces (default 6).

    This builds the correct KMP-style ("failure function") automaton
    states for the target sequence, so that it correctly handles
    self-overlapping sequences (e.g. ``[6, 6]`` or ``[1, 1, 2, 1, 1]``)
    where a naive "restart from state 0/1" rule (as used in the
    hand-derivation for [3,4,5]) would be wrong.

    Parameters
    ----------
    sequence:
        The target sequence of die face values, e.g. ``[3, 4, 5]``.
    num_faces:
        Number of faces on the die (default 6 for a standard die). Faces
        are assumed to be the integers ``1..num_faces``, each equally
        likely (probability ``1/num_faces``).
    """

    def __init__(self, sequence: Sequence[int], num_faces: int = 6) -> None:
        if len(sequence) == 0:
            raise ValueError("sequence must be non-empty")
        if any(not (1 <= s <= num_faces) for s in sequence):
            raise ValueError(f"all sequence values must be in [1, {num_faces}]")
        self.sequence = list(sequence)
        self.num_faces = num_faces
        self.m = len(self.sequence)  # number of non-absorbing progress levels
        self.p_face = 1.0 / num_faces
        self._failure = self._build_failure_function(self.sequence)

    @staticmethod
    def _build_failure_function(seq: List[int]) -> List[int]:
        """Standard KMP failure (partial-match) function for the target
        sequence, used to correctly compute the "fallback" state when a
        roll breaks the current partial match.
        """
        m = len(seq)
        fail = [0] * m
        k = 0
        for i in range(1, m):
            while k > 0 and seq[i] != seq[k]:
                k = fail[k - 1]
            if seq[i] == seq[k]:
                k += 1
            fail[i] = k
        return fail

    def _transition(self, state: int, face: int) -> int:
        """Return the next automaton state given current ``state``
        (number of symbols matched so far, 0..m-1, since state m is
        absorbing and handled separately) and the newly rolled ``face``.
        """
        k = state
        while k > 0 and face != self.sequence[k]:
            k = self._failure[k - 1]
        if face == self.sequence[k]:
            k += 1
        return k

    def build_transition_matrix(self) -> np.ndarray:
        """Build the ``m x m`` sub-stochastic transition matrix ``T``
        among the non-absorbing states 0..m-1 (probability mass that goes
        to the absorbing state m is simply ``1 - row_sum``).
        """
        m = self.m
        T = np.zeros((m, m))
        for state in range(m):
            for face in range(1, self.num_faces + 1):
                nxt = self._transition(state, face)
                if nxt < m:
                    T[state, nxt] += self.p_face
                # else: absorbed, contributes to (1 - row_sum), not stored
        return T

    def absorption_prob_per_state(self) -> np.ndarray:
        """Return a length-m vector giving, for each state, the
        probability that the *very next* roll causes absorption (i.e.
        completes the sequence)."""
        m = self.m
        out = np.zeros(m)
        for state in range(m):
            count = 0
            for face in range(1, self.num_faces + 1):
                if self._transition(state, face) == m:
                    count += 1
            out[state] = count * self.p_face
        return out

    def solve(self) -> Dict[str, float]:
        """Solve for P(Odd) and P(Even) that the sequence completes,
        starting fresh from state 0, using the same "flip parity per
        roll" linear-system technique as :class:`DiceSequenceAnalytics`,
        generalized to ``m`` states via ``numpy.linalg.solve``.

        For each state i, let ``p_i`` = P(sequence completes on an odd
        total roll count | starting fresh from state i). Then:

            p_i = sum_j T[i,j] * (1 - p_j) + a_i

        where ``a_i`` is the probability of *immediate* absorption from
        state i (which counts as completing on this one roll, i.e.
        contributes probability 1 * a_i to the odd branch since a single
        roll is odd).

        Rearranged: p_i - sum_j T[i,j]*(-p_j) = sum_j T[i,j] + a_i
                 => p_i + sum_j T[i,j]*p_j = row_sum(T)_i + a_i

        i.e. in matrix form: (I + T) @ p = row_sum(T) + a
        """
        T = self.build_transition_matrix()
        a = self.absorption_prob_per_state()
        m = self.m
        I = np.eye(m)
        row_sum = T.sum(axis=1)
        A_mat = I + T
        rhs = row_sum + a
        p = np.linalg.solve(A_mat, rhs)
        p_odd = float(p[0])
        return {
            "sequence": self.sequence,
            "P_odd": p_odd,
            "P_even": 1.0 - p_odd,
        }


# --------------------------------------------------------------------------- #
# Monte Carlo Validation Engine
# --------------------------------------------------------------------------- #
@dataclass
class DiceMonteCarloResult:
    """Container for Monte Carlo dice-sequence simulation results."""

    sequence: List[int]
    roll_counts: np.ndarray  # total rolls needed, per trial
    completed_on_odd: np.ndarray  # bool array, per trial

    @property
    def num_trials(self) -> int:
        return len(self.roll_counts)

    @property
    def empirical_p_odd(self) -> float:
        return float(np.mean(self.completed_on_odd))

    @property
    def mean_rolls_to_complete(self) -> float:
        return float(np.mean(self.roll_counts))

    def convergence_curve(self, checkpoints: Sequence[int]) -> Dict[int, float]:
        """Return empirical P(Odd) computed using only the first ``k``
        trials, for each ``k`` in ``checkpoints`` (useful for plotting
        Monte Carlo convergence toward the analytical value).
        """
        out = {}
        for k in checkpoints:
            k = min(k, self.num_trials)
            if k <= 0:
                continue
            out[k] = float(np.mean(self.completed_on_odd[:k]))
        return out


class DiceSequenceSimulator:
    """Monte Carlo simulator that repeatedly rolls a fair die and checks
    for the target sequence, for validation against the analytical
    solvers above.
    """

    def __init__(self, num_faces: int = 6, seed: int = 42) -> None:
        self.num_faces = num_faces
        self.rng = np.random.default_rng(seed)

    def simulate_single_trial(self, sequence: Sequence[int], max_rolls: int = 100_000) -> Tuple[int, bool]:
        """Roll the die until ``sequence`` appears consecutively (using a
        KMP automaton so overlapping sequences are handled correctly).
        Returns ``(num_rolls, completed_on_odd_roll)``.
        """
        solver = GeneralizedSequenceSolver(sequence, self.num_faces)
        state = 0
        m = solver.m
        rolls = 0
        while rolls < max_rolls:
            face = int(self.rng.integers(1, self.num_faces + 1))
            rolls += 1
            state = solver._transition(state, face)
            if state == m:
                return rolls, (rolls % 2 == 1)
        raise RuntimeError("max_rolls exceeded without completing sequence")

    def run(
        self,
        sequence: Sequence[int] = (3, 4, 5),
        num_trials: int = 1_000_000,
        max_rolls_per_trial: int = 10_000,
        chunk_rolls: int = 64,
    ) -> DiceMonteCarloResult:
        """Fully vectorized Monte Carlo simulation of ``num_trials``
        independent trials of rolling a die until ``sequence`` appears.

        All ``num_trials`` trials are advanced *in parallel*: on each
        iteration we draw a chunk of ``chunk_rolls`` random rolls for
        every trial that hasn't yet completed, update each trial's
        automaton state roll-by-roll (vectorized across trials with NumPy
        fancy indexing), and record the completion roll index the first
        time each trial's state reaches the absorbing state ``m``. Trials
        that finish early are excluded from subsequent chunks, so the
        cost shrinks as more trials complete. This comfortably handles
        ``num_trials=1_000_000`` in well under a minute on a laptop.
        """
        sequence = list(sequence)
        solver = GeneralizedSequenceSolver(sequence, self.num_faces)
        m = solver.m

        # Precompute transition lookup table: table[state, face-1] = next_state.
        # Row m (the absorbing state) is included and maps to itself for
        # every face, so that already-finished trials can be safely kept
        # in the batched update without special-casing their indices.
        table = np.zeros((m + 1, self.num_faces), dtype=np.int64)
        for state in range(m):
            for face in range(1, self.num_faces + 1):
                table[state, face - 1] = solver._transition(state, face)
        table[m, :] = m  # absorbing state stays absorbed

        roll_counts = np.zeros(num_trials, dtype=np.int64)
        done = np.zeros(num_trials, dtype=bool)
        states = np.zeros(num_trials, dtype=np.int64)
        active_idx = np.arange(num_trials)  # indices of trials still running
        total_rolls_elapsed = 0

        while active_idx.size > 0:
            if total_rolls_elapsed > max_rolls_per_trial:
                raise RuntimeError(
                    f"Exceeded max_rolls_per_trial={max_rolls_per_trial} "
                    f"with {active_idx.size} trials still incomplete."
                )

            n_active = active_idx.size
            # Draw a chunk of rolls (1..num_faces) for every active trial
            rolls_chunk = self.rng.integers(1, self.num_faces + 1, size=(chunk_rolls, n_active))

            cur_states = states[active_idx]
            just_finished_mask = np.zeros(n_active, dtype=bool)
            finish_roll_offset = np.zeros(n_active, dtype=np.int64)

            for step in range(chunk_rolls):
                faces = rolls_chunk[step]
                cur_states = table[cur_states, faces - 1]
                newly_done = (cur_states == m) & (~just_finished_mask)
                finish_roll_offset[newly_done] = step + 1
                just_finished_mask |= newly_done
                # Trials that already finished within this chunk should not
                # keep transitioning (freeze their state at absorption).
                cur_states = np.where(just_finished_mask, m, cur_states)
                if just_finished_mask.all():
                    break

            total_rolls_elapsed += step + 1  # rolls consumed this chunk

            finished_local = np.nonzero(just_finished_mask)[0]
            if finished_local.size > 0:
                finished_global = active_idx[finished_local]
                # Total rolls consumed = rolls elapsed before this chunk
                # plus this trial's offset within the current chunk.
                rolls_before_chunk = total_rolls_elapsed - (step + 1)
                roll_counts[finished_global] = rolls_before_chunk + finish_roll_offset[finished_local]
                done[finished_global] = True

            # Update states for trials still not finished (persist progress)
            still_running_local = np.nonzero(~just_finished_mask)[0]
            still_running_global = active_idx[still_running_local]
            states[still_running_global] = cur_states[still_running_local]

            active_idx = still_running_global

        completed_odd = (roll_counts % 2 == 1)

        return DiceMonteCarloResult(
            sequence=sequence, roll_counts=roll_counts, completed_on_odd=completed_odd
        )
