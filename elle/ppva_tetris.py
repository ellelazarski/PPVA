#!/usr/bin/env python3
"""
PPVA controller for three Tetris goals
H0 -> R1 -> H2 -> R3
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
import numpy as np

from mbag.environment.types import MbagObs, CURRENT_BLOCKS
from elle.ppva_action_simulator import simulate_block_edit
from elle.ppva_structural_validity import get_all_structurally_valid_actions


# ===================== Types =====================

Action = Tuple[int, int, int, int]
Cell   = Tuple[int, int, int]
Goal   = Set[Cell]


@dataclass
class BeliefState:
    log_weights: np.ndarray
    belief: np.ndarray


# ===================== Utilities =====================

def is_filled(obs: MbagObs, x: int, y: int, z: int) -> bool:
    return obs[0][CURRENT_BLOCKS, x, y, z] != 0


def remaining_cells_list(obs: MbagObs, G: Goal) -> List[Cell]:
    return [(x, y, z) for (x, y, z) in G if not is_filled(obs, x, y, z)]


def filter_actions_to_targets(actions: List[Action], targets: Set[Cell]) -> List[Action]:
    return [(x, y, z, m) for (x, y, z, m) in actions if m != 0 and (x, y, z) in targets]


def reachable_remaining(obs: MbagObs, G: Goal, mats: Tuple[int, ...]) -> int:
    legal = get_all_structurally_valid_actions(obs=obs, placeable_materials=mats)
    reachable = {(x, y, z) for (x, y, z, m) in legal if m != 0}
    return sum((c in reachable) and (not is_filled(obs, *c)) for c in G)


def remaining_count(obs: MbagObs, G: Goal) -> int:
    """Number of unfilled goal cells."""
    return len(remaining_cells_list(obs, G))

# ===================== Q =====================

def Q_reward_to_go(
    s_before: MbagObs,
    a_h: Action,
    a_r: Action,
    G: Goal,
    gamma: float,
    mats: Tuple[int, ...],
) -> float:
    xh, yh, zh, mh = a_h
    xr, yr, zr, mr = a_r

    r_h = float(mh != 0 and (xh, yh, zh) in G and not is_filled(s_before, xh, yh, zh))
    s_h = simulate_block_edit(s_before, a_h)

    r_r = float(mr != 0 and (xr, yr, zr) in G and not is_filled(s_h, xr, yr, zr))
    s_p = simulate_block_edit(s_h, a_r)

    # V = reachable_remaining(s_p, G, mats)
    V = remaining_count(s_p, G)
    return r_h + r_r + gamma * V


# ===================== Controller =====================

class PPVATetris:

    def __init__(
        self,
        placeable_materials: Tuple[int, ...],
        beta_R: float,
        beta_H: float,
        gamma: float,
        goals_abs: List[Goal],
    ):
        self.placeable_materials = tuple(placeable_materials)
        self.beta_R = beta_R
        self.beta_H = beta_H
        self.gamma = gamma

        self.goals_abs = goals_abs
        self.K = len(goals_abs)

        # prior = np.ones(self.K) / self.K
        prior = np.array([0.22, 0.68, 0.1])
        self.state = BeliefState(np.log(prior), prior)

        self._cached_A_r: Optional[List[Action]] = None
        self._cached_Q_per_goal: Optional[np.ndarray] = None
    

    # ==========================================================
    # H0: Solipsistic human likelihood
    # ==========================================================

    def _log_pi_H0(self, s_before: MbagObs, A_h: List[Action]) -> np.ndarray:
        print("\n==================== H0 ====================")

        logP = np.zeros((self.K, len(A_h)))

        for k, G in enumerate(self.goals_abs):
            Qs = []
            for a_h in A_h:
                x, y, z, m = a_h
                r = float(m != 0 and (x, y, z) in G and not is_filled(s_before, x, y, z))
                s_h = simulate_block_edit(s_before, a_h)
                # V = reachable_remaining(s_h, G, self.placeable_materials)
                V = remaining_count(s_h, G)
                Qs.append(r + self.gamma * V)

            logits = self.beta_H * (np.array(Qs) - np.max(Qs))
            logP[k] = logits - np.log(np.exp(logits).sum() + 1e-12)

            print(f"\nGoal G{k+1}")
            print("Q_H0:", Qs)
            print("pi_H0:", np.exp(logP[k]))

        print("============================================")
        return logP
    

    # ==========================================================
    # R1: Robot response model
    # ==========================================================

    def _pi_R1(self, s_before: MbagObs, a_h: Action, belief: np.ndarray):
        print("\n==================== R1 ====================")
        print("Belief used for R1:", belief)
        print("Human action a_h:", a_h)

        s_h = simulate_block_edit(s_before, a_h)

        targets = set()
        for G in self.goals_abs:
            targets |= set(remaining_cells_list(s_h, G))

        legal = get_all_structurally_valid_actions(s_h, self.placeable_materials)
        A_r = filter_actions_to_targets(legal, targets) or [a for a in legal if a[3] != 0]

        print("Candidate robot actions A_r:")
        for i, a in enumerate(A_r):
            print(f"  [{i}] {a}")

        Qpg = np.zeros((self.K, len(A_r)))
        Qmix = np.zeros(len(A_r))

        for j, a_r in enumerate(A_r):
            for k, G in enumerate(self.goals_abs):
                Qpg[k, j] = Q_reward_to_go(s_before, a_h, a_r, G, self.gamma, self.placeable_materials)
            Qmix[j] = belief @ Qpg[:, j]

        print("\nQ_per_goal (rows = goals, cols = robot actions):")
        print(Qpg)
        print("\nQmix (belief-weighted):", Qmix)

        logits = self.beta_R * (Qmix - Qmix.max())
        pi_r = np.exp(logits) / (np.exp(logits).sum() + 1e-12)

        print("R1 softmax policy pi_r:", pi_r)
        print("============================================")

        return A_r, pi_r, Qpg

    
    # ==========================================================
    # H2: Pedagogic human likelihood
    # ==========================================================

    def _log_pi_H2(self, s_before: MbagObs, A_h: List[Action]) -> np.ndarray:
        print("\n==================== H2 ====================")

        logP_H0 = self._log_pi_H0(s_before, A_h)
        EV = np.zeros((self.K, len(A_h)))

        for i, a_h in enumerate(A_h):
            log_post = self.state.log_weights + logP_H0[:, i]
            log_post -= np.log(np.exp(log_post).sum() + 1e-12)
            b_tp = np.exp(log_post)

            print(f"\nCandidate human action a_h[{i}] = {a_h}")
            print("Induced R1 belief:", b_tp)

            A_r, pi_r, Qpg = self._pi_R1(s_before, a_h, b_tp)
            for k in range(self.K):
                EV[k, i] = np.sum(pi_r * Qpg[k])

            print("EV for this action:", EV[:, i])

        logP_H2 = np.zeros_like(EV)
        for k in range(self.K):
            logits = self.beta_H * (EV[k] - EV[k].max())
            logP_H2[k] = logits - np.log(np.exp(logits).sum() + 1e-12)

            print(f"\nGoal G{k+1} pedagogic EV:", EV[k])
            print("pi_H2:", np.exp(logP_H2[k]))

        print("============================================")
        return logP_H2

    # ==========================================================
    # Belief update (R3)
    # ==========================================================

    def belief_update(self, s_before: MbagObs, a_h_obs: Action, is_anchor: bool) -> None:
        print("\n==================== R3 BELIEF UPDATE ====================")
        print("Observed human action:", a_h_obs)
        print("Prior belief:", self.state.belief)

        if is_anchor:
            print("Anchor step: belief unchanged")
            A_r, _, Qpg = self._pi_R1(s_before, a_h_obs, self.state.belief)
            self._cached_A_r = A_r
            self._cached_Q_per_goal = Qpg
            return

        targets = set()
        for G in self.goals_abs:
            targets |= set(remaining_cells_list(s_before, G))

        legal = get_all_structurally_valid_actions(s_before, self.placeable_materials)
        A_h = filter_actions_to_targets(legal, targets)
        if a_h_obs not in A_h:
            A_h = [a_h_obs] + A_h

        logP_H2 = self._log_pi_H2(s_before, A_h)
        idx = A_h.index(a_h_obs)

        log_post = self.state.log_weights + logP_H2[:, idx]
        log_post -= np.log(np.exp(log_post).sum() + 1e-12)

        self.state = BeliefState(log_post, np.exp(log_post))

        print("Posterior belief:", self.state.belief)

        A_r, _, Qpg = self._pi_R1(s_before, a_h_obs, self.state.belief)
        self._cached_A_r = A_r
        self._cached_Q_per_goal = Qpg

        print("===========================================================")

    # ==========================================================
    # Robot action
    # ==========================================================

    def robot_act(self) -> Optional[Action]:
        if self._cached_A_r is None:
            return None

        scores = self.state.belief @ self._cached_Q_per_goal
        best = int(np.argmax(scores))

        print("\n==================== ROBOT ACT ====================")
        print("Belief:", self.state.belief)
        print("Action scores:", scores)
        print("Chosen action:", self._cached_A_r[best])
        print("===================================================")

        return self._cached_A_r[best]
