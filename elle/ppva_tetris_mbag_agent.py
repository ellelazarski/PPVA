#!/usr/bin/env python3
"""
PPVA Tetris MBAG wrapper

- Detect first human PLACE action (anchor)
- Instantiate goals from anchor
- Call PPVA belief_update on every human action
- Call robot_act once per step
"""

from typing import Optional, Tuple
import numpy as np
import logging

from mbag.agents.mbag_agent import MbagAgent
from mbag.environment.actions import MbagAction, MbagActionTuple
from mbag.environment.types import MbagObs, MbagInfoDict

from elle.baseline_tetris import BaselineTetris as PPVATetris, Action

# from elle.ppva_tetris import PPVATetris, Action
from elle.ppva_tetris_goals import instantiate_three_goals

logger = logging.getLogger(__name__)


class PPVATetrisMbagAgent(MbagAgent):

    def __init__(self, agent_config, env_config):
        super().__init__(agent_config, env_config)

        self.beta_R = agent_config["beta_R"]
        self.beta_H = agent_config["beta_H"]
        self.gamma  = agent_config["gamma"]

        # Controller is created AFTER anchor is observed
        self.controller: Optional[PPVATetris] = None

        # Anchor and state tracking
        self.anchor: Optional[Tuple[int, int, int]] = None
        self.obs_before_human: Optional[MbagObs] = None
        self.episode_step: int = 0

        logger.info("PPVA Tetris MBAG agent initialized")

    # ------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None) -> None:
        super().reset(seed=seed)

        self.controller = None
        self.anchor = None
        self.obs_before_human = None
        self.episode_step = 0

        logger.info("PPVA Tetris MBAG agent reset")

    # ------------------------------------------------------------

    def get_action_with_info(
        self,
        obs: MbagObs,
        info: Optional[MbagInfoDict],
    ) -> MbagActionTuple:

        self.episode_step += 1

        # --------------------------------------------------------
        # No human action -> robot NOOP
        # --------------------------------------------------------
        if not (info and "human_action" in info):
            self.obs_before_human = obs
            return (MbagAction.NOOP, 0, 0)

        action_type, block_idx, block_id = info["human_action"]

        if action_type not in (MbagAction.PLACE_BLOCK, MbagAction.BREAK_BLOCK):
            self.obs_before_human = obs
            return (MbagAction.NOOP, 0, 0)

        # Convert to (x,y,z)
        world_size = self.env_config["world_size"]
        x, y, z = np.unravel_index(block_idx, world_size)
        material = 0 if action_type == MbagAction.BREAK_BLOCK else block_id
        a_h_obs: Action = (int(x), int(y), int(z), int(material))

        # State before human acted
        s_before = self.obs_before_human if self.obs_before_human is not None else obs

        # --------------------------------------------------------
        # FIRST HUMAN PLACE -> ANCHOR
        # --------------------------------------------------------
        if self.anchor is None and action_type == MbagAction.PLACE_BLOCK:
            print("\n" + "=" * 70)
            print("ANCHOR OBSERVED (first human PLACE)")
            print(f"Anchor location: ({x}, {y}, {z})")
            print("=" * 70)

            self.anchor = (x, y, z)

            goals = instantiate_three_goals(self.anchor)

            self.controller = PPVATetris(
                placeable_materials=(6,),   # wood planks
                beta_R=self.beta_R,
                beta_H=self.beta_H,
                gamma=self.gamma,
                goals_abs=goals,
            )

            # Anchor step: no belief update but cache must be populated
            self.controller.belief_update(
                s_before=s_before,
                a_h_obs=a_h_obs,
                is_anchor=True,
            )

        else:
            # --------------------------------------------------------
            # NORMAL STEP (PPVA)
            # --------------------------------------------------------

            # Corner case: first human action is BREAK
            if self.controller is None:
                self.obs_before_human = obs
                return (MbagAction.NOOP, 0, 0)
            
            self.controller.belief_update(
                s_before=s_before,
                a_h_obs=a_h_obs,
                is_anchor=False,
            )

        # --------------------------------------------------------
        # Robot action
        # --------------------------------------------------------
        a_r = self.controller.robot_act()
        self.obs_before_human = obs

        if a_r is None:
            return (MbagAction.NOOP, 0, 0)

        rx, ry, rz, rmaterial = a_r
        ridx = int(np.ravel_multi_index((rx, ry, rz), world_size))

        return (MbagAction.PLACE_BLOCK, ridx, rmaterial)

    # ------------------------------------------------------------

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        return self.get_action_with_info(obs, None)
