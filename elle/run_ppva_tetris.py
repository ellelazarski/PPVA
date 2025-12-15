#!/usr/bin/env python3
"""
Script to run PPVA Tetris with human in Malmo Minecraft

Three goals (instantiated from first human block):
    G1: vertical L       offsets {(0,0,0),(1,0,0),(0,1,0),(0,2,0)}
    G2: plus at anchor   offsets {(0,0,0),(-1,0,0),(1,0,0),(0,1,0)}
    G3: horizontal-then-up L  offsets {(0,0,0),(1,0,0),(2,0,0),(2,1,0)}

Launch Malmo Minecraft:
python -m malmo.minecraft launch --num_instances 2 --goal_visibility False False

Run the script:
python elle/run_ppva_tetris.py
"""

import argparse
import logging
import time

from mbag.environment.types import WorldSize
from mbag.environment.mbag_env import MbagEnv
from mbag.environment.config import MbagConfigDict
from mbag.environment.goals import GoalGenerator
from mbag.agents.human_agent import HumanAgent
from mbag.environment.blocks import MinecraftBlocks

from elle.ppva_tetris_mbag_agent import PPVATetrisMbagAgent
from elle.ppva_tetris import remaining_cells_list

# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Utility: sync goal blocks to prevent red highlights
# --------------------------------------------------

def _sync_goal_with_current_in_malmo(env: MbagEnv):
    if not hasattr(env, "malmo_interface"):
        return

    width, height, depth = env.config["world_size"]
    goal_offset_x = width + 1

    clone_cmd = (
        f"chat /clone 0 0 0 {width-1} {height-1} {depth-1} "
        f"{goal_offset_x} 0 0"
    )

    try:
        env.malmo_interface._malmo_client.send_command(0, clone_cmd)
    except Exception:
        pass

# --------------------------------------------------
# Empty goal generator (Tetris uses internal goals)
# --------------------------------------------------

class EmptyGoalGenerator(GoalGenerator):
    def __init__(self, config):
        super().__init__(config)

    def generate_goal(self, size: WorldSize):
        return MinecraftBlocks(size)

# --------------------------------------------------
# Main
# --------------------------------------------------

def run_ppva_tetris_with_human(
    beta_R: float = 5.0,
    beta_H: float = 5.0,
    gamma: float = 0.2,
    record_video: bool = False,
    max_episode_len: int = 500,
    seed: int = 42,
):

    env_config: MbagConfigDict = {
        "num_players": 2,
        "horizon": max_episode_len,
        "world_size": (11, 10, 10),
        "random_start_locations": False,
        "randomize_first_episode_length": False,
        "terminate_on_goal_completion": False,

        "goal_generator": EmptyGoalGenerator,
        "goal_generator_config": {
            "world_size": (11, 10, 10),
        },

        "players": [
            {
                "player_name": "Human",
                "is_human": True,
                "goal_visible": False,
                "timestep_skip": 1,
                "give_items": [],
            },
            {
                "player_name": "PPVA_Tetris_Robot",
                "is_human": False,
                "goal_visible": False,
                "timestep_skip": 1,
                "give_items": [],
            },
        ],

        "rewards": {k: 0.0 for k in [
            "noop", "action", "incorrect_action", "place_wrong",
            "get_resources", "own_reward_prop"
        ]},

        "abilities": {
            "teleportation": True,
            "flying": True,
            "inf_blocks": True,
        },

        "malmo": {
            "use_malmo": True,
            "use_spectator": record_video,
            "rotate_spectator": record_video,
            "restrict_players": True,
            "start_port": 10000,
            "action_delay": 3.0,
            "video_dir": "videos/tetris" if record_video else None,
        },
    }

    print("\n" + "=" * 70)
    print("PPVA TETRIS (THREE GOALS)")
    print("=" * 70)
    print("G1: vertical L")
    print("G2: plus at anchor")
    print("G3: horizontal-then-up L")
    print("=" * 70 + "\n")

    env = MbagEnv(env_config)

    human_agent = HumanAgent({}, env.config)
    tetris_agent = PPVATetrisMbagAgent(
        {"beta_R": beta_R, "beta_H": beta_H, "gamma": gamma},
        env.config,
    )
    agents = [human_agent, tetris_agent]

    for agent in agents:
        agent.reset(seed=seed)

    all_obs, all_infos = env.reset()
    prev_infos = all_infos
    timestep = 0

    # Sync goal blocks
    env.goal_blocks.blocks[:] = env.current_blocks.blocks
    env.goal_blocks.block_states[:] = env.current_blocks.block_states

    print("Episode started. Place your first block.\n")

    while timestep < max_episode_len:
        actions = []

        for i, agent in enumerate(agents):
            obs = all_obs[i]
            info = prev_infos[0] if i == 1 else prev_infos[i]
            actions.append(agent.get_action_with_info(obs, info))

        all_obs, rewards, done, infos = env.step(actions)
        prev_infos = infos
        timestep += 1

        # Keep Malmo visuals clean
        env.goal_blocks.blocks[:] = env.current_blocks.blocks
        env.goal_blocks.block_states[:] = env.current_blocks.block_states
        _sync_goal_with_current_in_malmo(env)

        # --------------------------------------------------
        # Check completion of any goal (G1/G2/G3)
        # --------------------------------------------------
        ctrl = tetris_agent.controller
        if ctrl is not None and ctrl.goals_abs is not None:
            obs_human = all_obs[0]

            rem = [
                len(remaining_cells_list(obs_human, ctrl.goals_abs[k]))
                for k in range(len(ctrl.goals_abs))
            ]

            if any(r == 0 for r in rem):
                print("\n" + "=" * 70)
                print("GOAL COMPLETED!")
                for k, r in enumerate(rem, start=1):
                    if r == 0:
                        print(f"G{k} COMPLETE")
                print("=" * 70)

                if record_video:
                    time.sleep(2)
                break

    if env.config["malmo"]["use_malmo"]:
        env.malmo_interface.end_episode()

    print("\n" + "=" * 70)
    print("Episode Complete")
    print("=" * 70)
    belief = tetris_agent.controller.state.belief
    print(f"Final belief: G1={belief[0]:.3f}, G2={belief[1]:.3f}, G3={belief[2]:.3f}")
    print("=" * 70 + "\n")

# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Run PPVA Tetris with Malmo")
    parser.add_argument("--beta_R", type=float, default=5.0)
    parser.add_argument("--beta_H", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--max_episode_len", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    run_ppva_tetris_with_human(
        beta_R=args.beta_R,
        beta_H=args.beta_H,
        gamma=args.gamma,
        record_video=args.record_video,
        max_episode_len=args.max_episode_len,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
