#!/usr/bin/env python3
"""
ppva_action_simulator.py

Simulates the effect of actions on observations without executing in the environment.
"""

import numpy as np
from typing import Tuple
from mbag.environment.actions import MbagAction, MbagActionTuple
from mbag.environment.types import (
    MbagObs,
    CURRENT_BLOCKS,
    PLAYER_LOCATIONS,
    CURRENT_PLAYER,
    OTHER_PLAYER,
    NO_ONE,
    WorldSize,
)


def simulate_full_action(
    obs: MbagObs,
    action_tuple: MbagActionTuple,
    world_size: WorldSize,
    agent_id: int = 0,
) -> MbagObs:
    """
    Simulate an action's effect on observation.
    
    Simplified simulation for oracle evaluation.
    It updates the key observable state (blocks, player positions, inventory)
    but doesn't handle complex dynamics (physics, collisions, etc.).
    
    The action is assumed to be valid (e.g., from action masking).
    This function does NOT validate:
    - Player overlap prevention (masking already prevents this)
    - Block collision checking (masking handles this)
    - Inventory constraints (inf_blocks=True by default)
    
    Args:
        obs: (world_obs, inventory, timestep)
        action_tuple: (action_type, block_location_idx, block_id)
        world_size: (X, Y, Z) world dimensions
        agent_id: Which agent is acting (0 or 1 for hivemind)
        
    Returns:
        Updated observation tuple
    """
    world_obs, inventory, timestep = obs
    action_type, block_location_idx, block_id = action_tuple
    
    # Copy to avoid modifying original
    world_obs = world_obs.copy()
    inventory = inventory.copy()
    
    # Convert flat location index to 3D coordinates
    location = np.unravel_index(block_location_idx, world_size)
    
    if action_type == MbagAction.PLACE_BLOCK:
        # Update blocks grid
        world_obs[CURRENT_BLOCKS][location] = block_id
        
    elif action_type == MbagAction.BREAK_BLOCK:
        # Update blocks grid (set to AIR)
        world_obs[CURRENT_BLOCKS][location] = 0
        
    elif action_type in MbagAction.MOVE_ACTION_TYPES:
        # Update player position
        # NOTE: Players occupy 2 cells (feet and head at y and y+1)
        delta = MbagAction.MOVE_ACTION_DELTAS[action_type]
        
        # Find current player position (feet)
        player_locations = world_obs[PLAYER_LOCATIONS]
        current_player_mask = (player_locations == CURRENT_PLAYER)
        
        if np.any(current_player_mask):
            # Get all cells occupied by player
            coords = np.argwhere(current_player_mask)
            if len(coords) > 0:
                # Find feet position (lowest y)
                feet_idx = np.argmin(coords[:, 1])  # y (height) is index 1 in world array indices [width, height, depth]
                old_x, old_y_feet, old_z = coords[feet_idx]
                
                # Compute new position
                new_x = old_x + delta[0]
                new_y_feet = old_y_feet + delta[1]
                new_z = old_z + delta[2]
                
                # Check bounds for both feet and head
                if (0 <= new_x < world_size[0] and 
                    0 <= new_y_feet < world_size[1] and
                    0 <= new_z < world_size[2]):
                    
                    # Clear old position (feet and head)
                    world_obs[PLAYER_LOCATIONS][old_x, old_y_feet, old_z] = NO_ONE
                    if old_y_feet + 1 < world_size[1]:
                        world_obs[PLAYER_LOCATIONS][old_x, old_y_feet + 1, old_z] = NO_ONE
                    
                    # Set new position (feet and head)
                    world_obs[PLAYER_LOCATIONS][new_x, new_y_feet, new_z] = CURRENT_PLAYER
                    if new_y_feet + 1 < world_size[1]:
                        world_obs[PLAYER_LOCATIONS][new_x, new_y_feet + 1, new_z] = CURRENT_PLAYER

    elif action_type == MbagAction.NOOP:
        # No changes
        pass
    
    return (world_obs, inventory, timestep.copy())


def simulate_block_edit(
    obs: MbagObs,
    action: Tuple[int, int, int, int],
) -> MbagObs:
    """
    Simulate PPVA-format action (x, y, z, material).
    
    Used for PPVA counterfactual reasoning where we only consider
    block placement/breaking actions, not movement or other action types.
    
    Args:
        obs: (world_obs, inventory, timestep)
        action: (x, y, z, material) where material=0 means break
        
    Returns:
        Updated observation tuple
    """
    world_obs, inventory, timestep = obs
    world_obs = world_obs.copy()
    
    x, y, z, material = action
    
    # Direct update (PPVA only does place/break)
    world_obs[CURRENT_BLOCKS, x, y, z] = material
    
    return (world_obs, inventory.copy(), timestep.copy())
