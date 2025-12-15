#!/usr/bin/env python3
"""
ppva_structural_validity.py

These checks are independent of player position/reachability. They only check
whether a block could physically be placed/broken at a location, ignoring
whether any player can actually reach it.

For counterfactual reasoning (H1 likelihood), we want to consider any edit
the human could have made structurally, not just edits reachable from their
current position. This is because we're asking: "Why did the human choose THIS
edit instead of THAT edit?" where "THAT" can be any structurally valid edit.

Uses MinecraftBlocks class to ensure consistency with environment.
Extracts only the structural constraints from blocks.py:

PLACE constraints (from try_break_place + _get_viewpoint_click_candidates):
  1. Target location must be AIR
  2. Material must be placeable
  3. Must have adjacent solid block for support

BREAK constraints (from try_break_place):
  1. Target must be non-AIR
  2. Target cannot be BEDROCK

Excluded checks:
  - Player position/reachability checks (from _get_viewpoint_click_candidates)
  - Line of sight checks (from action masking)
  - Inventory checks (inf_blocks=True by default anyway)
"""

import numpy as np
from typing import List, Tuple, cast
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.actions import MbagAction
from mbag.environment.types import CURRENT_BLOCKS, BlockLocation

# Type aliases
Action = Tuple[int, int, int, int]  # (x, y, z, material)
MbagObs = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _obs_to_blocks(obs: MbagObs) -> MinecraftBlocks:
    """Convert observation to MinecraftBlocks object."""
    world_obs, _, _ = obs
    world_grid = world_obs[CURRENT_BLOCKS]
    
    blocks_obj = MinecraftBlocks(world_grid.shape)
    blocks_obj.blocks[:] = world_grid
    return blocks_obj


def _has_adjacent_solid_block(
    blocks_obj: MinecraftBlocks,
    location: BlockLocation
) -> bool:
    """
    Check if location has at least one adjacent solid block.
    
    This replicates the logic from MinecraftBlocks._get_viewpoint_click_candidates
    but without the reachability check.
    """
    x, y, z = location
    
    # Check all 6 faces for adjacent solid blocks
    for face_dim in range(3):
        for direction in [-1, 1]:
            against_block_arr = np.array(location)
            against_block_arr[face_dim] += direction
            against_block_loc: BlockLocation = cast(
                BlockLocation, tuple(against_block_arr.astype(int))
            )
            
            # Check if valid location and has solid block
            if (blocks_obj.is_valid_block_location(against_block_loc) and
                blocks_obj.blocks[against_block_loc] in MinecraftBlocks.SOLID_BLOCK_IDS):
                return True
    
    return False


def is_structurally_valid_place(
    blocks_obj: MinecraftBlocks,
    location: BlockLocation,
    material: int
) -> bool:
    """
    Check if a PLACE action is structurally valid (ignoring player position).
    
    Uses the same rules as MinecraftBlocks.try_break_place
    plus adjacent solid block check.
    
    Args:
        blocks_obj: MinecraftBlocks object
        location: Target location (x, y, z)
        material: Material ID to place
        
    Returns:
        True if structurally valid
    """
    # Rule 1: Target must be AIR (from try_break_place line 414)
    if blocks_obj.blocks[location] != MinecraftBlocks.AIR:
        return False
    
    # Rule 2: Material must be placeable (from try_break_place line 417)
    if material not in MinecraftBlocks.PLACEABLE_BLOCK_IDS:
        return False
    
    # Rule 3: Must have adjacent solid block (from _get_viewpoint_click_candidates lines 223-234)
    if not _has_adjacent_solid_block(blocks_obj, location):
        return False
    
    return True


def is_structurally_valid_break(
    blocks_obj: MinecraftBlocks,
    location: BlockLocation
) -> bool:
    """
    Check if a BREAK action is structurally valid (ignoring player position).
    
    Uses the same rules as MinecraftBlocks.try_break_place.
    
    Args:
        blocks_obj: MinecraftBlocks object
        location: Target location (x, y, z)
        
    Returns:
        True if structurally valid
    """
    block = blocks_obj.blocks[location]
    
    # Rules from try_break_place lines 421-426
    if block in [MinecraftBlocks.AIR, MinecraftBlocks.BEDROCK]:
        return False
    
    return True


def get_all_structurally_valid_actions(
    obs: MbagObs,
    placeable_materials: Tuple[int, ...],
) -> List[Action]:
    """
    Get ALL structurally valid place/break actions, ignoring player position.
    
    This is the function to use for PPVA counterfactual reasoning, where we want
    to consider any edit the human could have made structurally, not just what's
    reachable from their current position.
    
    Args:
        obs: (world_obs, inventory, timestep)
        placeable_materials: Tuple of material IDs to consider for placement
        
    Returns:
        List of structurally valid actions (x, y, z, material)
    """
    # Convert observation to MinecraftBlocks object
    blocks_obj = _obs_to_blocks(obs)
    X, Y, Z = blocks_obj.size
    
    valid_actions: List[Action] = []
    
    # Generate all place actions
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                location: BlockLocation = (x, y, z)
                for material in placeable_materials:
                    if is_structurally_valid_place(blocks_obj, location, material):
                        valid_actions.append((x, y, z, int(material)))
    
    # Generate all break actions
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                location: BlockLocation = (x, y, z)
                if is_structurally_valid_break(blocks_obj, location):
                    valid_actions.append((x, y, z, 0))  # material=0 for break
    
    return valid_actions


def filter_structurally_valid_actions(
    obs: MbagObs,
    actions: List[Action],
) -> List[Action]:
    """
    Filter a list of actions to only those that are structurally valid.
    
    Args:
        obs: (world_obs, inventory, timestep)
        actions: List of actions to filter
        
    Returns:
        Filtered list of structurally valid actions
    """
    # Convert observation to MinecraftBlocks object
    blocks_obj = _obs_to_blocks(obs)
    
    valid_actions: List[Action] = []
    
    for action in actions:
        x, y, z, m = action
        location: BlockLocation = (x, y, z)
        
        if m == 0:  # Break action
            if is_structurally_valid_break(blocks_obj, location):
                valid_actions.append(action)
        else:  # Place action
            if is_structurally_valid_place(blocks_obj, location, m):
                valid_actions.append(action)
    
    return valid_actions
