#!/usr/bin/env python3
"""
Goal templates + instantiation utilities for PPVA Tetris
"""

from typing import Set, Tuple, List

Cell = Tuple[int, int, int]
Goal = Set[Cell]

# G1: vertical L (anchor, right, up, up-up)
G1_OFFSETS: Set[Cell] = {(0,0,0),(1,0,0),(0,1,0),(0,2,0)}

# G2: plus centered at anchor (center, left, right, up)
G2_OFFSETS: Set[Cell] = {(0,0,0),(-1,0,0),(1,0,0),(0,1,0)}

# G3: horizontal-then-up L (anchor, +1, +2, +2-up)
G3_OFFSETS: Set[Cell] = {(0,0,0),(1,0,0),(2,0,0),(2,1,0)}

def translate_offsets(offsets: Set[Cell], anchor: Cell) -> Goal:
    ax, ay, az = anchor
    return {(ax+dx, ay+dy, az+dz) for (dx,dy,dz) in offsets}

def instantiate_three_goals(anchor: Cell) -> List[Goal]:
    return [
        translate_offsets(G1_OFFSETS, anchor),
        translate_offsets(G2_OFFSETS, anchor),
        translate_offsets(G3_OFFSETS, anchor),
    ]
