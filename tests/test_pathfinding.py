import numpy as np
import pytest

from core.engine.pathfinding import weighted_astar

def test_astar_norisk():
    grid = np.zeros((5, 5), dtype=np.float32)
    start = (2, 0)
    end = (2, 4)

    # With 0 risk weight, it should take a straight horizontal line
    path = weighted_astar(grid, start, end, risk_weight=0.0)

    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == end

    # Ensure it didn't deviate from row 2
    for r, c in path:
        assert r == 2

def test_astar_highrisk():
    grid = np.zeros((5, 5), dtype=np.float32)

    # Create a high-risk wall in the middle column (rows 1-3; rows 0 and 4 are clear)
    grid[1:4, 2] = 1.0

    start = (2, 0)
    end = (2, 4)

    # With a high risk weight, it MUST route around the wall via row 0 or row 4
    path = weighted_astar(grid, start, end, risk_weight=100.0)

    wall_coordinates = [(1, 2), (2, 2), (3, 2)]

    for coord in path:
        assert coord not in wall_coordinates
