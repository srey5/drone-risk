import heapq
import math
from typing import Optional

import numpy as np


def weighted_astar(
    risk_map: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    risk_weight: float,
) -> Optional[list[tuple[int, int]]]:
    """Weighted A* over a 2D risk grid.

    cost = distance_to_neighbor + (risk_weight * neighbor_risk)
    Heuristic: octile (diagonal) distance.
    """
    rows, cols = risk_map.shape

    # 8-directional moves: (dy, dx, step_cost)
    moves = [
        (-1,  0, 1.0),
        ( 1,  0, 1.0),
        ( 0, -1, 1.0),
        ( 0,  1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1,  1, math.sqrt(2)),
        ( 1, -1, math.sqrt(2)),
        ( 1,  1, math.sqrt(2)),
    ]

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        dy = abs(a[0] - b[0])
        dx = abs(a[1] - b[1])
        return (dy + dx) + (math.sqrt(2) - 2) * min(dy, dx)

    g: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    heap: list[tuple[float, tuple[int, int]]] = [(heuristic(start, end), start)]

    while heap:
        _, current = heapq.heappop(heap)

        if current == end:
            path: list[tuple[int, int]] = []
            node = end
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return path[::-1]

        cy, cx = current
        for dy, dx, step_cost in moves:
            ny, nx = cy + dy, cx + dx
            if not (0 <= ny < rows and 0 <= nx < cols):
                continue
            neighbor = (ny, nx)
            tentative_g = g[current] + step_cost + risk_weight * float(risk_map[ny, nx])
            if tentative_g < g.get(neighbor, math.inf):
                g[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + heuristic(neighbor, end)
                heapq.heappush(heap, (f, neighbor))

    return None
