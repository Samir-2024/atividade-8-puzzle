from typing import Callable, Optional
import heapq
import itertools
from puzzle.base_search import BaseSearch
from puzzle.state import State
from puzzle.result import SearchResult


class AStar(BaseSearch):

    def __init__(self, heuristic: Optional[Callable[[State], int]] = None):
        self._custom_heuristic = heuristic

    def heuristic(self, state: State) -> int:
        """Função heurística admissível: distância de Manhattan das peças até suas posições finais."""
        if self._custom_heuristic is not None:
            return self._custom_heuristic(state)

        distance = 0
        for idx, val in enumerate(state.tiles):
            if val != 0:
                goal_row = (val - 1) // 3
                goal_col = (val - 1) % 3
                curr_row = idx // 3
                curr_col = idx % 3
                distance += abs(curr_row - goal_row) + abs(curr_col - goal_col)
        return distance

    def search(self, initial: State) -> SearchResult:
        if initial.is_goal:
            return SearchResult(
                solution=initial,
                nodes_expanded=0,
                nodes_generated=1,
                max_frontier_size=1,
                depth=0,
            )

        counter = itertools.count()
        frontier = []
        initial_f = initial.cost + self.heuristic(initial)
        heapq.heappush(frontier, (initial_f, next(counter), initial))

        cost_so_far = {initial.tiles: initial.cost}
        explored = set()

        nodes_expanded = 0
        nodes_generated = 1
        max_frontier_size = 1

        while frontier:
            f, _, state = heapq.heappop(frontier)

            if state.tiles in explored:
                continue
            explored.add(state.tiles)

            if state.is_goal:
                return SearchResult(
                    solution=state,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    max_frontier_size=max_frontier_size,
                    depth=state.cost,
                )

            nodes_expanded += 1

            for child in state.neighbors():
                nodes_generated += 1
                if child.tiles in explored:
                    continue

                if child.tiles not in cost_so_far or child.cost < cost_so_far[child.tiles]:
                    cost_so_far[child.tiles] = child.cost
                    child_f = child.cost + self.heuristic(child)
                    heapq.heappush(frontier, (child_f, next(counter), child))
                    if len(frontier) > max_frontier_size:
                        max_frontier_size = len(frontier)

        return SearchResult(
            solution=None,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier_size,
            depth=0,
        )
