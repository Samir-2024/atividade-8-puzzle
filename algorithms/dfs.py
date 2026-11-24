from puzzle.base_search import BaseSearch
from puzzle.state import State
from puzzle.result import SearchResult

DEFAULT_DEPTH_LIMIT = 50


class DFS(BaseSearch):

    def __init__(self, depth_limit: int = DEFAULT_DEPTH_LIMIT):
        self.depth_limit = depth_limit

    def search(self, initial: State) -> SearchResult:
        if initial.is_goal:
            return SearchResult(
                solution=initial,
                nodes_expanded=0,
                nodes_generated=1,
                max_frontier_size=1,
                depth=0,
            )

        stack = [initial]
        frontier_set = {initial.tiles}
        explored = set()

        nodes_expanded = 0
        nodes_generated = 1
        max_frontier_size = 1

        while stack:
            state = stack.pop()
            frontier_set.discard(state.tiles)

            if state.is_goal:
                return SearchResult(
                    solution=state,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    max_frontier_size=max_frontier_size,
                    depth=state.cost,
                )

            explored.add(state.tiles)

            if state.cost < self.depth_limit:
                nodes_expanded += 1
                for child in reversed(state.neighbors()):
                    nodes_generated += 1
                    if child.tiles not in explored and child.tiles not in frontier_set:
                        frontier_set.add(child.tiles)
                        stack.append(child)
                        if len(stack) > max_frontier_size:
                            max_frontier_size = len(stack)

        return SearchResult(
            solution=None,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier_size,
            depth=0,
        )
