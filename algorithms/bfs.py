from collections import deque
from puzzle.base_search import BaseSearch
from puzzle.state import State
from puzzle.result import SearchResult


class BFS(BaseSearch):

    def search(self, initial: State) -> SearchResult:
        if initial.is_goal:
            return SearchResult(
                solution=initial,
                nodes_expanded=0,
                nodes_generated=1,
                max_frontier_size=1,
                depth=0,
            )

        frontier = deque([initial])
        frontier_set = {initial.tiles}
        explored = set()

        nodes_expanded = 0
        nodes_generated = 1
        max_frontier_size = 1

        while frontier:
            state = frontier.popleft()
            frontier_set.remove(state.tiles)
            explored.add(state.tiles)
            nodes_expanded += 1

            for child in state.neighbors():
                nodes_generated += 1
                if child.tiles not in explored and child.tiles not in frontier_set:
                    if child.is_goal:
                        return SearchResult(
                            solution=child,
                            nodes_expanded=nodes_expanded,
                            nodes_generated=nodes_generated,
                            max_frontier_size=max(max_frontier_size, len(frontier)),
                            depth=child.cost,
                        )
                    frontier.append(child)
                    frontier_set.add(child.tiles)
                    if len(frontier) > max_frontier_size:
                        max_frontier_size = len(frontier)

        return SearchResult(
            solution=None,
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            max_frontier_size=max_frontier_size,
            depth=0,
        )
