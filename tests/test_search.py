import unittest
from puzzle.state import State, GOAL_STATE
from puzzle.result import SearchResult
from algorithms.bfs import BFS
from algorithms.dfs import DFS
from algorithms.a_star import AStar


class TestState(unittest.TestCase):

    def test_init_valid(self):
        s = State((1, 2, 3, 4, 5, 6, 7, 8, 0))
        self.assertEqual(s.tiles, GOAL_STATE)
        self.assertIsNone(s.parent)
        self.assertIsNone(s.action)
        self.assertEqual(s.cost, 0)

    def test_init_invalid(self):
        with self.assertRaises(ValueError):
            State((1, 2, 3, 4, 5, 6, 7, 8, 9))
        with self.assertRaises(ValueError):
            State((1, 2, 3, 4, 5, 6, 7, 8))
        with self.assertRaises(ValueError):
            State((0, 0, 1, 2, 3, 4, 5, 6, 7))

    def test_is_goal(self):
        self.assertTrue(State(GOAL_STATE).is_goal)
        self.assertFalse(State((1, 2, 3, 4, 5, 6, 7, 0, 8)).is_goal)

    def test_blank_index(self):
        s = State((1, 2, 0, 3, 4, 5, 6, 7, 8))
        self.assertEqual(s.blank_index, 2)

    def test_neighbors_corner(self):
        # Blank at index 0 (row 0, col 0) -> only DOWN and RIGHT possible
        s = State((0, 1, 2, 3, 4, 5, 6, 7, 8))
        nbrs = s.neighbors()
        self.assertEqual(len(nbrs), 2)
        actions = [n.action for n in nbrs]
        self.assertIn("DOWN", actions)
        self.assertIn("RIGHT", actions)
        for n in nbrs:
            self.assertIs(n.parent, s)
            self.assertEqual(n.cost, 1)

    def test_neighbors_center(self):
        # Blank at index 4 (row 1, col 1) -> UP, DOWN, LEFT, RIGHT all possible
        s = State((1, 2, 3, 4, 0, 5, 6, 7, 8))
        nbrs = s.neighbors()
        self.assertEqual(len(nbrs), 4)
        actions = [n.action for n in nbrs]
        self.assertEqual(actions, ["UP", "DOWN", "LEFT", "RIGHT"])

    def test_neighbors_edge(self):
        # Blank at index 1 (row 0, col 1) -> DOWN, LEFT, RIGHT possible
        s = State((1, 0, 2, 3, 4, 5, 6, 7, 8))
        nbrs = s.neighbors()
        self.assertEqual(len(nbrs), 3)
        actions = [n.action for n in nbrs]
        self.assertEqual(actions, ["DOWN", "LEFT", "RIGHT"])

    def test_path_and_actions(self):
        initial = State((1, 2, 3, 4, 5, 6, 7, 8, 0))
        self.assertEqual(initial.path(), [initial])
        self.assertEqual(initial.actions(), [])

        nbrs = initial.neighbors()
        first_child = nbrs[0]
        self.assertEqual(first_child.path(), [initial, first_child])
        self.assertEqual(first_child.actions(), [first_child.action])


class TestBFS(unittest.TestCase):

    def setUp(self):
        self.bfs = BFS()

    def test_already_goal(self):
        goal = State(GOAL_STATE)
        res = self.bfs.search(goal)
        self.assertTrue(res.found)
        self.assertEqual(res.depth, 0)
        self.assertEqual(res.path_cost, 0)
        self.assertEqual(res.actions, [])
        self.assertEqual(res.nodes_expanded, 0)
        self.assertEqual(res.nodes_generated, 1)

    def test_one_step(self):
        # Goal with blank moved UP (to index 5)
        # Goal: (1, 2, 3, 4, 5, 6, 7, 8, 0)
        # Up neighbor of goal: (1, 2, 3, 4, 5, 0, 7, 8, 6)
        s1 = State((1, 2, 3, 4, 5, 0, 7, 8, 6))
        res = self.bfs.search(s1)
        self.assertTrue(res.found)
        self.assertEqual(res.depth, 1)
        self.assertEqual(res.path_cost, 1)
        self.assertEqual(res.actions, ["DOWN"])
        self.assertEqual(res.solution.tiles, GOAL_STATE)

    def test_three_steps(self):
        # Blank moved from index 8 -> 7 -> 4 -> 3
        # State: (1, 2, 3, 0, 4, 6, 7, 5, 8)
        s3 = State((1, 2, 3, 0, 4, 6, 7, 5, 8))
        res = self.bfs.search(s3)
        self.assertTrue(res.found)
        self.assertEqual(res.depth, 3)
        self.assertEqual(res.path_cost, 3)
        self.assertEqual(res.actions, ["RIGHT", "DOWN", "RIGHT"])

    def test_unsolvable(self):
        # Inverted parity state
        s_unsolvable = State((2, 1, 3, 4, 5, 6, 7, 8, 0))
        # Use small state space check or known unsolvable
        # (1, 2, 3, 4, 5, 6, 8, 7, 0) has 1 inversion, odd, unsolvable
        s_odd = State((1, 2, 3, 4, 5, 6, 8, 7, 0))
        res = self.bfs.search(s_odd)
        self.assertFalse(res.found)
        self.assertIsNone(res.solution)
        self.assertEqual(res.actions, [])
        self.assertEqual(res.path_cost, -1)


class TestDFS(unittest.TestCase):

    def setUp(self):
        self.dfs = DFS(depth_limit=50)

    def test_already_goal(self):
        goal = State(GOAL_STATE)
        res = self.dfs.search(goal)
        self.assertTrue(res.found)
        self.assertEqual(res.depth, 0)
        self.assertEqual(res.actions, [])

    def test_one_step(self):
        s1 = State((1, 2, 3, 4, 5, 0, 7, 8, 6))
        res = self.dfs.search(s1)
        self.assertTrue(res.found)
        self.assertEqual(res.actions, ["DOWN"])

    def test_solvable_within_limit(self):
        s = State((1, 2, 3, 0, 4, 6, 7, 5, 8))
        res = self.dfs.search(s)
        self.assertTrue(res.found)
        self.assertLessEqual(res.depth, 50)
        self.assertEqual(res.solution.tiles, GOAL_STATE)

    def test_depth_limit_cutoff(self):
        # A state 3 steps away cannot be reached with depth_limit=2
        s3 = State((1, 2, 3, 0, 4, 6, 7, 5, 8))
        res = DFS(depth_limit=2).search(s3)
        self.assertFalse(res.found)


class TestAStar(unittest.TestCase):

    def setUp(self):
        self.astar = AStar()

    def test_heuristic(self):
        goal = State(GOAL_STATE)
        self.assertEqual(self.astar.heuristic(goal), 0)

        # Move 6 up: tile 6 at index 8 (dist 0), blank at 5
        # (1, 2, 3, 4, 5, 0, 7, 8, 6):
        # 6 is at index 8 (row 2, col 2), goal of 6 is index 5 (row 1, col 2) -> dist = 1
        s1 = State((1, 2, 3, 4, 5, 0, 7, 8, 6))
        self.assertEqual(self.astar.heuristic(s1), 1)

    def test_custom_heuristic(self):
        # Custom heuristic that always returns 0 (behaves like Dijkstra)
        custom_astar = AStar(heuristic=lambda s: 0)
        s1 = State((1, 2, 3, 4, 5, 0, 7, 8, 6))
        res = custom_astar.search(s1)
        self.assertTrue(res.found)
        self.assertEqual(res.depth, 1)

    def test_already_goal(self):
        goal = State(GOAL_STATE)
        res = self.astar.search(goal)
        self.assertTrue(res.found)
        self.assertEqual(res.depth, 0)
        self.assertEqual(res.path_cost, 0)
        self.assertEqual(res.actions, [])
        self.assertEqual(res.nodes_expanded, 0)

    def test_optimality_matches_bfs(self):
        s = State((1, 2, 3, 0, 4, 6, 7, 5, 8))
        bfs_res = BFS().search(s)
        astar_res = self.astar.search(s)
        self.assertTrue(astar_res.found)
        self.assertEqual(astar_res.depth, bfs_res.depth)
        self.assertEqual(astar_res.path_cost, bfs_res.path_cost)
        self.assertLessEqual(astar_res.nodes_expanded, bfs_res.nodes_expanded)

    def test_solution_actions_validity(self):
        # Test that applying returned actions actually produces the goal state
        s = State((1, 2, 3, 0, 4, 6, 7, 5, 8))
        for searcher in [BFS(), DFS(depth_limit=50), AStar()]:
            res = searcher.search(s)
            self.assertTrue(res.found)
            curr = s
            for action in res.actions:
                nbrs = {n.action: n for n in curr.neighbors()}
                self.assertIn(action, nbrs)
                curr = nbrs[action]
            self.assertEqual(curr.tiles, GOAL_STATE)


if __name__ == "__main__":
    unittest.main()
