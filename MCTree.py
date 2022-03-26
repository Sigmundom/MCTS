from math import log, sqrt
from random import choice, shuffle
from typing import Callable, Dict, Optional

from StateManager import State, StateManager
from type_aliases import Action, PossibleActions


class MCNode:
    def __init__(self, state: State, parent: Optional['MCNode'] = None):
        self.parent = parent
        self.children: Dict[int, MCNode] = {}

        self.state = state
        self._value = 0
        self._visit_count = 0

    def __str__(self) -> str:
        return '''
      State: {}
      Value: {}
      Visit count: {}
      Players turn: {}
      '''.format(self.state, self._value, self._visit_count, self._players_turn, )

    @property
    def _players_turn(self):
        '''Returns True if it's player 1's turn. False if it's player 2.'''
        return self.state[0]

    @property
    def exploration_bonus(self):
        '''Exploration bonus using Upper Confidence Bounds for Trees (UCT) as a metric'''
        return 2 * sqrt(log(self.parent._visit_count) / (1 + self._visit_count))

    @property
    def expected_final_result(self):
        '''Simple evaluation of expected final result: Percentage of won games from this state.'''
        return 0 if self._visit_count == 0 else self._value/self._visit_count

    @property
    def is_leaf(self):
        '''Returns True if the node has no children.'''
        return len(self.children) == 0

    @property
    def visit_count(self):
        return self._visit_count

    def tree_policy(self) -> int:
        '''Gives the greedy best action choice, with exploration bonus taken into account.'''
        shuffled_children = list(self.children.items())
        shuffle(shuffled_children)
        if self._players_turn:
            best_action = max(
                shuffled_children, key=lambda x: x[1].expected_final_result + x[1].exploration_bonus)[0]
        else:
            best_action = min(
                shuffled_children, key=lambda x: x[1].expected_final_result - x[1].exploration_bonus)[0]
        return best_action

    def add_child(self, child_state: State, action: int) -> None:
        '''Adds a child node with the given state and the action as the key.'''
        self.children[action] = MCNode(child_state, self)

    def update_node_statistics(self, reward: int):
        self._visit_count += 1
        self._value += reward
        # if self._players_turn:
        # else:
        #   self._value -= reward


class MCTree:
    def __init__(self, initial_state: State):
        self.root: MCNode = MCNode(initial_state)

    def __str__(self):
        def recursive_stringify(action, node, depth):
            s = ''
            for a, n in node.children.items():
                s += recursive_stringify(a, n, depth+1)
            return str(depth) + '.' + str(action) + ': ' + str(node) + s
        s = recursive_stringify(0, self.root, 0)
        return s

    def simulate(self, board: StateManager, default_policy: Callable[[State, PossibleActions], Action]):
        # – Use tree policy Pt to search from root to a leaf (L) of MCT. Update Bmc with each move.
        current_node = self.root

        while not current_node.is_leaf:
            next_action = current_node.tree_policy()
            current_node = current_node.children[next_action]
            board.perform_action(next_action)

        if not board.is_terminal_state:
            # Expand leaf node
            possible_actions = board.get_possible_actions()
            for action in possible_actions:
                child_state = board.generate_child_state(action)
                current_node.add_child(child_state, action)

            # Select and go to one of the new leaf nodes using ANET.
            next_action = default_policy(current_node.state, possible_actions)
            current_node = current_node.children[next_action]
            board.perform_action(next_action)

        # – Use ANET to choose rollout actions from L+1 to a final state (F). Update Bmc with each move.
        while not board.is_terminal_state:
            # Choose action
            possible_actions = board.get_possible_actions()
            next_action = default_policy(current_node.state, possible_actions)
            board.perform_action(next_action)

        # – Perform MCTS backpropagation from F to root.
        reward = board.get_winner()
        assert reward == 1 if board.state[0] else -1
        while current_node is not None:
            current_node.update_node_statistics(reward)
            current_node = current_node.parent

    def update_root(self, action: int) -> None:
        '''Select child based on action as the new root. Discard all other branches.'''
        self.root = self.root.children[action]
        self.root.parent = None
