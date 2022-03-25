from typing import Iterable, Optional, Tuple
import numpy as np

from StateManager import StateManager
from type_aliases import State
import config

class SimpleNIM(StateManager):
  max_pick = config.MAX_PICK

  def __init__(self, state:Optional[State] = None):
    self.reset(state)

  def __str__(self):
    return super().__str__() + '\nPieces left: '+str(self.num_pieces)

  @property
  def is_terminal_state(self) -> bool:
    return self.num_pieces == 0

  @property
  def players_turn(self):
    return self._players_turn

  @property
  def state(self) -> Tuple[int, int]:
    return [self._players_turn, self.num_pieces]

  def generate_child_state(self, action: int) -> State:
    return [not self._players_turn, self.num_pieces-action]

  def get_possible_actions(self) -> Iterable[int]:
    return range(1, min(self.max_pick, self.num_pieces)+1)

  def perform_action(self, action: int) -> State:
    self.num_pieces -= action
    self._players_turn = not self._players_turn

  def reset(self, state:Optional[State]) -> None:
    if state is None:
      self.num_pieces = config.NUM_PIECES
      self._players_turn = True
    else:
      self._players_turn, self.num_pieces = state

  def get_winner(self) -> int:
    if not self.is_terminal_state:
      raise Exception('Game is not finished yet')
    return 1 if self._players_turn else -1

  def visualize(self):
    print('''
    ------------------------
    Player {}'s turn
    Pieces left: {}
    '''.format(1 if self.players_turn else 2, self.num_pieces)
    )


  @classmethod
  def one_hot_encode(cls, state):
    return np.array([int(state[0]), *[1 if state[1] > i else 0 for i in range(config.NUM_PIECES)]])

  @classmethod
  def get_config(cls) -> str:
      return 'NP-{}_MP-{}'.format(config.NUM_PIECES, config.MAX_PICK)


if __name__ == '__main__':
  print(SimpleNIM.one_hot_encode([True, 0]))
  print(SimpleNIM.one_hot_encode([True, 5]))
  print(SimpleNIM.one_hot_encode([True, 6]))
  print(SimpleNIM.one_hot_encode([True, 8]))