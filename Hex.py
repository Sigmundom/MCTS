from math import cos, pi, sin
from typing import Iterable, Optional
import numpy as np
import matplotlib.pyplot as plt

from StateManager import StateManager
from type_aliases import State
import config

class Cell:
  def __init__(self, r: int, c: int, owner: Optional[int] = 0):
    self.r = r
    self.c = c
    self.owner = owner
    self.group = [self]

  def set_owner(self, owner:int):
    assert self.owner == 0
    self.owner = owner
    return self.group

  def merge_groups(self, new_group):
    new_group.extend(self.group) # Merges the two groups
    for member in self.group:
      member.group = new_group # Sets the new group membership for all member in old group

def get_neighbours(r, c, dim):
  potensial_neighbours = [(r-1, c), (r-1, c+1), (r, c-1), (r, c+1), (r+1, c-1), (r+1, c)]
  return list(filter(lambda x: 0 <= x[0] < dim and 0 <= x[1] < dim, potensial_neighbours))


class Hex(StateManager):
  def __init__(self, state:Optional[State] = None, dims=None):
    self._dims = dims if dims is not None else config.HEX_BOARD_DIMS
    self.reset(state)

  @property
  def players_turn(self):
    return self._players_turn

  @property
  def state(self) -> State:
    return [int(self.players_turn), *[cell.owner for cell in self.board.flatten()]]

  def __str__(self) -> str:
    return str(self.state)


  def generate_child_state(self, action: int) -> State:
    child_state = self.state
    child_state[action+1] = 1 if self.players_turn else 2
    return child_state

  def get_possible_actions(self) -> Iterable[int]:
    board = self.state[1:]
    return [i for i in range(len(board)) if board[i] == 0]

  def get_winner(self) -> int:
    return self._winner


  def is_terminal_state(self) -> bool:
    # Only necessery to check if the player who had the previous turn wins
    if self.players_turn:
      left_side = self.board[:, 0]
      for cell in left_side:
        # Opponent wins if cell on the left side is in same group with a cell on the right side
        if max(cell.group, key=lambda cell: cell.r).r == self._dims[0]-1:
          self._winner = -1
          return True
    else:
      top_side = self.board[0, :]
      for cell in top_side:
        print([(c.r, c.c) for c in cell.group])
        if max(cell.group, key=lambda cell: cell.c).c == self._dims[0] - 1:
          self._winner = 1
          return True
    return False

  def perform_action(self, action: int):
    r = action % self._dims[0]
    c = action // self._dims[0]
    cell = self.board[r, c]
    owner = 1 if self.players_turn else 2
    new_group = cell.set_owner(owner)
    for neighbour_idx in get_neighbours(r,c, self._dims[0]):
      neighbour = self.board[neighbour_idx]
      if neighbour.owner == owner and neighbour.group != new_group:
        neighbour.merge_groups(new_group)

    self._players_turn = not self._players_turn

  def reset(self, state: Optional[State] = None) -> None:
    # if state is not None:
    #   assert len(state) == self.board.size
    #   self.state = state
    # else:
    self._players_turn = True
    self.board = np.empty(self._dims, dtype='object')
    for r in range(self._dims[0]):
      for c in range(self._dims[1]):
        self.board[r,c] = Cell(r,c)


  def visualize(self):
    plt.axes()
    ax = plt.gca()
    o = self._dims[0]-1/2
    t1 = cos(pi/4)
    t2 = sin(pi/4)
    for cell in self.board.flatten():
      circle = plt.Circle(
        (cell.r*t1-cell.c*t2, cell.r*t2 + cell.c*t1),
        0.3,
        linewidth=3,
        ec='black',
        fc={0: 'white', 1: 'red', 2: 'blue'}[cell.owner],

        )
      ax.add_patch(circle)
    ax.axis('off')
    plt.axis('scaled')
    plt.show()

  @classmethod
  def one_hot_encode(cls, state) -> np.array:
    one_hot = np.zeros(2*len(state)-1)
    one_hot[0] = state[0]
    for i in range(len(state)-1):
      one_hot[2*i+1] = 1 if state[i] == 1 else 0
      one_hot[2*i+2] = 1 if state[i] == 2 else 0


  @classmethod
  def get_config(cls) -> str:
    '''Returns a string representation of the current game config'''
    return '{}x{}'.format(config.HEX_BOARD_DIMS, config.HEX_BOARD_DIMS)


if __name__ == '__main__':
  a = Cell(1,2)
  print(a)
  h = Hex()
  h.perform_action(4)
  h.perform_action(0)
  h.perform_action(3)
  h.perform_action(1)
  h.perform_action(6)
  h.perform_action(2)
  print(h.is_terminal_state())
  h.visualize()
  h.perform_action(5)
  print(h.is_terminal_state())
  h.perform_action(7)
  print(h.is_terminal_state())
  print(h.is_terminal_state())

  print(h)
  print([(c.r, c.c) for c in h.board.flatten()])
  print(h.get_possible_actions())
  # print(h.board[1,1].owner)
  # print(h.board[1,2].owner)
  print([(x.r, x.c) for x in h.board[0,1].group])
  # print(h.board[1,2].group)
  h.visualize()