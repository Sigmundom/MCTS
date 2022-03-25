from abc import ABC, abstractmethod
from typing import Iterable, Optional
import numpy as np

from type_aliases import State


class StateManager(ABC):

  @property
  @abstractmethod
  def players_turn(self):
    ...

  @property
  @abstractmethod
  def state(self) -> State:
    ...

  @abstractmethod
  def __str__(self) -> str:
    ...

  @abstractmethod
  def generate_child_state(self, action: int) -> State:
    '''
    Predicts a child state based on an action without affecting the board state.

      Parameters:
        action (int): A legal action

      Returns:
        child_state (State): The next state if the given action is performed.

    '''
    ...

  @abstractmethod
  def get_possible_actions(self) -> Iterable[int]:
    '''Returns a list of '''
    ...

  @abstractmethod
  def get_winner(self) -> int:
    '''
    Should only be called when game is terminated

      Returns:
        winner (int): 1 means player won. -1 means opponent won. 0 means draw.
    '''

  @abstractmethod
  def is_terminal_state(self) -> bool:
    ...

  @abstractmethod
  def perform_action(self, action: int):
    ...

  @abstractmethod
  def reset(self, state: Optional[State] = None) -> None:
    ...

  @abstractmethod
  def visualize(self):
    ...


  @classmethod
  @abstractmethod
  def one_hot_encode(cls, state) -> np.array:
    ...

  @classmethod
  @abstractmethod
  def get_config(cls) -> str:
    '''Returns a string representation of the current game config'''
    ...

