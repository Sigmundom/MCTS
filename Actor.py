from ANET import ANET
from StateManager import StateManager


# class Actor:
#   def __init__(self, game:StateManager, model_name:str=None, player:bool=False):
#     self._game = game
#     self.player = player
#     if not player:
#       if not model_name:
#         raise Exception('Actor must be a player or be supplied with a model name')
#       self.ANET = ANET(game.one_hot_encode, model_name)

#   def get_next_move(self):
#     if self.player:
