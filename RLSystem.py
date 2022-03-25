from math import sqrt
from random import sample
from typing import Type
import tensorflow as tf
import numpy as np


from ANET import ANET
from Hex import Hex
from SimpleNIM import SimpleNIM
from StateManager import StateManager
from MCTree import MCTree
import config

class RLSystem():
  def __init__(self):
    self._game: Type[StateManager] = {'NIM': SimpleNIM, 'Hex': Hex}[config.GAME]
    self.RBUF = []

  # 1. i_s = save interval for ANET (the actor network) parameters
    self.ANET = ANET(one_hot_encode=self._game.one_hot_encode)
  # 2. Clear Replay Buffer (RBUF)
  # TODO: 3. Randomly initialize parameters (weights and biases) of ANET

  def _run_episode(self):
      # (a) Initialize the actual game board (Ba) to an empty board.
      actual_board = self._game()
      # (b) sinit ← starting board state
      initial_state = actual_board.state
      # (c) Initialize the Monte Carlo Tree (MCT) to a single root, which represents sinit
      MC_tree = MCTree(initial_state)
      # (d) While Ba not in a final state:
      while not actual_board.is_terminal_state:
        # • Initialize Monte Carlo game board (Bmc) to same state as root.
        MC_board = self._game(MC_tree.root.state)
        # • For gs in number search games:
        for g_s in range(config.NUMBER_OF_SEACH_GAMES):
          MC_tree.simulate(MC_board, self.ANET.default_policy)
          MC_board.reset(MC_tree.root.state)

        # • next gs ???
        # • D = distribution of visit counts in MCT along all arcs emanating from root.
        D = np.zeros(config.NUM_ACTIONS)
        for action, node in MC_tree.root.children.items():
          D[action-1] = node.visit_count

        D /= D.sum()

        # rot av D og normaliser

        # D = [(action, child.visit_count) for action, child in MC_tree.root.children.items()]

        # • Add case (root, D) to RBUF
        root_state = self._game.one_hot_encode(MC_tree.root.state)
        print(MC_tree.root.state[1], D)

        self.RBUF.append((root_state, D))

        # • Choose actual move (a*) based on D
        D_a = np.power(D, (1/5))
        D_a /= D_a.sum()
        print(D_a)
        action = np.random.choice(len(D_a), p=D_a) + 1


        # • Perform a* on root to produce successor state s*
        # • Update Ba to s*
        actual_board.perform_action(action)
        # • In MCT, retain subtree rooted at s*; discard everything else.
        # • root ← s*
        # print(MC_tree)
        # exit()
        MC_tree.update_root(action)

      # (e) Train ANET on a random minibatch of cases from RBUF
      batch = sample(self.RBUF, min(len(self.RBUF), config.BATCH_SIZE))
      self.ANET.fit(batch)
      self.ANET.epsilon_decay()


  def run(self):
    print('Starting RL system')
    # 4. For ga in number actual games:
    for g_a in range(config.NUMBER_OF_GAMES):
      print('Episode {}'.format(g_a), end='\r')
      self._run_episode()

      if (g_a+1) % config.SAVE_INTERVAL == 0:
        print("Saving ANET's parameters")
        # • Save ANET’s current parameters for later use in tournament play.
        self.ANET.save('models/{}/{}/{}x{}'.format(self._game.__name__,self._game.get_config(), g_a+1, config.NUMBER_OF_SEACH_GAMES))

if __name__ == '__main__':
  rl = RLSystem()
  rl.run()