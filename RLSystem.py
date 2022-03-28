from queue import Empty
from random import sample
from typing import Type
import numpy as np
from multiprocessing import Process, Queue


from ANET import ANET
from Hex import Hex
from SimpleNIM import SimpleNIM
from StateManager import StateManager
from MCTree import MCTree
import config


class Episode(Process):
    def __init__(self, game, behaviour_policy, rbuf_queue):
        super().__init__()
        self.actual_board = game()
        self.MC_board = game()
        self.behaviour_policy = behaviour_policy
        self.rbuf_queue = rbuf_queue
        self.MC_tree = MCTree(self.actual_board.state)
        self.one_hot = game.one_hot_encode

    def run(self):
        while not self.actual_board.is_terminal_state:
            for _ in range(config.NUMBER_OF_SEACH_GAMES):
                self.MC_tree.simulate(self.MC_board, self.behaviour_policy)

            # • D = distribution of visit counts in MCT along all arcs emanating from root.
            D = np.zeros(config.NUM_ACTIONS)
            for action, node in self.MC_tree.root.children.items():
                D[action] = node.visit_count

            D /= D.sum()

            # • Add case (root, D) to RBUF
            root_state = self.one_hot(self.MC_tree.root.state)

            # self.RBUF.append((root_state, D))
            self.rbuf_queue.put((root_state, D))

            # • Choose actual move (a*) based on D (with a good portion of randomness)
            D_a = np.power(D, (1/config.DISTRIBUTION_SMOOTHING_FACTOR))
            D_a /= D_a.sum()
            action = np.random.choice(len(D_a), p=D_a)

            # • Perform a* on root to produce successor state s*
            # • Update Ba to s*
            self.actual_board.perform_action(action)
       
            # • In MCT, retain subtree rooted at s*; discard everything else.
            # • root ← s*
            self.MC_tree.update_root(action)


# def run_episode(game, behaviour_policy, rbuf_queue):

    # return


class RLSystem():
    def __init__(self):
        self._game: Type[StateManager] = {
            'NIM': SimpleNIM, 'Hex': Hex
        }[config.GAME]

        self.rbuf = []

    # 1. i_s = save interval for ANET (the actor network) parameters
        self.ANET = ANET(one_hot_encode=self._game.one_hot_encode)
        self.rbuf_queue = Queue()
        self.episode_counter = config.NUMBER_OF_EPISODES

    def run(self):
        print('Starting RL system with {} workers'.format(config.NUM_WORKERS))

        while self.episode_counter > 0:
            episodes = []

            print('Starting processes')
            for _ in range(min(self.episode_counter, config.NUM_WORKERS)):
                episode = Episode(
                    self._game, self.ANET.default_policy_epsilon, self.rbuf_queue)
                episode.start()
                episodes.append(episode)

            for episode in episodes:
                episode.join()
            print('Training complete')

            while not self.rbuf_queue.empty():
                item = self.rbuf_queue.get()
                print(item)
                self.rbuf.append(item)

            batch = sample(self.rbuf, min(
                len(self.rbuf), len(episodes)*config.BATCH_SIZE))
            self.ANET.fit(batch, epochs=10)

            self.ANET.epsilon_decay()

            for i in range(self.episode_counter, self.episode_counter-len(episodes), -1):
                if (i-1) % config.SAVE_INTERVAL == 0:
                    print("Saving ANET's parameters")
                    # • Save ANET’s current parameters for later use in tournament play.
                    self.ANET.save('models/{}/{}/{}x{}'.format(self._game.__name__,
                                                               self._game.get_config(), config.NUMBER_OF_EPISODES-i+1, config.NUMBER_OF_SEACH_GAMES))
            self.episode_counter -= len(episodes)

            print('{} episodes left'.format(self.episode_counter))


if __name__ == '__main__':
    rl = RLSystem()
    rl.run()
