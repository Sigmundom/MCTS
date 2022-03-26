from math import sqrt
from queue import Empty
from random import sample
from typing import Type
import tensorflow as tf
import numpy as np
from multiprocessing import Process, Queue, Lock


from ANET import ANET
from Hex import Hex
from SimpleNIM import SimpleNIM
from StateManager import StateManager
from MCTree import MCTree
import config


def run_episode(game, behaviour_policy, rbuf_queue):
    # (a) Initialize the actual game board (Ba) to an empty board.
    actual_board = game()
    # (b) sinit ← starting board state
    initial_state = actual_board.state
    # (c) Initialize the Monte Carlo Tree (MCT) to a single root, which represents sinit
    MC_tree = MCTree(initial_state)
    # (d) While Ba not in a final state:
    while not actual_board.is_terminal_state:
        # • Initialize Monte Carlo game board (Bmc) to same state as root.
        MC_board = game(MC_tree.root.state)
        # • For gs in number search games:
        for g_s in range(config.NUMBER_OF_SEACH_GAMES):
            MC_tree.simulate(MC_board, behaviour_policy)
            MC_board.reset(MC_tree.root.state)

        # • next gs ???
        # • D = distribution of visit counts in MCT along all arcs emanating from root.
        D = np.zeros(config.NUM_ACTIONS)
        for action, node in MC_tree.root.children.items():
            D[action-1] = node.visit_count

        D /= D.sum()

        # • Add case (root, D) to RBUF
        root_state = game.one_hot_encode(MC_tree.root.state)

        # self.RBUF.append((root_state, D))
        rbuf_queue.put((root_state, D))

        # • Choose actual move (a*) based on D (with a good portion of randomness)
        D_a = np.power(D, (1/5))
        D_a /= D_a.sum()
        action = np.random.choice(len(D_a), p=D_a) + 1

        # • Perform a* on root to produce successor state s*
        # • Update Ba to s*
        actual_board.perform_action(action)
        # • In MCT, retain subtree rooted at s*; discard everything else.
        # • root ← s*
        MC_tree.update_root(action)

    return


class RLSystem():
    def __init__(self):
        self._game: Type[StateManager] = {
            'NIM': SimpleNIM, 'Hex': Hex
        }[config.GAME]

        self.rbuf = []

    # 1. i_s = save interval for ANET (the actor network) parameters
        self.ANET = ANET(one_hot_encode=self._game.one_hot_encode)
        # self.ANET_lock = Lock()
        self.rbuf_queue = Queue()
        self.console_lock = Lock()
        self.episode_counter = config.NUMBER_OF_EPISODES

    # 2. Clear Replay Buffer (RBUF)
    # TODO: 3. Randomly initialize parameters (weights and biases) of ANET

    def train_ANET(self):
        # while not self.rbuf_queue.empty():
        try:
            training_case = self.rbuf_queue.get(timeout=2)
            print(training_case)

            self.rbuf.append(training_case)
            # (e) Train ANET on a random minibatch of cases from RBUF
            batch = sample(self.rbuf, min(len(self.rbuf), config.BATCH_SIZE))
            self.ANET.fit(batch, epochs=3)

        except Empty as e:
            print('No items in queue')

        # if (self.episode_counter-1) % config.SAVE_INTERVAL == 0:
        #     print("Saving ANET's parameters")
        #     # • Save ANET’s current parameters for later use in tournament play.
        #     self.ANET.save('models/{}/{}/{}x{}'.format(self._game.__name__,
        #                                                self._game.get_config(), config.NUMBER_OF_EPISODES-self.episode_counter-1, config.NUMBER_OF_SEACH_GAMES))

        # # (e) Train ANET on a random minibatch of cases from RBUF
        # batch = sample(self.RBUF, min(len(self.RBUF), config.BATCH_SIZE))
        # self.ANET.fit(batch)
        # self.ANET.epsilon_decay()

    def run(self):
        print('Starting RL system')

        while self.episode_counter > 0:
            episodes = [Process(target=run_episode, args=(self._game, self.ANET.default_policy_epsilon, self.rbuf_queue))
                        for _ in range(min(self.episode_counter, config.NUM_WORKERS))]
            print('Starting processes')
            for episode in episodes:
                episode.start()

            # print('Training ANET')
            # while not self.rbuf_queue.empty() or any([e.is_alive() for e in episodes]):
            #     self.train_ANET()
            print('Training complete')
            for episode in episodes:
                episode.join()

            while not self.rbuf_queue.empty():
                item = self.rbuf_queue.get()
                print(item)
                self.rbuf.append(item)

            for i in range(2*len(episodes)):
                batch = sample(self.rbuf, min(
                    len(self.rbuf), config.BATCH_SIZE))
                # print('Batch:')
                # print(batch)
                self.ANET.fit(batch, epochs=10)

            self.ANET.epsilon_decay()
            # print('Processes joined')
            # self.ANET.epsilon_decay()

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
