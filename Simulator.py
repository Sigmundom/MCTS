from turtle import pos
from typing import Optional, Type
from InquirerPy import inquirer

from ANET import ANET
from StateManager import StateManager
from type_aliases import PossibleActions, State


class Player:
    def get_next_move(self, state, possible_actions):
        ...


class ManualPlayer(Player):
    def get_next_move(self, state: State, possible_actions: PossibleActions):
        action = inquirer.number(
            message='Enter action:',
            min_allowed=possible_actions[0],
            max_allowed=possible_actions[-1],
            validate=lambda x: x and int(x) in possible_actions,
            invalid_message='You must choose a number between {} and {}'.format(
                possible_actions[0], possible_actions[-1]),
            default=None
        ).execute()
        return int(action)


class AIPlayer(Player):
    def __init__(self, ai: ANET):
        self.ai = ai

    def get_next_move(self, state, possible_actions):
        return self.ai.default_policy(state, possible_actions)


class Simulator:
    def __init__(self, game: Type[StateManager], player1: Optional[str] = None, player2: Optional[str] = None):
        self.game = game
        if player1 is None:
            self.player1 = ManualPlayer()
        else:
            ai = ANET(game.one_hot_encode, player1)
            self.player1 = AIPlayer(ai)

        if player2 is None:
            self.player2 = ManualPlayer()
        else:
            ai = ANET(game.one_hot_encode, player2)
            self.player2 = AIPlayer(ai)

    def run(self):
        board = self.game()
        while not board.is_terminal_state:
            board.visualize()
            if board._players_turn:
                next_move = self.player1.get_next_move(
                    board.state, board.get_possible_actions())
            else:
                next_move = self.player2.get_next_move(
                    board.state, board.get_possible_actions())
            board.perform_action(next_move)

        winner = board.get_winner()
        print('The winner is ' + ('Player 1' if winner == 1 else 'Player 2'))
        board.visualize()
