import os
from Hex import Hex
from InquirerPy import prompt, inquirer
from InquirerPy.base.control import Choice
from SimpleNIM import SimpleNIM
from Simulator import Simulator

games = [SimpleNIM, Hex]


def main():
    game = inquirer.select(
        message="Welcom!\nSelect a game!",
        choices=list(map(lambda game: Choice(
            value=game, name=game.__name__), games))
    ).execute()

    # mode = inquirer.select(
    #     message='What type of simulation do you want?',
    #     choices=['AI vs AI', ]
    # ).execute()

    players = []
    for i in range(2):
        ai = inquirer.confirm(
            message="Do you want to use and AI for player {}?".format(i+1),
            default=True
        ).execute()

        if ai:
            model_path = 'models/{}'.format(game.__name__)
            config = inquirer.select(
                message="Select configuration",
                choices=os.listdir(model_path)
            ).execute()
            model_path += '/{}'.format(config)
            episodes = inquirer.select(
                message='How well trained should the AI be? (number of episodes)',
                choices=os.listdir(model_path)
            ).execute()
            model_path += '/{}'.format(episodes)
            players.append(model_path)
        else:
            players.append(None)

    simulator = Simulator(game, *players)
    simulator.run()


if __name__ == "__main__":
    main()
