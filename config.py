from operator import mul

# GAME = 'NIM'
GAME = 'Hex'

SAVE_INTERVAL = 10  # save interval for ANET (the actor network) parameters
NUMBER_OF_EPISODES = 30
NUMBER_OF_SEACH_GAMES = 500
EPSILON = 1
EPSILON_DECAY_RATE = 0.5

# NIM
NUM_PIECES = 10
MAX_PICK = 4

# HEX
HEX_BOARD_DIMS = (3, 3)

NUM_ACTIONS = {'NIM': MAX_PICK, 'Hex': mul(*HEX_BOARD_DIMS)}[GAME]
STATE_SIZE = {'NIM': NUM_PIECES, 'Hex': 2*mul(*HEX_BOARD_DIMS)+1}[GAME]

BATCH_SIZE = 16
NUM_WORKERS = 5
