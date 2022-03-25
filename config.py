from operator import mul

GAME = 'NIM'
# GAME = 'Hex'

SAVE_INTERVAL = 25 # save interval for ANET (the actor network) parameters
NUMBER_OF_GAMES = 50
NUMBER_OF_SEACH_GAMES = 500
EPSILON = 1
EPSILON_DECAY_RATE = 0.5

# NIM
NUM_PIECES = 20
MAX_PICK = 3

# HEX
HEX_BOARD_DIMS = (3, 3)

NUM_ACTIONS = {'NIM': MAX_PICK, 'Hex': mul(*HEX_BOARD_DIMS)}[GAME]
STATE_SIZE = {'NIM': 1 + NUM_PIECES, 'Hex': mul(*HEX_BOARD_DIMS)+1}[GAME]

BATCH_SIZE = 32