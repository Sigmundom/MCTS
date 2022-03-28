import multiprocessing
from operator import mul
from tensorflow.keras.optimizers import Adam

# GAME = 'NIM'
GAME = 'Hex'

# NIM
NUM_PIECES = 10
MAX_PICK = 4

# HEX
HEX_BOARD_DIMS = (3, 3)

# General
SAVE_INTERVAL = 10  # save interval for ANET (the actor network) parameters
NUMBER_OF_EPISODES = 60
NUMBER_OF_SEACH_GAMES = 1000

# Factors
EPSILON = 1
EPSILON_DECAY_RATE = 0.5
DISTRIBUTION_SMOOTHING_FACTOR = {'NIM': 5, 'Hex': 2}[GAME]

# ANET
BATCH_SIZE = 8
LEARNING_RATE = 0.001
OPTIMIZER = Adam
LOSS = 'categorical_crossentropy'
ACTIVATION_HIDDEN = 'relu'
ACTIVATION_OUTPUT = 'softmax'
HIDDEN_LAYERS = [32, 32]

# Misc
# NUM_WORKERS = 1
NUM_WORKERS = multiprocessing.cpu_count()

# Infered parameters
NUM_ACTIONS = {'NIM': MAX_PICK, 'Hex': mul(*HEX_BOARD_DIMS)}[GAME]
STATE_SIZE = {'NIM': NUM_PIECES, 'Hex': 2*mul(*HEX_BOARD_DIMS)+1}[GAME]
