from random import choice, random
from typing import Callable, Optional, Tuple
import numpy as np
import tensorflow as tf
import config
from type_aliases import PossibleActions, State

# x_train, x_test = x_train / 255.0, x_test / 255.0
# (x_train, y_train),(x_test, y_test) = mnist.load_data()

# Define model


class ANET:
    def __init__(
        self,
        one_hot_encode: Callable[[State], State] = None,
        model_name: str = None
    ):
        self.one_hot_encode = one_hot_encode
        self._epsilon = config.EPSILON

        initializer = tf.keras.initializers.RandomNormal(
            mean=0.0, stddev=0.05, seed=None)

        if model_name:
            self.model = tf.keras.models.load_model(model_name)
        else:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.InputLayer(input_shape=(config.STATE_SIZE,)),
                *[tf.keras.layers.Dense(neurons, activation=config.ACTIVATION_HIDDEN)
                  for neurons in config.HIDDEN_LAYERS],
                tf.keras.layers.Dense(
                    config.NUM_ACTIONS, activation=config.ACTIVATION_OUTPUT)
            ])

            self.model.compile(optimizer=config.OPTIMIZER(learning_rate=config.LEARNING_RATE),
                               loss=config.LOSS,
                               metrics=['accuracy', 'mse'])
        self.model.build()
        self.model.summary()

    def fit(self, batch, epochs=5):
        x, y = zip(*batch)      
        self.model.fit(np.array(x), np.array(y), epochs=epochs,
                       batch_size=max(len(batch), config.BATCH_SIZE))

    def evaluate(self, input, target):
        self.model.evaluate(input, target)

    def save(self, name):
        self.model.save(name)

    def default_policy_epsilon(self, state: State, possible_actions: PossibleActions) -> int:
        if random() < self._epsilon:
            return choice(possible_actions)
        return self.default_policy(state, possible_actions)

    def default_policy(self, state: State, possible_actions: Tuple[int, ...]) -> int:
        one_hot = self.one_hot_encode(
            state) if self.one_hot_encode is not None else state
        output = self.model(np.expand_dims(one_hot, axis=0))[0].numpy()
        best_action = np.argmax(output)
        while best_action not in possible_actions:
            output[best_action] = -1
            best_action = np.argmax(output)
        # print(output)
        # print('Best action:', best_action)
        return best_action

    def epsilon_decay(self):
        self._epsilon *= config.EPSILON_DECAY_RATE
