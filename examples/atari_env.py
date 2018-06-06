from __future__ import division
import numpy as np
from PIL import Image
import gym
import argparse

from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, NoisyDense, BatchNormalization, Convolution2D, Permute
from keras.optimizers import Adam
import keras.layers as ke
import keras.backend as K


from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, BoltzmannGumbelQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor

from rl.callbacks import FileLogger, ModelIntervalCheckpoint


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)


parser = argparse.ArgumentParser()
parser.add_argument('--envname', type=str, default='BreakoutDeterministic-v4')
parser.add_argument('--nbsteps', default=1750000)
parser.add_argument('--mem', default=1000000)
parser.add_argument('--exp', choices=['eps', 'bq', 'bgq', 'leps', 'noisy', 'bs'], default='eps')
parser.add_argument('--bsheads', default=10)
parser.add_argument('--seed', default=1)
args = parser.parse_args()



ENV_NAME = args.envname
POL = args.exp
bsheads = int(args.bsheads)
seed = int(args.seed)
memory_limit = args.mem


nb_steps = args.nbsteps


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(seed)
env.seed(seed)
nb_actions = env.action_space.n
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE



def make_model():
    if K.image_dim_ordering() == 'tf':
        # (width, height, channels)
        model.add(Permute((2, 3, 1), input_shape=input_shape))
    elif K.image_dim_ordering() == 'th':
        # (channels, width, height)
        model.add(Permute((1, 2, 3), input_shape=input_shape))
    else:
        raise RuntimeError('Unknown image_dim_ordering.')
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    
    if POL == 'noisy':
        model.add(NoisyDense(512))
        model.add(Activation('relu'))
        model.add(NoisyDense(nb_actions))
    else:
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    return model



if POL == 'bs':
    bootstrap_models = []
    for m in range(bsheads):
        bootstrap_models.append(make_model())
    model = bootstrap_models[0]
    print(model.summary())
    bootstrap = True
    memory_limit = (int)(memory_limit/bsheads)
else:
    model = make_model()
    bootstrap_models= None
    print(model.summary())
    bootstrap = False

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=memory_limit, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

if POL == 'eps' or bootstrap:
    policy = EpsGreedyQPolicy()
elif POL == 'bq':
    policy = BoltzmannQPolicy()
elif POL == 'bgq':
    policy = BoltzmannGumbelQPolicy()
elif POL == 'leps':
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.02, value_test=.02,
                              nb_steps=(nb_steps/2))
else:
    policy = GreedyQPolicy()



dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50000, train_interval=4, processor=processor,
                   target_model_update=10000, delta_clip=1., gamma=.99, policy=policy, bootstrap=bootstrap, bootstrap_models=bootstrap_models)


if bootstrap:
    dqn.init_bootstrap()
    for m in range(10):
        dqn.get_bootstrap(m)
        dqn.compile(Adam(lr=.00025), metrics=['mae'])
        dqn.set_bootstrap(m)
else:
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

log_filename = 'logs/dqn_' + '_'.join('{}:{}'.format(key, val) for key, val in sorted(vars(args).items())) + '_log.json'

checkpoint_weights_filename = 'models/dqn_' + '_'.join('{}:{}'.format(key, val) for key, val in sorted(vars(args).items())) + '_{step}.h5f'

print(checkpoint_weights_filename)

callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=(nb_steps/10), verbose =1)]
callbacks += [FileLogger(log_filename, interval=(nb_steps/10))]
dqn.fit(env, nb_steps=nb_steps, callbacks=callbacks, verbose =2,log_interval=(nb_steps/10))

# After training is done, we save the final weights.
dqn.save_weights('models/dqn_' + '_'.join('{}:{}'.format(key, val) for key, val in sorted(vars(args).items())) + '_final.h5f', overwrite=True)

# Finally, evaluate our algorithm for n episodes.
#dqn.test(env, nb_episodes=100, visualize=False)
