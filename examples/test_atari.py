import numpy as np
import gym
import argparse
from PIL import Image

from keras import backend as K
from keras.models import Model

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Input, Concatenate, Add#, NoisyDense, BatchNormalization
from keras.optimizers import Adam
import keras.layers as ke

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, BoltzmannGumbelQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy, BootstrapPolicy
from rl.memory import SequentialMemory

from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import gym_chain
from keras.utils import plot_model

from rl.core import Processor


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
parser.add_argument('--bsheads', default=1)
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
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(16))
    model.add(Activation('relu'))

    if POL == 'noisy':
        model.add(NoisyDense(16))
        model.add(Activation('relu'))
        model.add(NoisyDense(nb_actions))
    else:
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    return model


def make_model_bs():
    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(16))
    model.add(Activation('relu'))

    input = Input(shape=input_shape)
    x = model(input)

    heads = []
    for _ in range(bsheads):
        head =Dense(16, activation='relu')(x)
        head = Dense(nb_actions, activation='linear')(head)
        heads.append(head)

    combine = Concatenate()(heads)



    out = Model(inputs=input, outputs=combine)

    return out


if POL == 'bs':
    model = make_model_bs()
    bootstrap_models= None
    print(model.summary())
    bootstrap = False
else:
    model = make_model()
    bootstrap_models= None
    print(model.summary())
    bootstrap = False


plot_model(model,to_file='demo.png',show_shapes=True)
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=memory_limit, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

if POL == 'eps':
    policy = EpsGreedyQPolicy()
elif POL == 'bs':
    policy = BootstrapPolicy(total_heads = bsheads)
elif POL == 'bgq':
    policy = BoltzmannGumbelQPolicy()
elif POL == 'leps':
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.02, value_test=.02,
                              nb_steps=(nb_steps/2))
else:
    policy = GreedyQPolicy()


dqn = DQNAgent(model=model, nb_actions=nb_actions, heads=bsheads, memory=memory, nb_steps_warmup=50000, processor=processor, train_interval=4,
                   target_model_update=10000, delta_clip=1., gamma=.99, policy=policy,enable_double_dqn=bootstrap)



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
