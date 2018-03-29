import numpy as np
import gym
import argparse
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute

from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, BoltzmannGumbelQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

from rl.core import Processor

from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import keras.backend as K


parser = argparse.ArgumentParser()
parser.add_argument('--envname', type=str, default='Pong-v0')
parser.add_argument('--nbsteps', default=1750000)
parser.add_argument('--mem', default=1000000)
parser.add_argument('--exp', choices=['eps', 'bq', 'bgq', 'leps'], default='eps')
args = parser.parse_args()



ENV_NAME = args.envname
POL = args.exp
nb_steps = args.nbsteps
memory_limit = args.mem


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



# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build our model. We use the same model that was described by Mnih et al. (2015).
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
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
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=memory_limit, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

if POL == 'eps':
	policy = EpsGreedyQPolicy()
elif POL == 'bq':
	policy = BoltzmannQPolicy()
elif POL == 'bgq':
	policy = BoltzmannGumbelQPolicy()	
elif POL == 'leps':
	policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.02, value_test=.02,
                              nb_steps=(nb_steps/2))



dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
               
dqn.compile(Adam(lr=.00025), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

log_filename = 'logs/dqn_' + '_'.join('{}:{}'.format(key, val) for key, val in sorted(vars(args).items())) + '_log.json'

checkpoint_weights_filename = 'models/dqn_' + '_'.join('{}:{}'.format(key, val) for key, val in sorted(vars(args).items())) + '_{step}.h5f'


callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=(nb_steps/10), verbose =1)]
callbacks += [FileLogger(log_filename, interval=(nb_steps/10))]
dqn.fit(env, nb_steps=nb_steps, callbacks=callbacks, verbose =1,log_interval=(nb_steps/10))

# After training is done, we save the final weights.
dqn.save_weights('models/dqn_' + '_'.join('{}:{}'.format(key, val) for key, val in sorted(vars(args).items())) + '_final.h5f', overwrite=True)

# Finally, evaluate our algorithm for n episodes.
#dqn.test(env, nb_episodes=100, visualize=False)
