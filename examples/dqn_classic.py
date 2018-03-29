import numpy as np
import gym
import argparse

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, BoltzmannGumbelQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

from rl.callbacks import FileLogger, ModelIntervalCheckpoint


parser = argparse.ArgumentParser()
parser.add_argument('--envname', type=str, default='MountainCar-v0')
parser.add_argument('--nbsteps', default=50000)
parser.add_argument('--mem', default=50000)
parser.add_argument('--exp', choices=['eps', 'bq', 'bgq', 'leps'], default='eps')
args = parser.parse_args()



ENV_NAME = args.envname
POL = args.exp
nb_steps = args.nbsteps
memory_limit = args.mem

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=memory_limit, window_length=1)

if POL == 'eps':
	policy = EpsGreedyQPolicy()
elif POL == 'bq':
	policy = BoltzmannQPolicy()
elif POL == 'bgq':
	policy = BoltzmannGumbelQPolicy()	
elif POL == 'leps':
	policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.02, value_test=.02,
                              nb_steps=(nb_steps/2))


dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

log_filename = 'logs/dqn_' + '_'.join('{}:{}'.format(key, val) for key, val in sorted(vars(args).items())) + '_log.json'

checkpoint_weights_filename = 'models/dqn_' + '_'.join('{}:{}'.format(key, val) for key, val in sorted(vars(args).items())) + '_{step}.h5f'

print(checkpoint_weights_filename)

callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=(nb_steps/10), verbose =1)]
callbacks += [FileLogger(log_filename, interval=(nb_steps/10))]
dqn.fit(env, nb_steps=nb_steps, callbacks=callbacks, verbose =1,log_interval=(nb_steps/10))

# After training is done, we save the final weights.
dqn.save_weights('models/dqn_' + '_'.join('{}:{}'.format(key, val) for key, val in sorted(vars(args).items())) + '_final.h5f', overwrite=True)

# Finally, evaluate our algorithm for n episodes.
#dqn.test(env, nb_episodes=100, visualize=False)
