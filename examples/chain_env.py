import numpy as np
import gym
import argparse

from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, NoisyDense, BatchNormalization
from keras.optimizers import Adam
import keras.layers as ke

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, BoltzmannGumbelQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory

from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import gym_chain




parser = argparse.ArgumentParser()
parser.add_argument('--envname', type=str, default='Chainbla-v0')
parser.add_argument('--n', default=10)
parser.add_argument('--episodes', default=2000)
parser.add_argument('--exp', choices=['eps', 'bq', 'bgq', 'leps', 'noisy', 'bs'], default='eps')
parser.add_argument('--bsheads', default=10)
parser.add_argument('--seed', default=1)
args = parser.parse_args()



ENV_NAME = args.envname
POL = args.exp
n = int(args.n)
episodes = int(args.episodes)
bsheads = int(args.bsheads)
seed = int(args.seed)


nb_steps = (n + 9) * (episodes + 2)

memory_limit = nb_steps

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
env.__init__(n=n)
np.random.seed(seed)
env.seed(seed)
nb_actions = env.action_space.n
input_shape = (1,) + env.observation_space.shape
print(input_shape)
print(env.observation_space.shape)


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



if POL == 'bs':
    bootstrap_models = []
    for m in range(bsheads):
        bootstrap_models.append(make_model())
    model = bootstrap_models[0]
    print(model.summary())
    bootstrap = True
else:
    model = make_model()
    bootstrap_models= None
    print(model.summary())
    bootstrap = False

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=memory_limit, window_length=1)

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



dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=5*(n + 9), train_interval=4,
                   target_model_update=1e-2, policy=policy, bootstrap=bootstrap, bootstrap_models=bootstrap_models)


if bootstrap:
    dqn.init_bootstrap()
    for m in range(10):
        dqn.get_bootstrap(m)
        dqn.compile(Adam(lr=1e-3), metrics=['mae'])
        dqn.set_bootstrap(m)
else:
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

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
