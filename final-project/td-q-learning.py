#!/usr/bin/env python3
#

#
# Using the FrozenLake environment from Farama's Gymnasium
#
# https://gymnasium.farama.org/environments/toy_text/frozen_lake/
#

import numpy as np 
import gymnasium as gym

import sys
import argparse
import logging
import os.path
import joblib


class QTable:
    def __init__(self, n_states, n_actions):
        """
        n_states:      integer; number of states
        n_actions:     integer; number of actions
        """
        self.Q1 = self.create_Q_table(n_states, n_actions)
        self.Q2 = self.create_Q_table(n_states, n_actions)
        return

    def n_states(self):
        """
        """
        return self.Q1.shape[0] # does not matter if we use Q1 or Q2 for n_state or n_actions as they both have the same number of state and actions

    def n_actions(self):
        """
        """
        return self.Q1.shape[1]

    def actions(self):
        """
        """
        return [action for action in range(self.n_actions())]

    def create_Q_table(self, n_states, n_actions):
        """
        n_states:      integer; number of states
        n_actions:     integer; number of actions
        """
        Q = np.zeros([n_states, n_actions])
        return Q

    def update(self, state, action, next_state, reward, alpha, gamma):
        """
        state:      integer; state index
        action:     numpy.int64; action index
        next_state: integer; state index
        reward:     float; immediate reward
        alpha:      float; immediate vs historical weight
        gamma:      float; future discount factor
        """
        if np.random.rand() < 0.5:
            # Update Q1 using Q2's estimates for the next state
            best_next_action = np.argmax(self.Q1[next_state, :])
            td_target = reward + gamma * self.Q2[next_state, best_next_action]
            self.Q1[state, action] += alpha * (td_target - self.Q1[state, action])
        else:
            # Update Q2 using Q1's estimates for the next state
            best_next_action = np.argmax(self.Q2[next_state, :])
            td_target = reward + gamma * self.Q1[next_state, best_next_action]
            self.Q2[state, action] += alpha * (td_target - self.Q2[state, action])
        return
    
    def get_best_action(self, state):
        """
        state:      integer; state index
        """
        average_Q = (self.Q1[state, :] + self.Q2[state, :]) / 2
        return np.argmax(average_Q)

    def get_Q_value(self, state, action):
        """
        state:      integer; state index
        action:     numpy.int64; action index
        """
        return (self.Q1[state, action] + self.Q2[state, action]) / 2

    def get_best_action_value(self, state):
        """
        state:      integer; state index
        """
        average_Q = (self.Q1[state, :] + self.Q2[state, :]) / 2
        best_action = np.argmax(average_Q)
        best_Q_value = average_Q[best_action]
        return best_action, best_Q_value

    def save(self, model_file):
        joblib.dump((self.Q1, self.Q2), model_file)
        return

    def load(self, model_file):
        self.Q1, self.Q2 = joblib.load(model_file)
        return


def get_model_filename(model_file, environment_name):
    if model_file == "":
        model_file = "{}-model.joblib".format(environment_name)
    return model_file

# The openai gym environment is loaded
def load_environment(my_args):
    if my_args.environment == 'lake':
        env = gym.make('Blackjack-v1')
    else:
        raise Exception("Unexpected environment: {}".format(my_args.environment))
    # env.observation.n, env.action_space.n gives number of states and action in env loaded
    return env

def encode_state(state):
    hand = state[0] + (state[1] * 32) + (state[2] * 32 * 11)
    return hand

def learn_epoch(Q, env, chance_epsilon, alpha, gamma, my_args):
    action_list = Q.actions()

    # Reset environment, getting initial state
    state, info = env.reset()
    state = encode_state(state)
    epoch_total_reward = 0
    epoch_done = False
    epoch_truncated = False

    # The Q-Table temporal difference learning algorithm
    while (not epoch_done) and (not epoch_truncated):
        # Choose action from Q table
        # To facilitate learning, have chance of random action
        # instead of always choosing the best action
        chance = np.random.sample(1)[0]
        if chance < chance_epsilon:
            action = np.random.choice(action_list)
        else:
            action = Q.get_best_action(state)

        # Take action, get the new state and reward
        next_state, reward, epoch_done, epoch_truncated, info = env.step(action)
        next_state = encode_state(next_state)
        if my_args.track_steps:
            print(env.render(mode="ansi"))

        # Update Q-Table with new data
        Q.update(state, action, next_state, reward, alpha, gamma)
        epoch_total_reward += reward
        state = next_state

    return state, epoch_total_reward

def evaluate_epoch(Q, env, my_args):
    action_list = Q.actions()

    # Reset environment, getting initial state
    state, info = env.reset()
    state = encode_state(state)
    epoch_total_reward = 0
    epoch_done = False
    epoch_truncated = False

    # The Q-Table policy evaluation
    while (not epoch_done) and (not epoch_truncated):
        # Choose action from Q table
        action = Q.get_best_action(state)

        # Take action, get the new state and reward
        next_state, reward, epoch_done, epoch_truncated, info = env.step(action)
        next_state = encode_state(next_state)
        if my_args.track_steps:
            print(env.render(mode="ansi"))

        # Update reward and state
        epoch_total_reward += reward
        state = next_state

    return state, epoch_total_reward

def Q_learn(Q, env, my_args):
    almost_one = my_args.epsilon_chance_factor
    alpha = my_args.alpha
    gamma = my_args.gamma
    epoch_rewards = [] # rewards per epochs
    chance_epsilon = almost_one

    for epoch_number in range(my_args.n_epochs):
        state, epoch_total_reward = learn_epoch(Q, env, chance_epsilon, alpha, gamma, my_args)
        epoch_rewards.append(epoch_total_reward)
        if my_args.track_epochs:
            print("epoch: {}  reward: {}".format(epoch_number, epoch_total_reward))

        # make less likely to experiment
        # assumes positive scores for successful completion
        if epoch_total_reward > 0:
            chance_epsilon *= almost_one
        
    return epoch_rewards

def Q_evaluate(Q, env, my_args):
    epoch_rewards = [] # rewards per epochs

    for epoch_number in range(my_args.n_epochs):
        state, epoch_total_reward = evaluate_epoch(Q, env, my_args)
        epoch_rewards.append(epoch_total_reward)
        if my_args.track_epochs:
            print("epoch: {}  reward: {}".format(epoch_number, epoch_total_reward))
        
    return epoch_rewards

def do_learn(my_args):
    # Load Environment
    env = load_environment(my_args)

    # Build new Q-table structure
    # assumes that the environment has discrete observation and action spaces
    Q = QTable(32 * 11 * 2, env.action_space.n)
    
    # Learn
    epoch_rewards = Q_learn(Q, env, my_args)

    print("Learn: Average reward on all epochs " + str(sum(epoch_rewards)/my_args.n_epochs))

    model_file = get_model_filename(my_args.model_file, my_args.environment)
    Q.save(model_file)
    print("Model saved to {}.".format(model_file))
    return

def do_score(my_args):
    # Load Environment
    env = load_environment(my_args)

    # Load existing Q-Table
    # assumes that the environment has discrete observation and action spaces
    Q = QTable(0, 0)
    model_file = get_model_filename(my_args.model_file, my_args.environment)
    print("Model loading from {}.".format(model_file))
    Q.load(model_file)


    # Evaluate model
    epoch_rewards = Q_evaluate(Q, env, my_args)

    print("Score: Average reward on all epochs " + str(sum(epoch_rewards)/my_args.n_epochs))
    
    return

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Q-Table Learning')
    parser.add_argument('action', default='learn',
                        choices=[ "learn", "score", ], 
                        nargs='?', help="desired action")
    
    parser.add_argument('--environment',   '-e', default="lake", type=str,  choices=('lake', ), help="name of the OpenAI gym environment")
    parser.add_argument('--model-file',    '-m', default="",    type=str,   help="name of file for the model (default is constructed from environment)")

    #
    # hyper parameters
    #
    parser.add_argument('--alpha', '-a', default=0.5,  type=float, help="Temporal difference learning hyper parameter (default=0.5)")
    parser.add_argument('--gamma', '-g', default=0.5,  type=float, help="Q-learning hyper parameter (default=0.5)")
    parser.add_argument('--epsilon-chance-factor', '-c', default=0.1,  type=float, help="Scaling factor for learning policy chance of choosing random action (default=0.1)")

    parser.add_argument('--n-epochs', '-n',   default=10, type=int,   help="number of episodes to run (default=10).")

    # debugging/observations
    parser.add_argument('--track-epochs',    '-t', default=0,         type=int,   help="0 = don't display per-epoch information, 1 = do display per-epoch information (default=0)")
    parser.add_argument('--track-steps',     '-s', default=0,         type=int,   help="0 = don't display per-step information, 1 = do display per-step information (default=0)")


    my_args = parser.parse_args(argv[1:])

    #
    # Do any special fixes/checks here
    #
    
    return my_args

def main(argv):
    my_args = parse_args(argv)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)

    if my_args.action == 'learn':
        do_learn(my_args)
    elif my_args.action == 'score':
        do_score(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))

    return

if __name__ == "__main__":
    main(sys.argv)

    
