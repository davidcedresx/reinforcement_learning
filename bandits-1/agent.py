import numpy as np
import random


class Bandit():
    def __init__(self, alpha=1, epsilon=0.5):
        """
        Initializes Bandit

        :param int alpha: Step size parameter for value estimation,
            it multiplies the estimation error and a smaller value means the agent
            has big trust on the current knowledge and won't learn easily from mistakes.

        :param int epsilon: Explore/Exploit probability, it is the probability that indicates
            whether the agent should explore (act randomly to consider all actions) instead of
            being greedy (taking the action that is known it has the best benefit), a high value
            will make the agent act randomly most often, not making use of the knowledge collected
        """
        self.arms = 2
        self.alpha = alpha
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.action = 0
        self.reward = 0
        self.iteration = 0
        self.total_regret = 0
        self.total_reward = 0
        self.estimations = np.zeros(self.arms)

    def learn(self, action, reward):
        # store
        self.action = action
        self.reward = reward
        self.total_reward += reward
        self.iteration += 1

        # using previous estimations and the outcome of the action taken
        # we calculate a new estimation for the given action
        old_estimate = self.estimations[action]
        error = (reward - self.estimations[action])
        new_estimate = old_estimate + self.alpha * error

        # store
        self.estimations[action] = new_estimate

    def get_action(self):
        if random.random() < self.epsilon:
            # explore
            return np.random.choice(self.arms)
        else:
            # exploit
            return np.argmax(self.estimations)

    def render(self):
        print("Iteration: {}, Action: {}, Reward: {}, Estimations: {}".format(
            self.iteration, self.action, self.reward, self.estimations))
