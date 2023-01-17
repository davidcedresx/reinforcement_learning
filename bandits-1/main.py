from agent import Bandit
import gym_environments  # important
import gym
import sys
from tabulate import tabulate

''' Author = David Cedres -- 27340336

    For this homework the following items will be covered:

    1 - A new epsilon parameter will be added to the Bandit.get_action method
    so that it doesn't always goes with random/greedy actions.

    2 - Since the model is dependent on two hyper params (Alpha, and Epsilon),
    several values for them are tried so that the performance of the model can be measured.

    3 - A performing measurement mechanism will be developed, capturing Total Reward and Regret for each model.
'''

# prepare environment
version = "v0" if len(sys.argv) < 3 else sys.argv[2]
env = gym.make(f"TwoArmedBandit-{version}")


def run_model(alpha, epsilon):
    env.reset(options={'delay': 1})
    agent = Bandit(alpha, epsilon)

    for _ in range(100):
        action = agent.get_action()
        _, reward, _, _, _ = env.step(action)
        agent.learn(action, reward)
        # agent.render()

    return agent


# prepare table
table = []

for alpha in range(0, 10, 1):
    for epsilon in range(0, 10, 1):
        agent = run_model(alpha/10, epsilon/10)
        table.append([alpha/10, epsilon/10, agent.estimations[0],
                     agent.estimations[1], agent.total_reward])

env.close()

table.sort(key=lambda x: -x[4])
subtable = table[0:10]

print(tabulate(subtable, ["Alpha", "Epsilon", "Arm0 Avg", "Arm1 Avg",
      "Collected"], tablefmt="github"))
