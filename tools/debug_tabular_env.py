import sys, os
# Ensure repo root is on path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mario_rl_env.ml_agents import TabularQAgent, run_training_loop
import gymnasium as gym
import numpy as np

class ToyEnv(gym.Env):
    def __init__(self, n_states=3, max_steps=5):
        self.n_states = n_states
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Box(low=0, high= n_states-1, shape=(1,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(n_states)
        self._state = 0
        self._steps = 0

    def reset(self):
        self._state = 0
        self._steps = 0
        return np.array([self._state], dtype=np.int32), {}

    def step(self, action):
        # reward = 1 if action equals current state, else 0
        reward = 1.0 if int(action) == int(self._state) else 0.0
        self._steps += 1
        # advance state
        self._state = (self._state + 1) % self.n_states
        terminated = self._steps >= self.max_steps
        truncated = False
        return np.array([self._state], dtype=np.int32), float(reward), terminated, truncated, {}

if __name__ == '__main__':
    agent = TabularQAgent(action_size=3, alpha=0.5, epsilon=0.5, epsilon_decay=0.99)
    history = run_training_loop(lambda: ToyEnv(), agent, n_episodes=200, batch_size=1, debug=True, verbose=True)
    print('Done. Last 5 episodes:')
    print(history[-5:])
