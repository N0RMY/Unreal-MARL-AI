import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, action_size):
        self.Q = defaultdict(lambda: np.zeros(action_size))
        self.action_size = action_size
        self.lr = 0.1
        self.gamma = 0.95
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995

    def act(self, state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.Q[state]))

    def learn(self, s, a, r, s2, done):
        best = 0 if done else np.max(self.Q[s2])
        self.Q[s][a] += self.lr * (r + self.gamma * best - self.Q[s][a])

    def end_episode(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
