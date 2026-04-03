from agent import QLearningAgent
from env import DummyEnv

def discretize(obs):
    dx, dy, dist = obs["dx"], obs["dy"], obs["dist"]
    direction = "R" if dx > 0 else "L" if dx < 0 else "C"
    near = "N" if dist < 500 else "F"
    return (direction, near)

env = DummyEnv()
agent = QLearningAgent(action_size=5)

for ep in range(100):
    obs = env.reset()
    state = discretize(obs)
    done = False

    while not done:
        a = agent.act(state)
        obs2, r, done, _ = env.step(a)
        s2 = discretize(obs2)
        agent.learn(state, a, r, s2, done)
        state = s2

    agent.end_episode()
    print(f"Episode {ep} done")
