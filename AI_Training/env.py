import random

class DummyEnv:
    def reset(self):
        self.steps = 0
        return {"dx": random.randint(-5,5), "dy": random.randint(-5,5), "dist": random.randint(0,1000), "role":"seeker"}

    def step(self, action):
        self.steps += 1
        obs = {"dx": random.randint(-5,5), "dy": random.randint(-5,5), "dist": random.randint(0,1000), "role":"seeker"}
        reward = random.random()
        done = self.steps > 50
        return obs, reward, done, {}
