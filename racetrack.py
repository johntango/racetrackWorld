import numpy as np
import matplotlib.pyplot as plt

# Define a simple 2D racetrack environment
class RaceTrack:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.track = np.zeros((height, width))
        self.track[1:-1, 1:-1] = 1  # drivable area
        self.car_position = (1, 1)
        self.finish = (height - 2, width - 2)

    def reset(self):
        self.car_position = (1, 1)
        return self.car_position

    def step(self, action):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        new_pos = (self.car_position[0] + moves[action][0], self.car_position[1] + moves[action][1])

        if (0 <= new_pos[0] < self.height and
            0 <= new_pos[1] < self.width and
            self.track[new_pos] == 1):
            self.car_position = new_pos

        done = self.car_position == self.finish
        reward = 1 if done else -0.1

        return self.car_position, reward, done

    def render(self):
        track_vis = self.track.copy()
        track_vis[self.car_position] = 2
        track_vis[self.finish] = 3
        plt.imshow(track_vis, cmap='gray')
        plt.show()

# Simple Q-learning Agent
class Agent:
    def __init__(self, track):
        self.track = track
        self.q_table = np.zeros((track.height, track.width, 4))
        self.epsilon = 0.2
        self.alpha = 0.6
        self.gamma = 0.9

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        return np.argmax(self.q_table[state[0], state[1]])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state[0], state[1], action]
        target = reward + self.gamma * np.max(self.q_table[next_state[0], next_state[1]])
        self.q_table[state[0], state[1], action] += self.alpha * (target - predict)

# Train the agent
def train_agent(episodes=200):
    track = RaceTrack()
    agent = Agent(track)

    for episode in range(episodes):
        state = track.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = track.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state

    return track, agent

# Demonstration
track, agent = train_agent()

# Visualize trained agent
state = track.reset()
done = False

while not done:
    track.render()
    action = agent.choose_action(state)
    state, _, done = track.step(action)

track.render()