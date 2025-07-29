import numpy as np
import matplotlib.pyplot as plt

# Initialize racetrack environment
def create_track(width=10, height=10):
    track = np.zeros((height, width))
    track[1:-1, 1:-1] = 1  # drivable area
    start = (1, 1)
    finish = (height - 2, width - 2)
    return track, start, finish

def reset():
    return (1, 1)

def step(track, position, action):
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    new_pos = (position[0] + moves[action][0], position[1] + moves[action][1])

    if (0 <= new_pos[0] < track.shape[0] and
        0 <= new_pos[1] < track.shape[1] and
        track[new_pos] == 1):
        position = new_pos

    done = position == (track.shape[0] - 2, track.shape[1] - 2)
    reward = 1 if done else -0.1

    return position, reward, done

def render(track, position, finish):
    track_vis = track.copy()
    track_vis[position] = 2
    track_vis[finish] = 3
    plt.imshow(track_vis, cmap='gray')
    plt.show()

# Initialize Q-learning agent
def initialize_q_table(track):
    return np.zeros((track.shape[0], track.shape[1], 4))

def choose_action(q_table, state, epsilon=0.2):
    if np.random.rand() < epsilon:
        return np.random.randint(4)
    return np.argmax(q_table[state[0], state[1]])

def learn(q_table, state, action, reward, next_state, alpha=0.6, gamma=0.9):
    predict = q_table[state[0], state[1], action]
    target = reward + gamma * np.max(q_table[next_state[0], next_state[1]])
    q_table[state[0], state[1], action] += alpha * (target - predict)

# Training function
def train_agent(episodes=200):
    track, start, finish = create_track()
    q_table = initialize_q_table(track)

    for episode in range(episodes):
        state = reset()
        done = False

        while not done:
            action = choose_action(q_table, state)
            next_state, reward, done = step(track, state, action)
            learn(q_table, state, action, reward, next_state)
            state = next_state

    return track, q_table, finish

# Demonstration
track, q_table, finish = train_agent()

# Visualize trained agent
state = reset()
done = False

while not done:
    render(track, state, finish)
    action = choose_action(q_table, state)
    state, _, done = step(track, state, action)

render(track, state, finish)
