from google.colab import drive
import imageio
import os
from vizdoom import *
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from datetime import datetime
import glob

# --- Configuration ---
RECORD_LMP = False       # Set to True to record .lmp demo files
RECORD_VIDEO = False     # Set to True to record a video (MP4)
VIDEO_FPS = 30          # Frames per second for video recording
LMP_DIR = "lmp_recordings"  # Directory to save .lmp files
SCENARIO_NAME = "defend_the_center.cfg"  # Change the scenario name here
# --- Google Drive Paths ---
DRIVE_MODEL_DIR = "/content/drive/My Drive/ViZDoomModels"  # Directory in Google Drive to save the model

# --- User Input for Recording, Number of Episodes, and FRAME_SKIP ---
record_choice = input("Do you want to record the game? (yes/no): ").lower()
if record_choice == "yes":
    record_lmp_choice = input("Record .lmp demo files? (yes/no): ").lower()
    RECORD_LMP = record_lmp_choice == "yes"
    record_video_choice = input("Record video (MP4)? (yes/no): ").lower()
    RECORD_VIDEO = record_video_choice == "yes"

num_episodes_choice = input("Enter the number of episodes to run: ")
try:
    NUM_EPISODES = int(num_episodes_choice)
except ValueError:
    print("Invalid input. Using default number of episodes (10).")
    NUM_EPISODES = 10

# User input for FRAME_SKIP during training and recording
FRAME_SKIP_TRAINING = int(input("Enter FRAME_SKIP value for training (default 4): ") or "4")
FRAME_SKIP_RECORDING = int(input("Enter FRAME_SKIP value for recording (default 1): ") or "1")
FRAME_SKIP = FRAME_SKIP_TRAINING if not RECORD_VIDEO else FRAME_SKIP_RECORDING

# --- Google Drive Setup ---
drive_mounted = os.path.exists('/content/drive/My Drive')
if not drive_mounted:
    drive.mount('/content/drive')

VIDEO_DIR = "/content/drive/My Drive/ViZDoomRecordings"
VIDEO_FILENAME = "game_recording.mp4"
VIDEO_PATH = os.path.join(VIDEO_DIR, VIDEO_FILENAME)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(DRIVE_MODEL_DIR, exist_ok=True)  # Create model directory in Drive if it doesn't exist

# --- DQN ---
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.get_conv_output(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_conv_output(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_shape, num_actions, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay, memory_capacity, batch_size):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_shape, num_actions).to(self.device)
        self.target_net = DQN(state_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(memory_capacity)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state)
                return q_values.argmax(dim=1).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = tuple(zip(*transitions))

        state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch[1], dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch[4], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# --- Training Parameters ---
LEARNING_RATE = 0.0001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10
# NUM_EPISODES is set via user input earlier
# FRAME_SKIP values are set via user input earlier

# --- ViZDoom Setup ---
game = DoomGame()
import vizdoom
vizdoom_path = os.path.dirname(vizdoom.__file__)
SCENARIO_PATH = os.path.join(vizdoom_path, "scenarios", SCENARIO_NAME)
game.load_config(SCENARIO_PATH)
game.set_window_visible(False)
game.set_screen_format(ScreenFormat.RGB24)
game.set_screen_resolution(ScreenResolution.RES_320X240)

if RECORD_LMP:
    game.set_mode(Mode.PLAYER)
    game.set_doom_scenario_path(SCENARIO_PATH)
    os.makedirs(LMP_DIR, exist_ok=True)

game.init()
num_actions = game.get_available_buttons_size()

# --- One-hot encoding for actions ---
actions = np.identity(num_actions, dtype=int).tolist()

# --- Get initial state shape ---
screen_height, screen_width = game.get_screen_height(), game.get_screen_width()
channels = game.get_screen_channels()
state_shape = (channels, screen_height, screen_width)

# --- Scan for existing models ---
model_files = glob.glob(os.path.join(DRIVE_MODEL_DIR, "*.pth"))

if model_files:
    print("Available models:")
    for i, file in enumerate(model_files):
        print(f"{i+1}. {os.path.basename(file)}")
    print(f"{len(model_files)+1}. Create new model")

    choice = input("Enter your choice: ")
    try:
        choice = int(choice)
        if choice in range(1, len(model_files) + 2):
          if choice <= len(model_files):
            # Load existing model
            model_path = model_files[choice - 1]
            agent = DQNAgent(state_shape, num_actions, LEARNING_RATE, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY, MEMORY_CAPACITY, BATCH_SIZE)
            agent.policy_net.load_state_dict(torch.load(model_path))
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"Model loaded from {model_path}")
          else:
            # --- Create Agent ---
            agent = DQNAgent(state_shape, num_actions, LEARNING_RATE, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY, MEMORY_CAPACITY, BATCH_SIZE)
            print("Creating new model...")
        else:
            print("Invalid choice. Creating new model...")
            # --- Create Agent ---
            agent = DQNAgent(state_shape, num_actions, LEARNING_RATE, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY, MEMORY_CAPACITY, BATCH_SIZE)
    except ValueError:
        print("Invalid input. Creating new model...")
        # --- Create Agent ---
        agent = DQNAgent(state_shape, num_actions, LEARNING_RATE, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY, MEMORY_CAPACITY, BATCH_SIZE)
else:
    print("No existing models found. Creating new model...")
    # --- Create Agent ---
    agent = DQNAgent(state_shape, num_actions, LEARNING_RATE, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY, MEMORY_CAPACITY, BATCH_SIZE)

# --- Video Writer ---
if RECORD_VIDEO:
    writer = imageio.get_writer(VIDEO_PATH, fps=VIDEO_FPS)

# --- Tracking Variables ---
episode_rewards = []  # List to store rewards per episode
episode_lengths = []  # List to store the number of steps per episode
training_start_time = time.time()

# --- Training Loop ---
TICRATE = 35 # Internal ticrate of ViZDoom (for syncing with video FPS, if needed)

for episode in range(NUM_EPISODES):
    if RECORD_LMP:
        lmp_file_path = os.path.join(LMP_DIR, f"episode_{episode+1}.lmp")
        game.new_episode(lmp_file_path)
    else:
        game.new_episode()

    state = game.get_state().screen_buffer
    state = np.transpose(state, (2, 0, 1))
    total_reward = 0
    step_count = 0

    while not game.is_episode_finished():
        action_index = agent.select_action(state)
        action = actions[action_index]

        # Make the action with the appropriate FRAME_SKIP value
        reward = game.make_action(action, FRAME_SKIP) 
        done = game.is_episode_finished()

        if not done:
            next_state = game.get_state().screen_buffer
            next_state = np.transpose(next_state, (2, 0, 1))
        else:
            next_state = np.zeros(state_shape)

        agent.memory.push(state, action_index, reward, next_state, done)
        agent.learn()

        state = next_state
        total_reward += reward
        step_count += 1

        if RECORD_VIDEO and state is not None:
            writer.append_data(state.transpose(1, 2, 0))

        if done:
            break

    episode_rewards.append(total_reward)
    episode_lengths.append(step_count)
    agent.update_epsilon()
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.update_target_network()

    # --- Logging ---
    avg_reward = np.mean(episode_rewards[-100:])
    print(f"Episode {episode+1}/{NUM_EPISODES}, Total Reward: {total_reward}, Steps: {step_count}, Epsilon: {agent.epsilon:.3f}, Avg Reward (last 100): {avg_reward:.2f}")

# --- Get current date and time for file name prefix ---
now = datetime.now()
dt_string = now.strftime("%Y%m%d%H%M")  # Format: YYYYMMDDHHMM

# --- Save metrics to CSV ---
training_end_time = time.time()
metrics_df = pd.DataFrame({
    'episode': range(1, NUM_EPISODES + 1),
    'reward': episode_rewards,
    'steps': episode_lengths,
    'epsilon': [EPSILON_START * (EPSILON_DECAY ** i) for i in range(NUM_EPISODES)],
    'training_time': training_end_time - training_start_time
})

metrics_filename = f"{dt_string}_training_metrics.csv"
metrics_path = os.path.join(DRIVE_MODEL_DIR, metrics_filename)
metrics_df.to_csv(metrics_path, index=False)

# --- Save the trained model ---
model_filename = f"{dt_string}_dqn_model.pth"
model_path = os.path.join(DRIVE_MODEL_DIR, model_filename)
torch.save(agent.policy_net.state_dict(), model_path)

# Close the writer
if RECORD_VIDEO:
    writer.close()

game.close()
print("Done!")