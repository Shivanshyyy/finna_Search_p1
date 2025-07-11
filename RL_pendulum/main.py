import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt


# First, we created a custom environment that simulates the pendulum's motion using basic physics.
class InvertedPendulumEnv:
    def __init__(self):  # Initialize environment parameters
        self.dt = 0.02  # Time step
        self.g = 9.81   # Gravty
        self.l = 1.0    # Length of pendulum
        self.m = 1.0    # Mass of pendulum
        self.max_torque = 3.0  # Maximum torque that can be applied
        self.reset()
    

    def reset(self):
        # Start with random angle between -π and π and zero angular velocity
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.theta_dot = 0.0
        return self.get_state()
    
    def get_state(self):
        # To represent the state, we use the cosine and sine of the angle — instead of the angle itself — 
        # along with the angular velocity. This avoids weird jumps at 360°, making learning smooth.
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot])
    
    def step(self, action):
        # Convert discrete action (0, 1, 2) into torque (-max, 0, +max)
        torque = (action - 1) * self.max_torque
        # Apply pendulum dynamics: θ'' = (torque - m*g*l*sin(θ)) / (m*l²)
        theta_ddot = (torque - self.m * self.g * self.l * np.sin(self.theta)) / (self.m * self.l**2)

        # Update angular velocity and angle using Euler integration
        self.theta_dot += theta_ddot * self.dt
        self.theta += self.theta_dot * self.dt

        # Normalize angle to stay within [-π, π]
        self.theta = ((self.theta + np.pi) % (2 * np.pi)) - np.pi

        # Reward is high for staying near upright with low velocity, penalized otherwise
        reward = -abs(self.theta) - 0.1 * abs(self.theta_dot)
        if abs(self.theta) < 0.1 and abs(self.theta_dot) < 0.1:
            reward += 10  # Bonus reward for near-perfect balance

        # Episode ends if pendulum falls too far from vertical
        done = abs(self.theta) > np.pi / 2
        return self.get_state(), reward, done


# Next comes the DQN agent itself. We designed a neural network with:
# - Input layer matching the state size
# - Hidden layer with ReLU
# - Output layer predicting Q-values for all 3 actions
class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0  # Start fully exploratory
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.gamma = 0.95  # Discount factor
        self.memory = deque(maxlen=2000)  # Experience replay buffer

        # Initialize network weights (manual network)
        self.weights_input_hidden = np.random.randn(state_size, 64) * 0.1
        self.weights_hidden_output = np.random.randn(64, action_size) * 0.1
        self.bias_hidden = np.zeros(64)
        self.bias_output = np.zeros(action_size)
    
    def relu(self, x):
        # ReLU activation function
        return np.maximum(0, x)
    
    def forward(self, state):
        # Forward pass through the network: input → hidden → output
        hidden = self.relu(np.dot(state, self.weights_input_hidden) + self.bias_hidden)
        output = np.dot(hidden, self.weights_hidden_output) + self.bias_output
        return output, hidden


    def remember(self, state, action, reward, next_state, done):
        # Experience replay: store the (s, a, r, s', done) tuple
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        # Epsilon-greedy strategy: explore with probability ε, exploit otherwise
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values, _ = self.forward(state)
        return np.argmax(q_values)
    
    def replay(self, batch_size=32):
        # Train the network using randomly sampled past experiences
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                q_next, _ = self.forward(next_state)
                target += self.gamma * np.amax(q_next)  # Temporal Difference target

            q_values, hidden = self.forward(state)
            error = target - q_values[action]  # TD error

            # Manual gradient descent update (simplified)
            self.weights_hidden_output[:, action] += self.learning_rate * error * hidden
            self.bias_output[action] += self.learning_rate * error
        
        # Slowly reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



# Training loop for DQN
def train_dqn():
    env = InvertedPendulumEnv()
    agent = DQN(3, 3)  # State has 3 dimensions; 3 actions possible
    episodes = 1000
    scores = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(200):  # Limit number of steps per episode
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.replay()  # Train from experience buffer
        scores.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    # Plot total reward per episode
    plt.plot(scores)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show()

    return scores



# Entry point for script execution
if __name__ == "__main__":
    print("Starting training loop...")
    train_dqn()
