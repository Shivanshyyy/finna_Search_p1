import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class InvertedPendulumEnv:
    def _init_(self):
        self.dt = 0.02  # time step
        self.g = 9.81   # gravity
        self.l = 1.0    # pendulum length
        self.m = 1.0    # pendulum mass
        self.max_torque = 3.0
        self.reset()
    
    def reset(self):
        # Start pendulum at random angle
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.theta_dot = 0.0
        return self.get_state()
    
    def get_state(self):
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot])
    
    def step(self, action):
        # Convert action to torque (0: left, 1: none, 2: right)
        torque = (action - 1) * self.max_torque
        
        # Pendulum dynamics
        theta_ddot = (torque - self.m * self.g * self.l * np.sin(self.theta)) / (self.m * self.l**2)
        
        # Update state
        self.theta_dot += theta_ddot * self.dt
        self.theta += self.theta_dot * self.dt
        
        # Normalize angle to [-π, π]
        self.theta = ((self.theta + np.pi) % (2 * np.pi)) - np.pi
        
        # Calculate reward
        reward = -abs(self.theta) - 0.1 * abs(self.theta_dot)
        
        # Bonus for balancing upright
        if abs(self.theta) < 0.1 and abs(self.theta_dot) < 0.1:
            reward += 10
        
        done = False
        return self.get_state(), reward, done

class DQN:
    def _init_(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.memory = deque(maxlen=2000)
        
        # Neural network weights (simplified)
        self.weights_input_hidden = np.random.randn(state_size, 64) * 0.1
        self.weights_hidden_output = np.random.randn(64, action_size) * 0.1
        self.bias_hidden = np.zeros(64)
        self.bias_output = np.zeros(action_size)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, state):
        hidden = self.relu(np.dot(state, self.weights_input_hidden) + self.bias_hidden)
        output = np.dot(hidden, self.weights_hidden_output) + self.bias_output
        return output
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.forward(state)
        return np.argmax(q_values)
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.forward(next_state))
            
            target_f = self.forward(state)
            target_f[action] = target
            
            # Simplified weight update (in practice, use backpropagation)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training loop example
def train_dqn():
    env = InvertedPendulumEnv()
    agent = DQN(3, 3)  # 3 state dimensions, 3 actions
    
    episodes = 1000
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(200):  # max steps per episode
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        agent.replay()
        scores.append(total_reward)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return scores

