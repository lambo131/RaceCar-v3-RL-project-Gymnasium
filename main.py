import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from datetime import datetime
import yaml
import msvcrt
# from tqdm import tqdm

# Import our components
from env_wrapper import MyRaceCarEnv
from dqn import DQN
from experience_replay import ReplayMemory

# Hyperparameters (from your YAML) or a dictionary defined here
CONFIG = {
    'MODEL_FILE': "./runs/best_model.pt",
    "GRAPH_FILE": "./runs/history.png",
}

def get_current_time() -> str:
    return datetime.now().strftime("%m-%d %H:%M:%S")

class DQNAgent:
    def __init__(self, config, render_mode='none'):
        # self.config = config
        yaml_path = "./config.yml"
        with open(yaml_path, 'r') as file:
            yaml_all = yaml.safe_load(file)
            self.config = yaml_all[config]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL_FILE = f"./runs/best_model_{config}.pt"
        self.GRAPH_FILE = f"./runs/history_{config}.png"
        
        # Environment setup
        self.env = MyRaceCarEnv(hyperparameter_set="RaceCar_1", render_mode=render_mode)
        config = self.config
        self.config['render_mode'] = render_mode

        # Networks
        self.policy_net = DQN(
            config['state_dim'],
            config['action_dim'],
            config['hidden_nodes'],
            config['hidden_layers']
        ).to(self.device)
        
        self.target_net = DQN(
            config['state_dim'],
            config['action_dim'],
            config['hidden_nodes'],
            config['hidden_layers']
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=config['learning_rate']
        )
        
        # Replay memory
        self.memory = ReplayMemory(config['replay_memory_size'])
        
        # Training state
        self.epsilon = config['initial_epsilon']
        self.epsilon_refresh = 0.3
        self.steps = 0
        self.episode_steps = 0
        self.episode_rewards = []
        self.episode_epsilons = []
    
    def select_action(self, state, epsilon=None):
        if epsilon == None:
            epsilon_ = self.epsilon
        else:
            epsilon_ = epsilon
        if np.random.random() < epsilon_:
            return np.random.randint(self.config['action_dim'])
        
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()
    
    def update_model(self):
        if len(self.memory) < self.config['mini_batch_size']:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config['mini_batch_size']
        )
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Compute Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.config['discount_factor'] * next_q
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def save_best_model(self):
        if len(self.episode_rewards) < 10:
            return
        prev_best = self.episode_rewards[:-1]
        best = self.episode_rewards[-1]
        if best > np.max(prev_best):
            print(f">>> {get_current_time()}: New best reward: {best}")
            print(f"saving model to {self.MODEL_FILE}\n")
            torch.save(self.policy_net.state_dict(), self.MODEL_FILE)
    
    def load_policy_network(self, model_path=None):
        if model_path is None:
            model_path = self.MODEL_FILE
        if os.path.exists(model_path):
            print(f">>> Loading policy network from {model_path}\n")
            self.policy_net.load_state_dict(torch.load(model_path))
        else:  
            print(f"### could not load policy network, {model_path} not found.\n")

    def update_epsilon(self):
        self.epsilon = max(
            self.config['min_epsilon'], 
            self.epsilon * self.config['epsilon_decay']
        )
        if self.epsilon <= self.config['min_epsilon']*1.01 and np.random.random() < 0.001:
            self.epsilon = self.epsilon_refresh
    
    def train(self, num_episodes=1000):
        os.makedirs('./runs', exist_ok=True)
        print(f">>> {get_current_time()}: Starting training for {self.config['env_name']}\n")
        current_render_en = self.config['render_mode']
        for episode in range(num_episodes):
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b'r':  # ESC key
                    if current_render_en == 'none':
                        current_render_en = self.config['render_mode']
                    elif current_render_en == self.config['render_mode']:
                        current_render_en = 'none'
                    self.env.close()
                    self.env = MyRaceCarEnv(hyperparameter_set="RaceCar_1", render_mode=current_render_en)

            state = self.env.initialize_env()
            
            total_reward = 0
            self.episode_steps = 0
            done = False
            episode_start = time.time()
            
            while not done:
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store in memory
                self.memory.push(state, action, reward, next_state, done)
                
                # Move to next state
                state = next_state
                total_reward += reward
                self.steps += 1
                self.episode_steps += 1
                
                # Update model
                self.update_model()
                
                # Update target network
                if self.steps % self.config['target_sync_freq'] == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                # Update exploration rate
                self.update_epsilon()
            
            # Record metrics
            self.episode_rewards.append(total_reward)
            self.episode_epsilons.append(self.epsilon)

            # Print progress
            fps = self.episode_steps / (time.time() - episode_start)
            print(f"Ep {episode+1}/{num_episodes} | "
                  f"Reward: {total_reward:.1f} | "
                  f"Epsilon: {self.epsilon:.4f} | "
                  f"Steps: {self.steps} | FPS: {fps:.1f}")
            
            # updates
            self.save_best_model()

            # Plot and save every 10 episodes
            if (episode + 1) % 1 == 0:
                self.save_history()
                
            # Early stopping
            if (len(self.episode_rewards) >= 10 and 
                np.mean(self.episode_rewards[-10:]) >= self.config['train_stop_reward']):
                print(f"Training complete! Average reward reached {self.config['train_stop_reward']}")
                break
        

    def run(self, num_episodes=1, epsilon=0.001):
        print(f">>> {get_current_time()}: Starting run() for {self.config['env_name']}\n")
        # self.env.close()
        # self.env = MyRaceCarEnv(hyperparameter_set="RaceCar_1", render_mode=self.config['render_mode'])
        run_memory = ReplayMemory(10000)
        episode_start = time.time()

        for episode in range(num_episodes):
            state = self.env.initialize_env()
            steps = 0
            episode_reward = 0
            done = False
            episode_start = time.time()
            while not done:
                action = self.select_action(state, epsilon=epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                run_memory.push(state, action, reward, next_state, done)
                steps += 1
                episode_reward += reward
                state = next_state

            self.episode_steps+=steps
            self.episode_rewards.append(episode_reward)
            self.episode_epsilons.append(epsilon)

            fps = self.episode_steps / (time.time() - episode_start)
            print(f"Ep {episode+1}/{num_episodes} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Epsilon: {epsilon:.4f} | "
                  f"Steps: {steps} | FPS: {fps:.1f}")

        print(f"run completed...")
        print(f"\t>>> run_mem_len: {len(run_memory)} | best_reward: {np.max(self.episode_rewards)}\n")
        run_data = {
            'episode_steps': self.episode_steps,
            'episode_rewards': self.episode_rewards,
            'epsilon_history': self.episode_epsilons,
            'run_memory': run_memory
        }
        return run_data
    
    def save_history(self):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_epsilons)
        plt.title("Epsilon Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon Value")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.GRAPH_FILE)
        plt.close()


import argparse

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Train or run DQN agent')
    parser.add_argument('config', type=str, help='name of config')
    parser.add_argument('render_mode', type=str, default='none', help='none, opencv, or jupyter')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--run', action='store_true', help='Run the agent')
    # parser.add_argument("--load_exp", action='store_true', help='Load experience from file')
    parser.add_argument("--load_policy", action='store_true', help='Load policy network from file')
    args = parser.parse_args()

    agent = DQNAgent(config=args.config, render_mode=args.render_mode)
    if args.train:
        # Train the agent
        if args.load_policy:
            agent.load_policy_network()
        agent.train()

    elif args.run:
        # Run the agent
        if args.load_policy:
            agent.load_policy_network()
        agent.run(num_episodes=5)