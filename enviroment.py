import numpy as np
from game2048 import Game2048
from RLAgent import DQNAgent
from gameInterface import gameInterface
import time
import sys
import pygame
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
import  tensorflow as tf

# Use all available CPU cores
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())  
os.environ["TF_NUM_INTEROP_THREADS"] = str(os.cpu_count())  
os.environ["TF_NUM_INTRAOP_THREADS"] = str(os.cpu_count())  

# Enable optimizations
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.optimizer.set_jit(True)  # Enable XLA


if __name__ == "__main__":
    env = Game2048()
    state_size = env.get_state().shape[0]
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    # model = sorted([file for file in os.listdir(os.getcwd()) if file.endswith("h5") ])[-1]
    # agent.load(model)
    episodes = 3600*3
    interface = gameInterface(env, draw=False)
    
    clock = pygame.time.Clock() if interface.draw else None
    scores, losses, rewards_track = [], [], []
    training_freq = 8
    num_train_cycles = 3
    for episode in tqdm(range(episodes), desc="Episodes"):
        env = Game2048()
        state = env.get_state() # get the initial state# flatten state
        done = False
        step = 0
        while not done:
            if interface.draw :
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if done:
                        break
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            interface.setEnv(env)
            rewards_track.append(reward)
            
            if len(agent.memory) > 128 and step % training_freq == 0:
                for train_cycle in range(num_train_cycles):
                    states, actions, rewards, next_states, dones = agent.sample_memory(128)
                    loss = agent.train_step(states, actions, rewards, next_states, dones)
                    losses.append(loss)
                    
            if step % 100 == 0:
                agent.target_train()
                
            step += 1
            
        scores.append([state.sum(), state.max()])
        agent.reduce_epsilon()
        
        # Print summary every 100 episodes
        if (episode + 1) % 100 == 0:
            recent_scores = np.array(scores[-100:])
            print(f"\n[Episode {episode+1}/{episodes}] Avg Score: {recent_scores[:, 0].mean():.1f}, "
                  f"Avg Max Tile: {recent_scores[:, 1].mean():.1f}, "
                  f"Best Max Tile: {recent_scores[:, 1].max():.0f}, "
                  f"Epsilon: {agent.epsilon:.4f}")
        
    date = time.strftime("%Y-%m-%d_%H-%M-%S")
    agent.save(f"model-{date}.keras")
    print("\nTraining done")
    print("Scores:")
    scores = np.array(scores)
    
    plt.figure()
    plt.plot(scores,label=["Score", "Max"])
    plt.xlabel("Episodes")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(f"results/scores-{date}.png")
    
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("DQN Training Loss")
    plt.savefig(f"results/losses-{date}.png")
    
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_track)
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.title("DQN Training Reward")
    plt.savefig(f"results/rewards-{date}.png")

    pygame.quit()
