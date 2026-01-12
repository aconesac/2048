from RLAgent import DQNAgent
from game2048 import Game2048
from gameInterface import gameInterface
from keras.models import load_model
import numpy as np
import pygame
import sys
import os

model_file = os.listdir(os.getcwd())
model_file = [file for file in model_file if file.endswith(".h5")]
model = load_model(model_file[1])
agent = DQNAgent(16, 4, model=model)
agent_random = DQNAgent(16, 4)

env = Game2048()
state = env.get_state()
state = np.reshape(state, [1, 16])
done = False
interface = gameInterface(env)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if done:
            break
    action = agent_random.act(state)
    state, reward, done = env.step(action)
    interface.setEnv(env)
    pygame.time.wait(50)
    
    if done:
        print("\nNo more moves!")
        print(f"Score: {state.sum()}")
        print(f"Max tile: {state.max()}")
        break
    