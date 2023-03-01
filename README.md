# CoderOne Q_Learning Agent 
  
## What is CoderOne?

CoderOne is a competition where participants develop a variety of AI agents to play the game Bomberman aginst one enemy(normally a human player or another AI agent). In this particular case this agent would be being developed for the Dungeons-and-data-structures Python version of this competition. 

## How to run this agent?
 
 ##### *If you want to run without optimizing, and only with the trained NN, be sure to set EPS_START and EPS_END to ZERO, and comment out the `self.optimize_model` in line 664*
 
 Please follow the instructions in the following link to install the game and create your virtual environment. That `.gz` file can be found as part of this repository
 
 https://www.notion.so/Getting-Started-438550daf0234e3fa53e8179ea00066f
 
 After this is done seven things must be copied inside the following directory `~\venv\Lib\site-packages\coderone\dungeon`. 
  1. `RunTrainingLoops.py`
  2. `counter.py`
  3. `IntelAgent.py` 
  4. `CompetentBot2.py`
  5. `my_agent.py` (This one a sample agent that does random movements to test different functionalities)
  6. `target_net.pth`
  7. `policy_net.pth`

After this is done, inside the virtual environment, and inside the `dungeon` directory run the following command to install all dependencies

`pip install requirements.txt`

Once all dependencies are sorted, please run this command to make sure everything is working right.

`python -m coderone.dungeon.main my_agent.py IntelAgent.py`

You should see the Board Matrix, the player ammo quantity and the player reward being printed in the terminal, as well as the knight move do random actions in game.


## Current state of the development
The development of this agent consists in the following stages
 1. [x] Testing movement and ability interaction with the game 
 2. [x] Read the game state and normalize the input as numbers
 3. [x] Read the player state and get the ammo and reward from it 
 4. [x] Develop the Neural Network constructor
 5. [x] Develop the Neural Network training logic
 6. [x] Develop the optimizing algorithm
 7. [x] Test all logic and algorithms and correct as needed 
 8. [x] Train the Neural Network 
 9. [x] Save the NN and use as the sole agent against different AI's to measure performance and robustness

## What the agent is based on?
This agent is being developed as Machine Learning agent that will use Q_Learning in order to learn how to play the game. As part of the training it will play against other AI's developed for this tournament. Specified in the following link: https://github.com/CoderOneHQ/dungeons-and-data-structures

### I/O of the agent. 
This agent will take the following information as input: 
 * Game State
 * Player State
 * Enemy State

And it will produce as an output the corresponding action for the current state of the game. 

### Reward
The reward is based on maximizing the amount of eliminations the agent obtains, and minimizing the amount of deaths. Also rewarding being close to the enemy when ammo is available and punishing standing close to a bomb. 
