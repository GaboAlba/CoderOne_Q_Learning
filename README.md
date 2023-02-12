# CoderOne Q_Learning Agent 
  
## What is CoderOne?

CoderOne is a competition where participants develop a variety of AI agents to play the game Bomberman aginst one enemy(normally a human player or another AI agent). In this particular case this agent would be being developed for the Dungeons-and-data-structures Python version of this competition. 

## How to run this agent?
 Please follow the instructions in the following link to install the game and create your virtual environment. 
 
 https://www.notion.so/Getting-Started-438550daf0234e3fa53e8179ea00066f
 
 After this is done three things must be copied inside the following directory `~\venv\Lib\site-packages\coderone\dungeon`. 
  1. `RL_Agent folder`
  2. `IntelAgent.py` (Right now it only does random movements, as it is being used to test the game state reads for accuracy)
  3. `my_agent.py` (This one a sample agent that does random movements to test different functionalities)

After this is done, inside the virtual environment, and inside the `dungeon` directory run the following command to test it

`python -m coderone.dungeon.main my_agent.py IntelAgent.py`

## Current state of the development
The development of this agent consists in the following stages
 1. [x] Testing movement and ability interaction with the game 
 2. [x] Read the game state and normalize the input as numbers
 3. [ ] Read the player state and get the ammo and reward from it 
 4. [x] Develop the Neural Network constructor
 5. [x] Develop the Neural Network training logic
 6. [ ] Develop the optimizing algorithm
 7. [ ] Test all logic and algorithms and correct as needed 
 8. [ ] Train the Neural Network 
 9. [ ] Save the NN and use as the sole agent against different AI's to measure performance and robustness

## What the agent is based on?
This agent is being developed as Machine Learning agent that will use Q_Learning in order to learn how to play the game. As part of the training it will play against other AI's developed for this tournament. Specified in the following link: https://github.com/CoderOneHQ/dungeons-and-data-structures

### I/O of the agent. 
This agent will take the following information as input: 
 * Game State
 * Player State

And it will produce as an output the corresponding action for the current state of the game. 

### Reward
In order to keep the reward as simple as possible, the score of the player in proportion to the enemy's score will be used, basically maximizing the score as well as minimizing the opponents score. In this way, the agent will try to win as fast as possible. For the corner case of the opponent getting a score of 0 a bonus reward will be given to prevent a division by zero.
Also, as part of the reward, a bonus will be given at the end of the game if the agent won. And a penalty will be dealt if it lost. In this way not only maximizing the score but also looking to get into a winning condition. 
