"""python3 -m venv venv
venv\Scripts\activate   """

import random
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import os 
import time
from math import sqrt, pow


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from coderone.dungeon import run_match, Game, GameStats, counter

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory) 

class DQN(nn.Module) :

    def __init__(self,n_inputs, n_outputs) :
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(self.n_inputs, 80)
        self.layer2 = nn.Linear(80,40)
        self.layer3 = nn.Linear(40,40)
        self.layer4 = nn.Linear(40,self.n_outputs)
        

    def forward(self, x):
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        x = F.tanh(self.layer3(x)) 
        return self.layer4(x)
    
class NeuralNetwork :
    

    global Board 
    Board= [[0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0]]

    def __init__(self) :
        self.Board = Board
        self.XYBoard = Board
        self.enemyX = 0
        self.enemyY = 0

    def GetState(self,game_state, player_state, stats) :
        ## Player State and Game State must be read as input to the NN
        ## GAME STATE READING
        self.game_state = game_state
        self.player_state = player_state
        self.Board =   [[0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0]]
        if not isinstance(self.game_state,dict)  :
            self.game_state = game_state.__dict__
        else :
            pass
        if not isinstance(self.player_state,dict) :
            self.player_state = player_state.__dict__
        else :
            pass

        Xkeys = self.game_state['_game_map'].keys()
        for x in Xkeys :
            for y in self.game_state['_game_map'][x] :
                state = self.game_state['_game_map'][x][y]
                # 0-10 codification to make it easy to normalize as NN inputs 
                match state :
                    case "a" :
                        state = 1/10
                    case "t" :
                        state = 2/10
                    case "b" :
                        state = 3/10
                    case "sb" :
                        state = 4/10
                    case "ob" :
                        state = 5/10
                    case "ib" :
                        state = 6/10
                    case 0 :
                        if self.player_state["id"] == 1:
                            #print("ID-0: ", self.player_state["id"])
                            state = 7/10
                            self.enemyX = x
                            self.enemyY = y
                        else :
                            state = 0
                    case 1 :
                        if self.player_state["id"] == 0:
                            #print("ID-1: ", self.player_state["id"])
                            self.enemyX = x
                            self.enemyY = y
                            state = 7/10
                        else :
                            state = 0
                    case other :
                        state = 0
                self.Board[x][y] = state
                self.XYBoard[x][y] = state
                
        tempBoard = [0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0]
        counter = 0
        for x in range(0,12) :
            for y in range(0,10) :
                tempBoard[counter] = self.Board[x][y]
                counter += 1

        self.Board = tempBoard
                
        ## PLAYER STATE READING

        self.ammo = self.player_state["ammo"]
        self.reward = self.player_state["reward"]
        self.terminated = self.game_state["is_over"]
        self.position =  self.player_state["location"]
        self.hp = self.player_state["hp"]
        self.winner = GameStats.winner_pid
        self.Board.append(self.position[0]/12)
        self.Board.append(self.position[1]/10)
        self.Board.append(self.ammo/3)

        ## ENEMY STATE READING
        self.enemyPosition = (self.enemyX,self.enemyY)
        self.Board.append(self.enemyPosition[0]/12)
        self.Board.append(self.enemyPosition[1]/10)

        ## Get Distance to enemy
        xDifference = abs(self.position[0] - self.enemyPosition[0])
        yDifference = abs(self.position[1] - self.enemyPosition[1])
        distanceToEnemy = sqrt(pow(xDifference,2) + pow(yDifference,2))

        return  self.Board, self.winner, self.reward, self.terminated, self.hp, distanceToEnemy

class Agent :

    def __init__(self):
        ## TEST MODE

        self.testing = False
        
        # Setting up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()
        
        # GPU Capability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))
        
        self.BATCH_SIZE = 64
        self.GAMMA = 0.99
        self.EPS_START = 0.4
        self.EPS_END = 0.05
        self.EPS_DECAY = 500
        self.TAU = 0.005
        self.LR = 0.005
        # Inputs are every single cell, X and Y for player and enemy and our ammo
        self.n_inputs = 125
        self.n_outputs = 6
        # Defining Neural Networks
        self.policy_net = DQN(self.n_inputs,self.n_outputs).to(self.device)
        self.target_net = DQN(self.n_inputs,self.n_outputs).to(self.device)
        # Loading pre-existent NN
        if os.path.exists("venv\Lib\site-packages\coderone\dungeon\policy_net.pth") :
            self.policy_net.load_state_dict(torch.load("policy_net.pth"))
        else :
            pass
        if os.path.exists("venv\Lib\site-packages\coderone\dungeon\target_net.pth") :
            self.target_net.load_state_dict(torch.load("target_net.pth"))
        else :
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = self.LR, amsgrad = True)
        self.memory = ReplayMemory(10000)
        self.steps_done = counter.globalCounter
        self.actions = ['','u','d','l','r','p']
        self.actions = [0,1,2,3,4,5]
        self.episode_durations = []
        self.NN = NeuralNetwork()
        self.pastHP = 3
        self.pastReward = 3
        self.game = Game()
        self.TotalReward = 0
        self.pastTotalReward = 0
    
    def randomAction(self,state) :

        player_location_x = int(state[len(state)-5]*12)
        player_location_y = int(state[len(state)-4]*10)
        ammo = state[len(state)-3]

        if player_location_x > 0 :
            leftTile = state[10*player_location_x + player_location_y - 10]
        else :
            leftTile = 99
        if player_location_x < 11 :
            rightTile = state[10 * player_location_x + player_location_y + 10]
        else :
            rightTile = 99
        if player_location_y < 9 :
            upTile = state[10 * player_location_x + player_location_y + 1]
        else :
            upTile = 99
        if player_location_y > 0 :
            downTile = state[10 * player_location_x + player_location_y - 1]
        else :
            downTile = 99
        ammo = state[len(state)-3]        
        self.actions = [0]
        
        ## UP MOVES
        if upTile >= 4/10 and upTile <= 7/10 or upTile == 99  :
            pass
        else :
            self.actions.append(1)
            self.actions.append(1)
            self.actions.append(1)

        ## DOWN MOVES
        if downTile >= 4/10 and downTile <= 7/10 or downTile == 99:
            pass 
        else :
            self.actions.append(2)
            self.actions.append(2)
            self.actions.append(2)
        
        ## LEFT MOVES
        if leftTile >= 4/10 and leftTile <= 7/10 or leftTile == 99 :
            pass 
        else :
            self.actions.append(3)
            self.actions.append(3)
            self.actions.append(3)
        
        ## RIGHT MOVES
        if rightTile >= 4/10 and rightTile <= 7/10 or rightTile == 99 :
            pass 
        else :
            self.actions.append(4)
            self.actions.append(4)
            self.actions.append(4)

        ## AMMO AVAILABLE
        if ammo > 0 :
            self.actions.append(5)
        else :
            pass

        randomNumber = random.randint(0,len(self.actions)-1)
        return torch.tensor([[self.actions[randomNumber]]], device=self.device, dtype=torch.long)


    def select_action(self,state,randomActive=False)  :
        global steps_done
        sample = random.random()
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        stateList = state
        state = torch.tensor([state])
        if not randomActive :
            if sample >= self.eps_threshold :
                with torch.no_grad() :
                    action = self.policy_net(state).max(1)[1].view(1,1)
                    return action
            
            else :

                return self.randomAction(stateList)
        else :

                return self.randomAction(stateList)
                

    # FUNCTION TAKEN FROM PYTORCH
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def next_move(self, game_state, player_state):
        
        ## TEST validAction functionality. Helps determine problems with data structures
        def testInput(game_state, player_state, Board) :
            playerX = int(Board[len(Board)-5]*12)
            playerY = int(Board[len(Board)-4]*10)
            playerLocation = 10*playerX + playerY
            for action in range(0,6):
                print("VALID ", str(action), ": ",validAction(game_state, player_state, action, Board))
            print("ABS LOC: ", playerLocation)
            return random.randint(0,5)
        
        def getEightNeighbors(player_state, Board) :
            player_state['location'] = list(player_state['location'])
            playerXLocation = player_state['location'][0]
            playerYLocation = player_state['location'][1]
            player = 10 * playerXLocation + playerYLocation
            ## PLAYER TILE
            playerTile = Board[player]
            
            ## LEFT TILES
            if playerXLocation > 0 :
                oneLeftTile = Board[player - 10]
                if playerXLocation > 1 :
                    twoLeftTile = Board[player - 20]
                else :
                    twoLeftTile = 99
            else :
                oneLeftTile = 99
                twoLeftTile = 99
            
            ## RIGHT TILES
            if playerXLocation < 11 :
                oneRightTile = Board[player + 10]
                if playerXLocation < 10 :
                    twoRightTile = Board[player + 20]
                else :
                    twoRightTile = 99
            else :
                oneRightTile = 99
                twoRightTile = 99

            ## UP TILES
            if playerYLocation < 9 :
                oneUpTile = Board[player + 1]
                if playerYLocation < 8 :
                    twoUpTile = Board[player + 2]
                else :
                    twoUpTile = 99
            else :
                oneUpTile = 99
                twoUpTile = 99

            ## DOWN TILES
            if playerYLocation > 0 :
                oneDownTile = Board[player - 1]
                if playerYLocation > 1 :
                    twoDownTile = Board[player - 2]
                else :
                    twoDownTile = 99
            else :
                oneDownTile = 99
                twoDownTile = 99

            return playerTile, \
                    oneLeftTile, twoLeftTile, \
                    oneRightTile, twoRightTile, \
                    oneUpTile, twoUpTile, \
                    oneDownTile, twoDownTile
        
        def validAction(game_state, player_state, action, Board) :
            player_state = player_state.__dict__
            player_state['location'] = list(player_state['location'])
            playerLocation = player_state["location"]
            player_location_x = int(Board[len(Board)-5]*12)
            player_location_y = int(Board[len(Board)-4]*10)

            
            if player_location_x > 0 :
                leftTile = Board[10*player_location_x + player_location_y - 10]
            else :
                leftTile = 99
            if player_location_x < 11 :
                rightTile = Board[10 * player_location_x + player_location_y + 10]
            else :
                rightTile = 99
            if player_location_y < 9 :
                upTile = Board[10 * player_location_x + player_location_y + 1]
            else :
                upTile = 99
            if player_location_y > 0 :
                downTile = Board[10 * player_location_x + player_location_y - 1]
            else :
                downTile = 99

            match action :
                case 0 :
                    return True
                case 1 :
                    if upTile >= 4/10 and upTile <= 7/10 or upTile == 99   :
                        return False
                    else :
                        return True  
                case 2 :
                    if downTile >= 4/10 and downTile <= 7/10 or downTile == 99:
                        return False 
                    else :
                        return True 
                case 3 :
                    if leftTile >= 4/10 and leftTile <= 7/10 or leftTile == 99:
                        return False 
                    else :
                        return True
                case 4 :
                    if rightTile >= 4/10 and rightTile <= 7/10 or rightTile == 99:
                        return False 
                    else :
                        return True
                case 5 :
                    if player_state['ammo'] > 0 :
                        return True
                    else :
                        return False
        
        def future_state(game_state, player_state, action, Board) :
            rewardBonus = 0
            game_state = game_state.__dict__
            player_state = player_state.__dict__
            player_state['location'] = list(player_state['location'])
            ammo = player_state["ammo"]

            match action :
                case 0 :
                    pass
                case 1 :
                    player_state['location'][1] += 1
                case 2 :
                    player_state['location'][1] -= 1
                case 3 :
                    player_state['location'][0] -= 1
                case 4 :
                    player_state['location'][0] += 1
                case 5 :
                    if ammo > 0 :
                        game_state['_game_map'][player_state['location'][0]][player_state['location'][1]] = 0.3
                    elif ammo == 0 :
                        pass
                case other :
                    pass
            
            playerTile, \
            oneLeftTile, twoLeftTile, \
            oneRightTile, twoRightTile, \
            oneUpTile, twoUpTile, \
            oneDownTile, twoDownTile = getEightNeighbors(player_state, Board)


            ## PLAYER TILE REWARD
            match playerTile :
                case 0.1 :
                    player_state['ammo'] += 1 
                    rewardBonus += 1
                case 0.2 :
                    rewardBonus += 0.02
                case 0.3 :
                    rewardBonus -= 0.1
                case other :
                    pass   

            ## LEFT TILES 
            match oneLeftTile :
                case 0.3 :
                    rewardBonus -= 0.02
                case other :
                    pass 
            
            match twoLeftTile :
                case 0.3 :
                    rewardBonus -= 0.02
                case other :
                    pass 

            ## RIGHT TILES 
            match oneRightTile :
                case 0.3 :
                    rewardBonus -= 0.02
                case other :
                    pass 
            
            match twoRightTile :
                case 0.3 :
                    rewardBonus -= 0.02
                case other :
                    pass 
            
            
            ## UP TILES 
            match oneUpTile :
                case 0.3 :
                    rewardBonus -= 0.02
                case other :
                    pass 
            
            match twoUpTile :
                case 0.3 :
                    rewardBonus -= 0.02
                case other :
                    pass 
            
            
            ## DOWN TILES 
            match oneDownTile :
                case 0.3 :
                    rewardBonus -= 0.02
                case other :
                    pass 
            
            match twoDownTile :
                case 0.3 :
                    rewardBonus -= 0.02
                case other :
                    pass 
            
            if action == 5 and ammo > 0:
                if oneLeftTile == 0.7 or twoLeftTile == 0.7 or oneRightTile == 0.7 or twoRightTile == 0.7 or oneUpTile == 0.7 or twoUpTile == 0.7 or oneDownTile == 0.7 or twoDownTile ==0.7 :  
                    rewardBonus += 1
                if oneLeftTile == 0.4 or twoLeftTile == 0.4 or oneRightTile == 0.4 or twoRightTile == 0.4 or oneUpTile == 0.4 or twoUpTile == 0.4 or oneDownTile == 0.4 or twoDownTile ==0.4 :  
                    rewardBonus += 0.01
                if oneLeftTile == 0.5 or twoLeftTile == 0.5 or oneRightTile == 0.5 or twoRightTile == 0.5 or oneUpTile == 0.5 or twoUpTile == 0.5 or oneDownTile == 0.5 or twoDownTile ==0.5 :  
                    rewardBonus += 0.005
            
            
            player_state['location'] = tuple(player_state['location'])
            return game_state,player_state,rewardBonus

        
        GameState, GameWinner, PlayerReward, is_over, hp, distanceToEnemy = self.NN.GetState(game_state,player_state, self.game.stats.players)
        state = torch.tensor(GameState, dtype=torch.float32, device=self.device).unsqueeze(0)
        actionValid = False
        randomActive = False

        while not actionValid :
            action = self.select_action(GameState, randomActive)
            if validAction(game_state, player_state, action, GameState) :
                actionValid = True
            else  :
                 randomActive = True

        if action == 0 :
            self.TotalReward -= 0.002

        RewardDifference = 0
        RewardDifference = PlayerReward - self.pastReward
        self.TotalReward += RewardDifference

        ## Reward bonifiers
        if self.pastHP > hp :
            self.TotalReward -= 5
        if RewardDifference > 20 :
            self.TotalReward += 15


        obsGameState, obsPlayerState, obsReward = future_state(game_state,player_state,action, GameState)
        obsState, GameWinner, PlayerReward, is_over, hp, newEnemyDistance  = self.NN.GetState(obsGameState,obsPlayerState,self.game.stats.players)
        self.TotalReward += obsReward
        ammo = GameState[len(GameState)-3]

        if newEnemyDistance > 0 :
            if ammo > 0 :
                self.TotalReward += (0.01/newEnemyDistance)
        else :
            if ammo > 0 :
                self.TotalReward += 0.0105
            else :
                self.TotalReward -= 0.01

        reward = torch.tensor([self.TotalReward], device=self.device)
        done = is_over
        next_state = torch.tensor(obsState, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Store the transition in memory
        self.memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        
        if self.testing == True :
            action = testInput(game_state, player_state, GameState)
        else :
            # Perform one step of the optimization (on the policy network)
            self.optimize_model()

        # Soft update of the target network's weights
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

        match action :
            case 0 :
                action = ''
            case 1 :
                action = 'u'
            case 2 :
                action = 'd'
            case 3 :
                action = 'l'
            case 4 :
                action = 'r'
            case 5 :
                action = 'p'
        self.pastHP = hp
        self.pastTotalReward = self.TotalReward
        self.pastReward = PlayerReward
        print(self.TotalReward)
        torch.save(self.target_net.state_dict(),"target_net.pth")
        torch.save(self.policy_net.state_dict(),"policy_net.pth")

        return action