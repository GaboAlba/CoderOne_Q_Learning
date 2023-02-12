import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

## Setting up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

"""

GAME STATE SAMPLE

{'is_over': True, 
'tick_number': 1128, 
'_size': (12, 10), 
'_game_map': 
	{2: {6: 0, 4: 'ib', 3: 'ib', 7: 'ib', 0: 'sb', 1: 'ob'}, 
	10: {3: 1, 0: 'sb'}, 
	0: {5: 'ib', 0: 'ib', 6: 'ib', 2: 'sb', 1: 't'}, 
	11: {8: 'ib', 4: 'ib', 7: 'sb'}, 
	1: {5: 'ib', 3: 'ib', 1: 'sb', 0: 'sb'},
	3: {8: 'ib', 1: 'sb', 2: 'a'}, 
	4: {5: 'ib', 2: 'ib', 1: 'sb', 3: 'ob'}, 
	5: {9: 'ib', 6: 'ob'}, 
	9: {2: 'ib', 3: 'ib', 7: 'ob', 9: 'a', 5: 't'}, 
	7: {6: 'ib'}, 
    6: {3: 'ib'}, 
	8: {9: 'sb', 7: 'sb', 5: 'ob', 6: 't'}}, 
'_treasure': [(0, 1), (9, 5), (8, 6)], '
_ammo': [(3, 2), (9, 9)], 
'_bombs': [], 
'_blocks': [('ib', (2, 4)), 
		('ib', (0, 5)), 
		('ib', (11, 8)), 
		('ib', (1, 5)), 
		('ib', (3, 8)), 
		('ib', (4, 5)), 
		('ib', (5, 9)), 
		('ib', (2, 3)), 
		('ib', (4, 2)), 
		('ib', (0, 0)), 
		('ib', (0, 6)), 
		('ib', (2, 7)), 
		('ib', (9, 2)), 
		('ib', (11, 4)), 
		('ib', (7, 6)), 
		('ib', (1, 3)), 
		('ib', (9, 3)), 
		('ib', (6, 3)), 
		('sb', (4, 1)), 
		('sb', (8, 9)), 
		('sb', (11, 7)), 
		('sb', (3, 1)), 
		('sb', (0, 2)), 
		('sb', (1, 1)), 
		('sb', (2, 0)), 
		('sb', (8, 7)), 
		('sb', (1, 0)), 
		('sb', (10, 0)), 
		('ob', (2, 1)), 
		('ob', (4, 3)), 
		('ob', (8, 5)), 
		('ob', (9, 7)), 
		('ob', (5, 6))], 
'_players': [(0, (2, 6)), (1, (10, 3))]}

PLAYER STATE
{'id': 1, 'ammo': 1, 'hp': 1, 'location': (10, 3), 'reward': 18, 'power': 2}

# We have total 12x10 = 120 cells
# actions = ['','u','d','l','r','p']

ACRONYMS
Ammo = "a" = 1
Treasure = 't' = 2
Bomb = "b" = 3
SoftBlock = 'sb' = 4
OreBlock = 'ob' = 5
IndestructibleBlock = 'ib' = 6
Player0 = 'p0' = 0
Player1 = 'p1' = 1
Clear = "c"  = 7

"""      

# GPU Capability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module) :

    def __init__(self,n_inputs, n_outputs) :
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(self.n_inputs, 125)
        self.layer2 = nn.Linear(125,125)
        self.layer3 = nn.Linear(125,self.n_outputs)
        

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class Trainer :

    def __init__(self,n_inputs, n_outputs) :
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 0.001
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # Defining Neural Networks
        self.policy_net = DQN(self.n_inputs,self.n_outputs).to(device)
        self.target_net = DQN(self.n_inputs,self.n_outputs).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = self.LR, amsgrad = True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0
        self.actions = ['','u','d','l','r','p']
        self.episode_durations = []
    
    def select_action(self,state)  :
        global steps_done
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample >= eps_threshold :
            with torch.no_grad() :
                return self.policy_net(state).max(1)[1].view(1,1)
        else :
            return torch.tensor([[self.actions.sample()]], device=device, dtype=torch.long)
    def plot_durations(show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    
    def optimize_model() :
        pass


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
        pass

    def GetState(self,game_state,player_state) :
        ## Player State and Game State must be read as input to the NN

        ## GAME STATE READING
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
        self.game_state = game_state.__dict__
        #self.game_state = game_state
        #self.player_state = player_state
        Xkeys = self.game_state['_game_map'].keys()
        for x in Xkeys :
            for y in self.game_state['_game_map'][x] :
                state = self.game_state['_game_map'][x][y]
                # 0-10 codification to make it easy to normalize as NN inputs 
                match state :
                    case "a" :
                        state = 1
                    case "t" :
                        state = 2
                    case "b" :
                        state = 3
                    case "sb" :
                        state = 4
                    case "ob" :
                        state = 5
                    case "ib" :
                        state = 6
                    case 0 :
                        state = 9 
                    case 1 :
                        state = 10
                    case other :
                        state = -1
                self.Board[x][y] = state
        return  self.Board

    def Policy(self) :
        return 0

    def GetAction(self,state) :
        #self.output = self.Policy(state)
        return state   ## Change this for policy in the future
    
    def CreateNeuralNetwork(self) :
        #InputSize = len(self.input())
        pass

    class Trainer :
        def __init__(self) -> None:
            pass

def main() :
    game_state = {'is_over': True, 
    'tick_number': 1128, 
    '_size': (12, 10), 
    '_game_map': 
        {2: {6: 0, 4: 'ib', 3: 'ib', 7: 'ib', 0: 'sb', 1: 'ob'}, 
        10: {3: 1, 0: 'sb'}, 
        0: {5: 'ib', 0: 'ib', 6: 'ib', 2: 'sb', 1: 't'}, 
        11: {8: 'ib', 4: 'ib', 7: 'sb'}, 
        1: {5: 'ib', 3: 'ib', 1: 'sb', 0: 'sb'},
        3: {8: 'ib', 1: 'sb', 2: 'a'}, 
        4: {5: 'ib', 2: 'ib', 1: 'sb', 3: 'ob'}, 
        5: {9: 'ib', 6: 'ob'}, 
        9: {2: 'ib', 3: 'ib', 7: 'ob', 9: 'a', 5: 't'}, 
        7: {6: 'ib'}, 
        6: {3: 'ib'}, 
        8: {9: 'sb', 7: 'sb', 5: 'ob', 6: 't'}}, 
    '_treasure': [(0, 1), (9, 5), (8, 6)], 
    '_ammo': [(3, 2), (9, 9)], 
    '_bombs': [], 
    '_blocks': [('ib', (2, 4)), 
            ('ib', (0, 5)), 
            ('ib', (11, 8)), 
            ('ib', (1, 5)), 
            ('ib', (3, 8)), 
            ('ib', (4, 5)), 
            ('ib', (5, 9)), 
            ('ib', (2, 3)), 
            ('ib', (4, 2)), 
            ('ib', (0, 0)), 
            ('ib', (0, 6)), 
            ('ib', (2, 7)), 
            ('ib', (9, 2)), 
            ('ib', (11, 4)), 
            ('ib', (7, 6)), 
            ('ib', (1, 3)), 
            ('ib', (9, 3)), 
            ('ib', (6, 3)), 
            ('sb', (4, 1)), 
            ('sb', (8, 9)), 
            ('sb', (11, 7)), 
            ('sb', (3, 1)), 
            ('sb', (0, 2)), 
            ('sb', (1, 1)), 
            ('sb', (2, 0)), 
            ('sb', (8, 7)), 
            ('sb', (1, 0)), 
            ('sb', (10, 0)), 
            ('ob', (2, 1)), 
            ('ob', (4, 3)), 
            ('ob', (8, 5)), 
            ('ob', (9, 7)), 
            ('ob', (5, 6))], 
    '_players': [(0, (2, 6)), (1, (10, 3))]}

    player_state = {'id': 1, 'ammo': 1, 'hp': 1, 'location': (10, 3), 'reward': 18, 'power': 2}

    NN = NeuralNetwork()
    Board = NeuralNetwork.Input(NN,game_state,player_state)
    print(Board)

if __name__ == "__main__" :
    main()
