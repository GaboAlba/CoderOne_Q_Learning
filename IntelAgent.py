"""python3 -m venv venv
venv\Scripts\activate   """

from RL_Agent.DQN import NeuralNetwork
import random

class Agent :

    def __init__(self):
        pass

    def next_move(self, game_state, player_state):

        #print(game_state)
        #print(player_state)
        #print(game_state.__dict__)
        #print(player_state.__dict__)

        #POLICY WILL GO HERE
        #action = policy()
        actions = ['','u','d','l','r','p']
        NN = NeuralNetwork()
        state = NN.GetState(game_state,player_state)
        print(state)
        action = NN.GetAction(random.choice(actions))

        return action