from coderone.dungeon import counter
import time
import argparse
import os
#from . import counter
#counter.init()

SCREEN_TITLE = "Coder One: Dungeons & Data Structures"
parser = argparse.ArgumentParser(description=SCREEN_TITLE)
parser.add_argument('--headless', action='store_true',
					default=False,
					help='run without graphics')
parser.add_argument('--interactive', action='store_true',
                default=False,
                help='all a user to contol a player')
parser.add_argument('--no_text', action='store_true',
                default=False,
                help='Graphics bug workaround - disables all text')
parser.add_argument('--players', type=str,
                help="Comma-separated list of player names")
parser.add_argument('--hack', action='store_true',
                default=False,
                help=argparse.SUPPRESS)
parser.add_argument('--start_paused', action='store_true',
                default=False,
                help='Start a game in pause mode, only if interactive')
parser.add_argument('--single_step', action='store_true',
                default=False,
                help='Game will run one step at a time awaiting for player input')
parser.add_argument('--endless', action='store_true',
                default=False,
                help='Game will restart after the match is over. indefinitely')

parser.add_argument('--submit', action='store_true',
                default=False,
                help="Don't run the game, but submit the agent as team entry into the trournament")

parser.add_argument('--record', type=str,
                help='file name to record game')
parser.add_argument('--watch', action='store_true',
                default=False,
                help='automatically reload agents on file changes')
parser.add_argument('--config', type=str,
                default=None,
                help='path to the custom config file')

args = parser.parse_args()
args.headless = True
#args.record = True
#args.endless = True

def returnCounter() :
    return counter.globalCounter

def main() :
    
    enemyAgent = "venv\Lib\site-packages\coderone\dungeon\my_agent.py"
    playerAgent = "venv\Lib\site-packages\coderone\dungeon\IntelAgent.py"
    # run_match([enemyAgent,playerAgent],["my_agent.py",
    #                                         "IntelAgent.py"], 
    #                                         args = args)
    os.system("cd ./venv/Lib/site-packages/coderone/dungeon")
    while counter.globalCounter < 1500 :
        #run_match([enemyAgent,playerAgent],["my_agent.py",
        #                                     "IntelAgent.py"], 
        #                                     args = args)
        os.system("python -m coderone.dungeon.main my_agent.py IntelAgent.py --headless")
        print(counter.globalCounter)
        counter.globalCounter += 1

if __name__ == '__main__' :
    main()