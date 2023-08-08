consts = {}

# PROJECT STRUCTURE
CHECKPOINT_DIR = './train'
LOG_DIR = './logs/'
ERROR_DIR = './errors/'
MODELS_DIR = './train/'

# SERVER CONFIG
TEAM_ID = 2888
SERVER_ADDRESS = '127.0.0.1'
SERVER_PORT = 2004

# SERVER MESSAGE RESPONSES
STATE_UNKNOWN = 0
STATE_MAIN_MENU = 1
STATE_EPISODE_MENU = 2
STATE_LEVEL_SELECTION = 3
STATE_LOADING = 4
STATE_PLAYING = 5
STATE_WON = 6
STATE_LOST = 7

# RL ENVIRONMENT
SCREEN_WIDTH = 840
SCREEN_HEIGHT = 480

GAME_WIDTH = 840
GAME_HEIGHT = 480
GAME_BATCH = 1
GAME_ACTION_SPACE = 30

ACTION_MAP = {
    0: [-10, 10],
    1: [-10, 20],
    2: [-10, 30],
    3: [-10, 40],
    4: [-10, 50],
    5: [-10, 60],
    6: [-10, 70],
    7: [-10, 80],
    8: [-10, 90],
    9: [-10, 100],
    10: [-20, 10],
    11: [-20, 20],
    12: [-20, 30],
    13: [-20, 40],
    14: [-20, 50],
    15: [-20, 60],
    16: [-20, 70],
    17: [-20, 80],
    18: [-20, 90],
    19: [-20, 100],
    20: [-30, 10],
    21: [-30, 20],
    22: [-30, 30],
    23: [-30, 40],
    24: [-30, 50],
    25: [-30, 60],
    26: [-30, 70],
    27: [-30, 80],
    28: [-30, 90],
    29: [-30, 100],

}

SCORE_NORMALIZATION = 10000

# OCR STUFF
REWARD_TOP = 20
REWARD_BOTTOM = 51
REWARD_LEFT = 670
REWARD_RIGHT = 790
