consts = {}

# PROJECT SRUCTURE
CHECKPOINT_DIR = './train'
LOG_DIR = './logs/'
ERROR_DIR = './errors/'


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
GAME_ACTION_SPACE = 6

# OCR STUFF
REWARD_TOP = 20
REWARD_BOTTOM = 51 
REWARD_LEFT = 670
REWARD_RIGHT = 790

