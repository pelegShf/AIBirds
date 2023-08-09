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

GAME_WIDTH = 128
GAME_HEIGHT = 73
GAME_BATCH = 1
CHANNELS = 3


ACTION_MAP = {
    0: [-10, 5],
    1: [-10, 10],
    2: [-10, 15],
    3: [-10, 20],
    4: [-10, 25],
    5: [-10, 30],
    6: [-10, 35],
    7: [-10, 40],
    8: [-15, 5],
    9: [-15, 10],
    10: [-15, 15],
    11: [-15, 20],
    12: [-15, 25],
    13: [-15, 30],
    14: [-15, 35],
    15: [-15, 40],
    16: [-20, 5],
    17: [-20, 10],
    18: [-20, 15],
    19: [-20, 20],
    20: [-20, 25],
    21: [-20, 30],
    22: [-20, 35],
    23: [-20, 40],
    24: [-25, 5],
    25: [-25, 10],
    26: [-25, 15],
    27: [-25, 20],
    28: [-25, 25],
    29: [-25, 30],
    30: [-25, 35],
    31: [-25, 40],
    32: [-30, 5],
    33: [-30, 10],
    34: [-30, 15],
    35: [-30, 20],
    36: [-30, 25],
    37: [-30, 30],
    38: [-30, 35],
    39: [-30, 40],
    40: [-35, 5],
    41: [-35, 10],
    42: [-35, 15],
    43: [-35, 20],
    44: [-35, 25],
    45: [-35, 30],
    46: [-35, 35],
    47: [-35, 40],
    48: [-40, 5],
    49: [-40, 10],
    50: [-40, 15],
    51: [-40, 20],
    52: [-40, 25],
    53: [-40, 30],
    54: [-40, 35],
    55: [-40, 40],

}

GAME_ACTION_SPACE = len(ACTION_MAP.keys())

SCORE_NORMALIZATION = 10000

# OCR STUFF
REWARD_TOP = 20
REWARD_BOTTOM = 51
REWARD_LEFT = 670
REWARD_RIGHT = 790

CROP_X = 70
CROP_Y = 110

SLINGSHOT_BOUNDRIES = [70, 100, 400, 380]  # X MIN, Y MIN, X MAX, Y MAX
DEFAULT_SLINGSHOT = [192, 326]