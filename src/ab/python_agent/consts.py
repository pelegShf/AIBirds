consts = {}

# PROJECT STRUCTURE
CHECKPOINT_DIR = './train'
LOG_DIR = './logs/'
ERROR_DIR = './errors/'
MODELS_DIR = './train/'
BEST_MODEL_DIR = f'{MODELS_DIR}NEW_STATE_1000.zip'
PYTESSERACT_DIR = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

GAME_WIDTH = 182
GAME_HEIGHT = 72
GAME_BATCH = 1
CHANNELS = 1


# ACTION_MAP = {
#     0: [-10, 5,0],
#     1: [-10, 10,0],
#     2: [-10, 15,0],
#     3: [-10, 20,0],
#     4: [-10, 25,0],
#     5: [-10, 30,0],
#     6: [-10, 35,0],
#     7: [-10, 40,0],
#     8: [-15, 5,0],
#     9: [-15, 10,0],
#     10: [-15, 15,0],
#     11: [-15, 20,0],
#     12: [-15, 25,0],
#     13: [-15, 30,0],
#     14: [-15, 35,0],
#     15: [-15, 40,0],
#     16: [-20, 5,0],
#     17: [-20, 10,0],
#     18: [-20, 15,0],
#     19: [-20, 20,0],
#     20: [-20, 25,0],
#     21: [-20, 30,0],
#     22: [-20, 35,0],
#     23: [-20, 40,0],
#     24: [-25, 5,0],
#     25: [-25, 10,0],
#     26: [-25, 15,0],
#     27: [-25, 20,0],
#     28: [-25, 25,0],
#     29: [-25, 30,0],
#     30: [-25, 35,0],
#     31: [-25, 40,0],
#     32: [-30, 5,0],
#     33: [-30, 10,0],
#     34: [-30, 15,0],
#     35: [-30, 20,0],
#     36: [-30, 25,0],
#     37: [-30, 30,0],
#     38: [-30, 35,0],
#     39: [-30, 40,0],
#     40: [-35, 5],
#     41: [-35, 10],
#     42: [-35, 15],
#     43: [-35, 20],
#     44: [-35, 25],
#     45: [-35, 30],
#     46: [-35, 35],
#     47: [-35, 40],
#     48: [-40, 5],
#     49: [-40, 10],
#     50: [-40, 15],
#     51: [-40, 20],
#     52: [-40, 25],
#     53: [-40, 30],
#     54: [-40, 35],
#     55: [-40, 40],
#
# }

def create_state_dict():
    data_dict = {}

    dx_range = range(-10, -50, -5)
    dy_range = range(5, 45, 5)
    t_range = range(500, 4000, 250)

    key_counter = 0
    for dx in dx_range:
        for dy in dy_range:
            for t in t_range:
                data_dict[key_counter] = [dx, dy, t]
                key_counter += 1
    print(data_dict)
    return data_dict


ACTION_MAP = create_state_dict()

THREE_STARS_SCORES = {
    1: 32000,
    2: 60000,
    3: 41000,
    4: 28000,
    5: 64000,
    6: 35000,
    7: 45000,
    8: 50000,
    9: 50000,
    10: 55000,
    11: 54000,
    12: 45000,
    13: 47000,
    14: 70000,
    15: 41000,
    16: 64000,
    17: 53000,
    18: 48000,
    19: 35000,
    20: 50000,
    21: 75000,
}

# Create a dictionary to map class labels to unique values
CLASS_TO_VALUE = {
    "unknown": 0,
    "hill": 1,
    "Wood": 2,
    "Ice": 3,
    "Stone": 4,
    "slingshot": 5,
    "pig": 6,
    "RedBird": 7,
    "YellowBird": 8,
    "WhiteBird": 9,
    "BlackBird": 10,
    "BlueBird": 11,
    "tnt": 12,
}
CLASS_COLOR = {
    "unknown": "#454545",
    "Wood": "#9C5E20",
    "Ice": "#EBE2D8",
    "Stone": "#18189E",
    "slingshot": "#9E9818",
    "pig": "#30C618",
    "hill": "#76801E",
    "tnt": "#9E1887",
    "RedBird": "#F62817",
    "YellowBird": "#E9AB17",
    "WhiteBird": "#F8F8FF",
    "BlackBird": "#0C090A",
    "BlueBird": "#728FCE"
    # Add more class labels and values as needed
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

OFFSETX = 0.5
OFFSETY = 0.65

SLINGSHOT_BOUNDRIES = [70, 100, 400, 380]  # X MIN, Y MIN, X MAX, Y MAX
DEFAULT_SLINGSHOT = [192, 326, 18, 66]
