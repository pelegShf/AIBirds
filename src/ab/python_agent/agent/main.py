from datetime import datetime
from os.path import dirname, abspath, join
import sys

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..'))
sys.path.append(CODE_DIR)

from consts import *
from other.clientActionRobot import *
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import pytesseract

# IMPORT CALLBACK STUFF
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker

# IMPORT NETWORK STUFF
from stable_baselines3 import DQN


class AngryBirdGame(Env):

    def __init__(self):
        super().__init__()
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.ar = ClientActionRobot(SERVER_ADDRESS, SERVER_PORT, TEAM_ID)
        # low/high - inputs value, shape(batch,height,width), datatype
        self.observation_space = Box(low=0, high=255,
                                     shape=(GAME_BATCH, GAME_HEIGHT, GAME_WIDTH),
                                     dtype=np.uint8)
        # TODO: need to check out continues action space.
        self.action_space = Discrete(GAME_ACTION_SPACE)
        # Top right corner, get score during round.
        self.reward_location = {'top': 50, 'left': 630, 'width': 100, 'height': 50}

        # config message, and gets the [Round Info, Time_limit , Number of Levels]
        config = self.ar.configure()  # configures message to the server
        self.solved = [0] * config[2]
        self.current_level = 1

    def step(self, action):
        # Action key - [dx,dy]
        action_map = {
            0: [-10, 10],
            1: [-10, 20],
            2: [-10, 30],
            3: [-20, 40],
            4: [-30, 50],
            5: [-40, 20],

        }
        # SEND ACTION TO SERVER
        makeshot = self.ar.c_shoot(196, 326, action_map[action][0], action_map[action][1], 0,
                                   0)  # shot in cartesian random values

        # Get the next observation
        new_observation = self.get_observation()
        # Check if game is done
        done, lvl = self.get_done()
        if not done:
            img_dir = self.ar.do_screen_shot()
            image = cv2.imread(img_dir)
            cropped = image[REWARD_TOP:REWARD_BOTTOM, REWARD_LEFT:REWARD_RIGHT]

            grayImage = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 170, 255, cv2.THRESH_BINARY)
            res = pytesseract.image_to_string(blackAndWhiteImage, config="--psm 10")
            # reward = int(res) if self._check_OCR(res,blackAndWhiteImage) else 0
            reward = int(res)
        else:
            score = self.ar.get_my_score()
            reward = score[self.current_level - 1]
        # Info dict - not relevant
        info = {'lvl': lvl}

        return new_observation, reward, done, False, info

    def render(self):
        cv2.imshow('Game', self.ar.do_screen_shot())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.ar.get_state()[0]
        if state == STATE_WON or state == STATE_LEVEL_SELECTION:
            load_lvl = env.ar.load_level(self.current_level)  # TODO: add an if load_lvl failed?
        else:
            self.ar.restart_level()
        new_observ = self.get_observation()
        return new_observ, {}

    def close(self):
        cv2.destroyAllWindows()

    def get_observation(self, return_raw=False):
        # get screen caputre of game
        img_dir = self.ar.do_screen_shot()
        image = Image.open(img_dir)
        raw = np.array(image)

        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (SCREEN_HEIGHT, SCREEN_WIDTH))
        channel = np.reshape(resized, (GAME_BATCH, SCREEN_HEIGHT, SCREEN_WIDTH))
        return channel

    def get_done(self):
        state = self.ar.get_state()[0]
        if state == STATE_WON:
            self.solved[self.current_level - 1] = 1
            self.current_level = self.current_level + 1
            return True, {'current_lvl': self.current_level - 1}
        elif state == STATE_LOST:
            return True, {'current_lvl': self.current_level}
        elif state == STATE_PLAYING:
            return False, {'current_lvl': self.current_level}
        else:
            print('not good err')
            return True, {}

    def _get_next_level(self):
        level = 0
        unsolved = False
        for i in range(len(self.solved)):
            if self.solved[i] == 0:
                unsolved = True
                level = level + 1
                if level <= self.current_level < self.solved[i]:
                    continue
                else:
                    return level
        if unsolved:
            return level
        level = (self.current_level + 1) % len(self.solved)

        if level == 0:
            level = len(self.solved)
        return level

    def _check_my_scores(self):
        scores = self.ar.get_my_score()
        level = 1
        for score in scores:
            if score > 0:
                self.solved[level - 1] = 1
            level = level + 1

    def _check_OCR(self, ocr_text, img):
        # TODO: fix the check if a number is returned from the OCR
        if ocr_text.isnumeric():
            return True
        else:
            try:
                img_dir = f'./{ERROR_DIR}{self.current_level}_{datetime.now()}.jpg'
                cv2.imwrite(img_dir, img)
                print(f'Err with the OCR, saved the image, in {img_dir}. Reward set to 0.')
                return False
            except cv2.error as e:
                print(e)
                return False


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True


def train(env):
    for episode in range(5):
        obs = env.reset()
        done = False
        total_reward = 0
        lvl = 1
        while not done:
            obs, reward, done, trunct, info = env.step(env.action_space.sample())
            print(f'playing level: {info["lvl"]} got reward: {reward} / {total_reward}')
            total_reward += reward
            lvl = info["lvl"]
        print(f'Total reward for episode {episode} is {total_reward} in level: {lvl}')
        print('______________________________')


print('started to run')

env = AngryBirdGame()
train(env)
# env_checker.check_env(env)
callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)

# TRAIN TODO: implement
# create DQN model policy, the enviorment, where to save logs, verbose logs, how big is the buffer depends on RAM,
# start training after 1000 steps
# model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=1200000, learning_starts=1000)
# print(model)
#
# # kickoff training
# model.learn(total_timesteps=100000, callback=callback)
