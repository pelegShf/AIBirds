from datetime import datetime
from os.path import dirname, abspath, join
import sys
import re

# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..'))
sys.path.append(CODE_DIR)

from consts import *
from other.clientActionRobot import *
from vision.game_state_extractor import get_slingshot, crop_img

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
                                     shape=(GAME_HEIGHT, GAME_WIDTH, CHANNELS),
                                     dtype=np.uint8)
        self.action_space = Discrete(GAME_ACTION_SPACE)
        # config message, and gets the [Round Info, Time_limit , Number of Levels]
        config = self.ar.configure()  # configures message to the server
        self.state = self.ar.get_state()[0]
        self.solved = [0] * config[2]
        self.current_level = 1
        # basic values for slingshot
        self.slingshotX = 196
        self.slingshotY = 326

    def step(self, action):
        # Action key - [dx,dy]
        print(f'using action: {action}')
        # SEND ACTION TO SERVER (196,326)
        # makeshot = self.ar.c_shoot(191, 344, ACTION_MAP[action][0], ACTION_MAP[action][1], 0,
        #                            0)
        makeshot = self.ar.c_shoot(self.slingshotX, self.slingshotY, ACTION_MAP[action][0], ACTION_MAP[action][1], 0,
                                   0)
        # Get the next observation
        new_observation = self.get_observation()
        # Check if game is done
        done, is_win = self.get_done()

        if not done:
            img_dir = self.ar.do_screen_shot()
            image = cv2.imread(img_dir)
            cropped = image[REWARD_TOP:REWARD_BOTTOM, REWARD_LEFT:REWARD_RIGHT]

            grayImage = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 170, 255, cv2.THRESH_BINARY)
            res = pytesseract.image_to_string(blackAndWhiteImage, config="--psm 10")
            # reward = int(res) if self._check_OCR(res,blackAndWhiteImage) else 0
            if re.match("^[0-9 ]+$", res):
                reward = int(res)
            else:
                reward = 0
        else:
            score = self.ar.get_my_score()
            reward = score[self.current_level - 1]

        # Info dict - not relevant
        info = {'is_win': is_win["is_win"]}
        if info["is_win"]:
            self.current_level = self.current_level + 1

        return new_observation, reward / SCORE_NORMALIZATION, done, False, info

    def render(self):
        cv2.imshow('Game', self.ar.do_screen_shot())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.ar.get_state()[0]
        if state == STATE_WON or state == STATE_LEVEL_SELECTION:
            load_lvl = env.ar.load_level(self.current_level)  # TODO: add an if load_lvl failed?
        elif state == STATE_LOST:
            self.ar.restart_level()
        new_observ = self.get_observation()
        return new_observ, {}

    def close(self):
        cv2.destroyAllWindows()

    def get_observation(self, state=STATE_PLAYING):
        # get screen caputre of game
        img_dir = self.ar.do_screen_shot()
        cropped_img, cropped_img_dir = crop_img(img_dir)
        if state == STATE_PLAYING:
            x, y = get_slingshot(cropped_img_dir)
            self.slingshotX = x
            self.slingshotY = y
            print(x, y)

        img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (GAME_WIDTH, GAME_HEIGHT), interpolation=cv2.INTER_AREA)

        return img

    def get_done(self):
        self.state = self.ar.get_state()[0]
        if self.state == STATE_WON:
            self.solved[self.current_level - 1] = 1
            return True, {'is_win': True}
        elif self.state == STATE_LOST:
            return True, {'is_win': False}
        elif self.state == STATE_PLAYING:
            return False, {'is_win': False}
        else:
            print('not good err')
            return True, {'is_win': False}

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
        done = False
        total_reward = 0
        while not done:
            obs, reward, done, trunct, info = env.step(env.action_space.sample())
            total_reward = total_reward + (reward - total_reward)
        print(f'Total reward for episode {episode} is {total_reward}')
        print('______________________________')


def test_model(env):
    model_path = f'{MODELS_DIR}/best_model_5000.zip'
    model = DQN.load(model_path, env)
    for episode in range(5):
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, trunct, info = env.step(int(action))
            time.sleep(0.01)
            total_reward = total_reward + (reward - total_reward)
        print(f'Total reward for episode {episode} is {total_reward}')
        print('______________________________')


print('started to run')

env = AngryBirdGame()
env_checker.check_env(env)

# create DQN model policy, the environment, where to save logs, verbose logs, how big is the buffer depends on RAM,
# start training after 1000 steps
model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=12000, learning_starts=1000)
print(model)

# TRAIN
callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
model.learn(total_timesteps=100000, callback=callback)

# TESTING
# test_model(env)
