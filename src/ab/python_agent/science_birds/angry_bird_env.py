import json
import logging
import socket
from datetime import datetime
from os.path import dirname, abspath, join
import sys
import re

from PIL import Image

from .agent_client import AgentClient, GameState

# Find code directory relative to our directory
# THIS_DIR = dirname(__file__)
# CODE_DIR = abspath(join(THIS_DIR, '..'))
# sys.path.append(CODE_DIR)


# Find code directory relative to our directory
THIS_DIR = dirname(__file__)
CODE_DIR = abspath(join(THIS_DIR, '..'))
sys.path.append(CODE_DIR)

from .consts import *
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

    def __init__(self, agent_configs, agent_ind):
        super().__init__()
        # test for a single shot
        self.shot_done = False

        self.agent_ind = agent_ind

        #################initalising the logger#################
        # file_handler saves all logs to a log file with agent_ind

        self.logger = logging.getLogger(self.agent_ind)

        formatter = logging.Formatter("%(asctime)s-Agent %(name)s-%(levelname)s : %(message)s")

        if not os.path.exists("log"):
            os.mkdir("log")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.WARNING)
        stream_handler.setFormatter(formatter)
        if agent_configs.save_logs:
            file_handler = logging.FileHandler(os.path.join("log", "%s.log" % (self.agent_ind)))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        agent_ip = agent_configs.agent_host
        agent_port = agent_configs.agent_port
        observer_ip = agent_configs.observer_host
        observer_port = agent_configs.observer_port

        self.current_level = 0
        self.training_level_backup = 0;
        self.failed_counter = 0
        self.solved = []
        self.id = 28888
        self.first_shot = True
        self.prev_target = None
        self.novelty_existence = -1;
        self.sim_speed = 50
        self.prev_gt = None
        self.repeated_gt_counter = 0
        self.gt_patient = 10
        self.if_check_gt = False

        ########################################################

        # Wrapper of the communicating messages

        self.ar = AgentClient(agent_ip, agent_port, logger=self.logger)

        # the observer agent can only execute 6 command: configure, screenshot
        # and the four groundtruth related ones
        # self.observer_ar = AgentClient(observer_ip, observer_port)

        try:
            self.ar.connect_to_server()
        except socket.error as e:
            self.logger.error("Error in client-server communication: " + str(e))

        # ################################################
        # low/high - inputs value, shape(batch,height,width), datatype
        self.observation_space = Box(low=0, high=255,
                                     shape=(GAME_BATCH, GAME_HEIGHT, GAME_WIDTH),
                                     dtype=np.uint8)
        # TODO: need to check out continues action space.
        self.action_space = Discrete(GAME_ACTION_SPACE)
        # Top right corner, get score during round.
        self.reward_location = {'top': 50, 'left': 630, 'width': 100, 'height': 50}

        # self.solved = [0] * config[2]
        self.current_level = 1

    def step(self, action):
        game_state = self.ar.get_game_state()
        print(f'STATE2: {game_state}')
        release_point = None
        # Action key - [dx,dy]
        print(f'using action: {action}')
        # SEND ACTION TO SERVER (196,326)
        # makeshot = self.ar.c_shoot(191, 344, ACTION_MAP[action][0], ACTION_MAP[action][1], 0,
        #                            0)
        batch_gt = self.ar.shoot_and_record_ground_truth(40, 40, 0, 0, 1, 0)
        time.sleep(2 / self.sim_speed)

        # Get the next observation
        new_observation = self.get_observation()
        # Check if game is done
        done, is_win = self.get_done()

        if not done:
            img_dir = self.ar.do_screenshot()
            image = cv2.imread(img_dir)
            cropped = image[REWARD_TOP:REWARD_BOTTOM, REWARD_LEFT:REWARD_RIGHT]

            gray_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            (thresh, blackAndWhiteImage) = cv2.threshold(gray_image, 170, 255, cv2.THRESH_BINARY)
            res = pytesseract.image_to_string(blackAndWhiteImage, config="--psm 10")
            # reward = int(res) if self._check_OCR(res,blackAndWhiteImage) else 0
            if re.match("^[0-9 ]+$", res):
                reward = int(res)
            else:
                reward = 0
        else:
            score = self.ar.get_current_score()
            reward = score[self.current_level - 1]

        # Info dict - not relevant
        info = {'is_win': is_win["is_win"]}
        if info["is_win"]:
            self.current_level = self.current_level + 1

        return new_observation, reward / SCORE_NORMALIZATION, done, False, info

    def render(self):
        cv2.imshow('Game', self.ar.do_screenshot())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.ar.get_game_state()
        print(f'STATE: {state}')
        if state == GameState.WON or state == GameState.LEVEL_SELECTION:
            load_lvl = self.ar.load_level(self.current_level)  # TODO: add an if load_lvl failed?
        elif state == GameState.LOST:
            self.ar.restart_level()
        elif state == GameState.NEWTRAININGSET:
            self.ar.load_level(1)
            state = self.ar.get_game_state()
            print(f'STATE: {state}')
        new_observ = self.get_observation()
        return new_observ, {}

    def close(self):
        cv2.destroyAllWindows()

    def get_observation(self, return_raw=False):
        # get screen caputre of game
        image = self.ar.do_screenshot()
        # print(img_dir)
        # image = Image.open(img_dir)
        # TODO: here needs to call a function that sends image to YOLO returns a state
        raw = np.array(image)
        crop = raw[100:400, 40:]
        plt.imshow(image)
        plt.show()
        # TODO: dont think gray is a good idea...
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (GAME_HEIGHT, GAME_WIDTH))
        channel = np.reshape(resized, (GAME_BATCH, GAME_HEIGHT, GAME_WIDTH))
        return channel

    def get_done(self):
        state = self.ar.get_game_state()
        if state == GameState.WON:
            self.solved[self.current_level - 1] = 1
            return True, {'is_win': True}
        elif state == GameState.LOST:
            return True, {'is_win': False}
        elif state == GameState.PLAYING:
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
        scores = self.ar.get_current_score()
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

    def check_current_level_score(self):
        current_score = self.ar.get_current_score()
        self.logger.info("current score is %d " % current_score)
        return current_score

    def run(self):
        self.ar.configure(self.id)
        self.ar.set_game_simulation_speed(self.sim_speed)
        model = DQN('CnnPolicy', self, tensorboard_log=LOG_DIR, verbose=1, buffer_size=12000, learning_starts=1000)
        # TRAIN
        state = self.ar.get_game_state()
        print(f'STATE1: {state}')
        if state == GameState.NEWTRIAL:
            self.repeated_gt_counter = 0
            # Make a fresh agent to continue with a new trial (evaluation)
            self.logger.critical("new trial state received")
            (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
             allowNoveltyInfo) = self.ar.ready_for_new_set()
            self.current_level = 0
            self.training_level_backup = 0
        elif state == GameState.NEWTRAININGSET:
            self.repeated_gt_counter = 0
            # DO something to start a fresh agent for a new training set
            self.logger.critical("new training set state received")
            (time_limit, interaction_limit, n_levels, attempts_per_level, mode, seq_or_set,
             allowNoveltyInfo) = self.ar.ready_for_new_set()
            self.current_level = 0
            self.training_level_backup = 0
            change_from_training = True
        self.logger.info("sovling level %s" % self.current_level)
        game_state = self.ar.get_game_state()
        if game_state != GameState.PLAYING:
            state =  game_state

        else:
            self.repeated_gt_counter += 1
            if self.repeated_gt_counter > self.gt_patient:
                self.logger.warning("counter %s reached, game state set to lost"%(self.gt_patient))
                self.repeated_gt_counter = 0
                return GameState.LOST
        callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
        model.learn(total_timesteps=100000, callback=callback)


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
        obs = env.reset()
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

# env = AngryBirdGame()
# env_checker.check_env(env)
#
# # create DQN model policy, the enviorment, where to save logs, verbose logs, how big is the buffer depends on RAM,
# # start training after 1000 steps
# model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=12000, learning_starts=1000)
# print(model)
#
# # TRAIN
# callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
# model.learn(total_timesteps=100000, callback=callback)

# TESTING
# test_model(env)
