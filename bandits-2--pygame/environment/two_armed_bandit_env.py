"""
ISPTSC 2023
Lecture 1

Author: Alejandro Mujica - alejandro.j.mujic4@gmail.com
Author: Jesús Pérez - perezj89@gmail.com

This file contains the the Two-Armed Bandit Environment.
"""
import time

import gym

import numpy as np

import pygame

from . import settings

import random

pygame.init()
pygame.display.init()


class Arm:
    def __init__(self, p=0, earn=0):
        self.probability = p
        self.earn = earn

    def pull(self):
        return self.earn if np.random.random() < self.probability else 0

    def __str__(self):
        return f"Arm: p = {self.probability}, earn: {self.earn}"

    def __repr__(self):
        return f"Arm: p = {self.probability}, earn: {self.earn}"


class TwoArmedBanditEnv(gym.Env):

    def __init__(self):
        self.arms = (
            Arm(0.5, 1),
            Arm(0.1, 100)
        )
        self.action = None
        self.reward = None
        self.total_reward = 0
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(len(self.arms))

        self.window = pygame.display.set_mode(
            (settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT)
        )
        pygame.display.set_caption("Two-Armed Bandit Environment")

    def _get_observations(self):
        return 0

    def _get_info(self):
        return {'state': 0}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._get_observations(), self._get_info()

    def step(self, action):
        self.action = action
        self.reward = self.arms[action].pull()
        self.total_reward += self.reward
        self.render()
        return self._get_observations(), self.reward, False, False, self._get_info()

    def render(self):
        self.window.fill((255, 255, 255))

        # Render the first machine
        self.window.blit(
            settings.TEXTURES['machine'], (50, settings.TITLE_HEIGHT + 100))

        # Render the second machine
        self.window.blit(
            settings.TEXTURES['machine'], (100 + settings.MACHINE_WIDTH,  settings.TITLE_HEIGHT + 100))

        # Render the arrow under the selected machine
        x = 50 + settings.MACHINE_WIDTH / 2

        if self.action == 1:
            x += settings.MACHINE_WIDTH + 50

        y = settings.WINDOW_HEIGHT - 50 - settings.ARROW_HEIGHT / 2
        self.window.blit(
            settings.TEXTURES['arrow'], (x - settings.ARROW_WIDTH / 2 - 80, y))

        # Render the reward
        text_obj0 = settings.FONTS['font'].render(
            f"{self.reward}", True, (0, 0, 0))
        text_rect1 = text_obj0.get_rect()
        text_rect1.center = (x + 2, settings.TITLE_HEIGHT + 52)
        self.window.blit(text_obj0, text_rect1)

        text_obj1 = settings.FONTS['font'].render(
            f"{self.reward}", True, (255, 250, 26))
        text_rect2 = text_obj1.get_rect()
        text_rect2.center = (x, settings.TITLE_HEIGHT + 50)
        self.window.blit(text_obj1, text_rect2)

        # Render Total Reward
        text_obj0 = settings.FONTS['font'].render(
            f"{self.total_reward}", True, (0, 0, 0))
        text_rect1 = text_obj0.get_rect()
        text_rect1.center = (settings.WINDOW_WIDTH / 2 + 4,
                             50 + 4)
        self.window.blit(text_obj0, text_rect1)

        text_obj0 = settings.FONTS['font'].render(
            f"{self.total_reward}", True, (255, 215, 0))
        text_rect1 = text_obj0.get_rect()
        text_rect1.center = (settings.WINDOW_WIDTH / 2,
                             50)
        self.window.blit(text_obj0, text_rect1)

        # Render something to identify the current machine
        self.window.blit(
            pygame.transform.rotate(
                settings.TEXTURES['coin'], random.randint(0, 180)),
            (x - 100 - random.randint(-20, 20), settings.TITLE_HEIGHT + settings.MACHINE_HEIGHT + 10))

        pygame.event.pump()
        pygame.display.update()

        time.sleep(0.5)
        # time.sleep(2)

    def close(self):
        pygame.display.quit()
        pygame.font.quit()
        pygame.quit()
