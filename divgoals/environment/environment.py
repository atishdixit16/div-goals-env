from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product, combinations
import logging
from typing import Iterable
import operator
from copy import copy, deepcopy
from itertools import combinations

from gym import Env
import gym
from gym.utils import seeding
import numpy as np


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4

class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class DivGoalsEnv(Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]
    Observation = namedtuple(
        "Observation",
        ["field", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # reward is available only if is_self

    def __init__(
        self,
        players,
        field_size,
        sight,
        max_episode_steps,
        normalize_reward=True,
        penalty=0.0,
    ):
        self.logger = logging.getLogger(__name__)
        self.seed()
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32)

        self.penalty = penalty

        self._food_spawned = 0.0

        self.sight = sight
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(5)] * len(self.players)))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(self.players)))

        self.viewer = None

        self.n_agents = len(self.players)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """
        The Observation Space for each agent.
        - player description (x, y, level)*player_count
        """
        field_x = self.field.shape[1]
        field_y = self.field.shape[0]
        # field_size = field_x * field_y

        max_num_food = len(self.players)

        min_obs = [-1, -1, 0] * max_num_food + [-1, -1] * len(self.players)
        max_obs = [field_x-1, field_y-1, len(self.players)] * max_num_food + [field_x-1, field_y-1] * len(self.players)

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 1 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_food(self, max_num_food):
        food_count = 0
        attempts = 0
        self.goals = {}

        while food_count < max_num_food and attempts < 1000:
            attempts += 1
            row = self.np_random.randint(1, self.rows - 1)
            col = self.np_random.randint(1, self.cols - 1)

            # check if it has neighbors:
            if (
                self.neighborhood(row, col).sum() > 0
                or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                or not self._is_empty_location(row, col)
            ):
                continue

            self.field[row, col] = food_count + 1
            self.goals[food_count + 1] = np.array([row, col])

            food_count += 1

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self):
        for i,player in enumerate(self.players):

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.randint(0, self.rows)
                col = self.np_random.randint(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        i+1,
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                # and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                # and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                # and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                # and self.field[player.position[0], player.position[1] + 1] == 0
            )

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, self.sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            max_num_food = len(self.players)

            for i in range( len(self.players) ):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = observation.field[y, x]

            for i in range(len(self.players)):
                obs[max_num_food * 3 + 2 * i] = -1
                obs[max_num_food * 3 + 2 * i + 1] = -1

            for i, p in enumerate(seen_players):
                obs[max_num_food * 3 + 2 * i] = p.position[0]
                obs[max_num_food * 3 + 2 * i + 1] = p.position[1]

            return obs

        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward

        observations = [self._make_obs(player) for player in self.players]
        nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [get_player_reward(obs) for obs in observations]
        ndone = [obs.game_over for obs in observations]
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = {}
        
        # check the space of obs
        for i, obs in  enumerate(nobs):
            assert self.observation_space[i].contains(obs), \
                f"obs space error: obs: {obs}, obs_space: {self.observation_space[i]}"
        
        return nobs, nreward, ndone, ninfo
    
    def reset(self):
        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players()
        self.spawn_food( len(self.players) )

        self.current_step = 0
        self._game_over = False

        nobs, _, _, _ = self._make_gym_obs()
        return nobs
    
    def compute_rewards(self, players, actions, swap=False):

        for p in players:
            p.reward = 0

        actions = [
            Action(a) if self._is_valid_action(p,Action(a)) else Action.NONE
            for p, a in zip(players, actions)
        ]

        for player, action in zip(players, actions):
            if action == Action.NORTH:
                player.position = (player.position[0] - 1 , player.position[1])
            elif action == Action.SOUTH:
                player.position = (player.position[0] + 1 , player.position[1] )
            elif action == Action.WEST:
                player.position = (player.position[0], player.position[1] - 1)
            elif action == Action.EAST:
                player.position = (player.position[0], player.position[1] + 1)

            # reward negative manhattan distance from the goals
            player.reward -= sum( np.abs(self.goals[player.level] - player.position) )

            if self._normalize_reward:
                player.reward = player.reward/(sum(self.field.shape)*len(self.players))

        # return reward array if swapping is on
        if swap:
            return [player.reward for player in players]
    
    def swap_players(self, swap, actions):
        players_copy = deepcopy(self.players)
        actions_copy = list(copy(actions))
        actions_copy[swap[0]], actions_copy[swap[1]] = actions_copy[swap[1]], actions_copy[swap[0]] 
        players_copy[swap[0]].position, players_copy[swap[1]].position = players_copy[swap[1]].position, players_copy[swap[0]].position 

        return players_copy, actions_copy

    def reward_mapping_function(self, actions):
        players_copy =  deepcopy(self.players)
        reward_array = np.zeros((len(players_copy), len(players_copy)))

        # diagonal values of reward_array are default reward values
        r_list = self.compute_rewards(players_copy, actions, swap=True)
        for i,r in enumerate(r_list):
            reward_array[i,i] = r
        
        # combination of agents' trajecory sharing for reward mapping
        swap_combs = combinations(list(range(len(self.players))), 2)
        for swap in swap_combs:
            swapped_players, swapped_actions = self.swap_players(swap, actions)
            r_list = self.compute_rewards(swapped_players, swapped_actions, swap=True)
            reward_array[swap[0], swap[1] ] = r_list[swap[0]]
            reward_array[swap[1], swap[0] ] = r_list[swap[1]]

        return reward_array

    def step(self, actions):

        self.current_step += 1

        self.compute_rewards(self.players, actions, swap=False)

        self._game_over = (self._max_episode_steps <= self.current_step)

        for p in self.players:
            p.score += p.reward

        return self._make_gym_obs()

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()