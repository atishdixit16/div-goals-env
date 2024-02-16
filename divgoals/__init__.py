from gym.envs.registration import registry, register, make, spec
from itertools import product

sizes = range(5, 15)
players = range(2, 10)
partial_obs = [True, False]
pens = [True, False]

for s, p, po, pen in product(sizes, players, partial_obs, pens):
    register(
        id="DivGoals{2}-{0}x{0}-{1}p{3}-v1".format(s, p, "-2s" if po else "", "-pen" if pen else ""),
        entry_point="divgoals.environment:DivGoalsEnv",
        kwargs={
            "players": p,
            "field_size": (s, s),
            "sight": 2 if po else s,
            "max_episode_steps": 25,
            "penalty": 1.0 if pen else 0.0,
        },
    )
