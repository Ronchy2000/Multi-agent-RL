from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
import numpy as np
from gymnasium.utils import EzPickle


from pettingzoo.mpe.simple.simple import env, parallel_env, raw_env

class PredatorPreyEnv(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_predators=3,
        max_cycles=25,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            num_predators=num_predators,
            max_cycles=max_cycles,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_predators)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=False,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "predator_prey_v1"

env = make_env(PredatorPreyEnv)
parallel_env = parallel_wrapper_fn(env)