"""An approach that has access to the hidden parameters."""

import numpy as np

from robot_intake.algorithms.value_iteration import (
    value_function_to_greedy_policy,
    value_iteration,
)
from robot_intake.approaches.random_approach import RandomCalibrativeApproach
from robot_intake.envs.calibrative_mdp import CalibrativeMDP


class OracleApproach(RandomCalibrativeApproach):
    """An approach that has access to the hidden parameters."""

    def __init__(
        self,
        env: CalibrativeMDP,
        rng: np.random.Generator,
    ) -> None:
        self._env = env
        calibrator = None
        super().__init__(
            env.state_space,
            env.action_space,
            env.calibrative_action_space,
            env.observation_space,
            calibrator,  # type: ignore
            rng,
        )

    def finish_calibration(self) -> None:
        value_fn = value_iteration(self._env)
        self._policy = value_function_to_greedy_policy(value_fn, self._env, self._rng)