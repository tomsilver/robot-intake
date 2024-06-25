"""An approach that has access to the hidden parameters."""

from typing import Collection

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
        seed: int,
    ) -> None:
        self._env = env
        calibrator = None
        super().__init__(
            env.state_space,
            env.action_space,
            env.calibrative_action_space,
            env.observation_space,
            calibrator,  # type: ignore
            seed,
        )

    def _train(self, training_envs: Collection[CalibrativeMDP]) -> None:
        pass

    def finish_calibration(self) -> None:
        value_fn = value_iteration(self._env)
        tiebreak_rng = np.random.default_rng(self._seed)
        self._policy = value_function_to_greedy_policy(
            value_fn, self._env, tiebreak_rng
        )
