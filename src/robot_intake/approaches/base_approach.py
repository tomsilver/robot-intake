"""Base class for approaches to solving CalibrativeMDPs."""

import abc
from typing import List, Set, Tuple

from robot_intake.calibrators.base_calibrator import Calibrator
from robot_intake.envs.calibrative_mdp import CalibrativeAction, Observation
from robot_intake.envs.mdp import MDPAction, MDPPolicy, MDPState


class CalibrativeApproach(abc.ABC):
    """Base class for approaches to solving CalibrativeMDPs."""

    def __init__(
        self,
        state_space: Set[MDPState],
        action_space: Set[MDPAction],
        calibrative_action_space: Set[CalibrativeAction],
        observation_space: Set[Observation],
        calibrator: Calibrator,
    ) -> None:
        self._state_space = state_space
        self._action_space = action_space
        self._calibrative_action_space = calibrative_action_space
        self._observation_space = observation_space
        self._calibrator = calibrator
        self._last_calibrative_action: CalibrativeAction | None = None
        self._calibration_data: List[Tuple[CalibrativeAction, Observation]] = []
        self._policy: MDPPolicy | None = None

    def get_calibrative_action(self) -> CalibrativeAction:
        """Called during the calibration phase."""
        self._last_calibrative_action = self._get_calibrative_action()
        return self._last_calibrative_action

    @abc.abstractmethod
    def _get_calibrative_action(self) -> CalibrativeAction:
        """Called during the calibration phase."""

    def observe_calibrative_response(self, obs: Observation) -> None:
        """Called during the calibration phase."""
        assert self._last_calibrative_action is not None
        self._calibration_data.append((self._last_calibrative_action, obs))

    def finish_calibration(self) -> None:
        """Called at the end of the calibration phase."""
        self._policy = self._calibrator.calibrate(self._calibration_data)

    def step(self, state: MDPState) -> MDPAction:
        """Called during evaluation, after calibration."""
        assert self._policy is not None
        return self._policy(state)
