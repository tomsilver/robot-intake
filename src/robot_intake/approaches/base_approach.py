"""Base class for approaches to solving CalibrativeMDPs."""

import abc
from typing import Set

from robot_intake.envs.calibrative_mdp import CalibrativeAction, Observation
from robot_intake.envs.mdp import MDPAction, MDPState


class CalibrativeApproach(abc.ABC):
    """Base class for approaches to solving CalibrativeMDPs."""

    def __init__(
        self,
        state_space: Set[MDPState],
        action_space: Set[MDPAction],
        calibrative_action_space: Set[CalibrativeAction],
        observation_space: Set[Observation],
    ) -> None:
        self._state_space = state_space
        self._action_space = action_space
        self._calibrative_action_space = calibrative_action_space
        self._observation_space = observation_space

    @abc.abstractmethod
    def get_calibrative_action(self) -> CalibrativeAction:
        """Called during the calibration phase."""

    @abc.abstractmethod
    def observe_calibrative_response(self, obs: Observation) -> None:
        """Called during the calibration phase."""

    @abc.abstractmethod
    def finish_calibration(self) -> None:
        """Called at the end of the calibration phase."""

    @abc.abstractmethod
    def step(self, state: MDPState) -> MDPAction:
        """Called during evaluation, after calibration."""
