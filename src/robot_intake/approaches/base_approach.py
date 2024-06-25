"""Base class for approaches to solving CalibrativeMDPs."""

import abc
from typing import Callable, Generic, List, Set, Tuple, TypeVar

import numpy as np

from robot_intake.calibrators.base_calibrator import Calibrator
from robot_intake.envs.calibrative_mdp import CalibrativeAction, Observation
from robot_intake.envs.mdp import MDPAction, MDPState

_S = TypeVar("_S", bound=MDPState)
_A = TypeVar("_A", bound=MDPAction)
_C = TypeVar("_C", bound=CalibrativeAction)
_O = TypeVar("_O", bound=Observation)


class CalibrativeApproach(Generic[_S, _A, _C, _O]):
    """Base class for approaches to solving CalibrativeMDPs."""

    def __init__(
        self,
        state_space: Set[_S],
        action_space: Set[_A],
        calibrative_action_space: Set[_C],
        observation_space: Set[_O],
        calibrator: Calibrator,
        seed: int,
    ) -> None:
        self._state_space = state_space
        self._action_space = action_space
        self._calibrative_action_space = calibrative_action_space
        self._observation_space = observation_space
        self._calibrator = calibrator
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._last_calibrative_action: _C | None = None
        self._calibration_data: List[Tuple[_C, _O]] = []
        self._policy: Callable[[_S], _A] | None = None

    def get_calibrative_action(self) -> _C:
        """Called during the calibration phase."""
        self._last_calibrative_action = self._get_calibrative_action()
        return self._last_calibrative_action

    @abc.abstractmethod
    def _get_calibrative_action(self) -> _C:
        """Called during the calibration phase."""

    def observe_calibrative_response(self, obs: _O) -> None:
        """Called during the calibration phase."""
        assert self._last_calibrative_action is not None
        self._calibration_data.append((self._last_calibrative_action, obs))

    def finish_calibration(self) -> None:
        """Called at the end of the calibration phase."""
        self._policy = self._calibrator.calibrate(self._calibration_data)

    def step(self, state: _S) -> _A:
        """Called during evaluation, after calibration."""
        assert self._policy is not None
        return self._policy(state)
