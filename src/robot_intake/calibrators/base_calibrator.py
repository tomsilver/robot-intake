"""Creates a policy given data from the calibration phase."""

import abc
from typing import Callable, Generic, List, Tuple, TypeVar

from robot_intake.envs.calibrative_mdp import CalibrativeAction, Observation
from robot_intake.envs.mdp import MDPAction, MDPState

_S = TypeVar("_S", bound=MDPState)
_A = TypeVar("_A", bound=MDPAction)
_C = TypeVar("_C", bound=CalibrativeAction)
_O = TypeVar("_O", bound=Observation)


class Calibrator(Generic[_S, _A, _C, _O]):
    """Creates a policy given data from the calibration phase."""

    @abc.abstractmethod
    def calibrate(self, data: List[Tuple[_C, _O]]) -> Callable[[_S], _A]:
        """Creates a policy given data from the calibration phase."""
