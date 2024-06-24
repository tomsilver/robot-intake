"""Creates a policy given data from the calibration phase."""

import abc
from typing import List, Tuple

from robot_intake.envs.calibrative_mdp import CalibrativeAction, Observation
from robot_intake.envs.mdp import MDPPolicy


class Calibrator(abc.ABC):
    """Creates a policy given data from the calibration phase."""

    @abc.abstractmethod
    def calibrate(self, data: List[Tuple[CalibrativeAction, Observation]]) -> MDPPolicy:
        """Creates a policy given data from the calibration phase."""
