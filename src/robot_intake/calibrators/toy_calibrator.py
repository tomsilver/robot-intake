"""Domain-specific calibrator for the ToyCalibrativeMDP."""

from typing import List, Tuple

from robot_intake.calibrators.base_calibrator import Calibrator
from robot_intake.envs.calibrative_mdp import CalibrativeAction, Observation
from robot_intake.envs.mdp import MDPPolicy


class ToyCalibrativeMDP(Calibrator):
    """Domain-specific calibrator for the ToyCalibrativeMDP."""

    def calibrate(self, data: List[Tuple[CalibrativeAction, Observation]]) -> MDPPolicy:
        import ipdb

        ipdb.set_trace()
