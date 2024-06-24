"""An approach that takes random calibrative actions."""

from functools import cached_property
from typing import List

from robot_intake.approaches.base_approach import CalibrativeApproach
from robot_intake.envs.calibrative_mdp import CalibrativeAction


class RandomCalibrativeApproach(CalibrativeApproach):
    """An approach that takes random calibrative actions."""

    @cached_property
    def _ordered_calibrative_actions(self) -> List[CalibrativeAction]:
        return sorted(self._calibrative_action_space)

    def _get_calibrative_action(self) -> CalibrativeAction:
        idx = self._rng.choice(len(self._ordered_calibrative_actions))
        return self._ordered_calibrative_actions[idx]
