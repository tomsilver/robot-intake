"""A hidden-parameter MDP with special calibrative actions and observations."""

import abc
from typing import Generic, Set, TypeAlias, TypeVar

from robot_intake.envs.hip_mdp import HiPMDP
from robot_intake.structs import CategoricalDistribution, HashableComparable

HiddenParameter: TypeAlias = HashableComparable
State: TypeAlias = HashableComparable
Action: TypeAlias = HashableComparable
CalibrativeAction: TypeAlias = HashableComparable
Observation: TypeAlias = HashableComparable

_H = TypeVar("_H", bound=HiddenParameter)
_S = TypeVar("_S", bound=State)
_A = TypeVar("_A", bound=Action)
_C = TypeVar("_C", bound=CalibrativeAction)
_O = TypeVar("_O", bound=Observation)


class CalibrativeMDP(HiPMDP[_H, _S, _A], Generic[_H, _S, _A, _C, _O]):
    """A hidden-parameter MDP with calibrative actions and observations."""

    @property
    @abc.abstractmethod
    def calibrative_action_space(self) -> Set[_C]:
        """Representation of the MDP calibrative action set."""

    @property
    @abc.abstractmethod
    def observation_space(self) -> Set[_O]:
        """Representation of the observation space for calibration."""

    @abc.abstractmethod
    def get_observation_distribution(
        self, hidden_parameter: _H, calibrative_action: _C
    ) -> CategoricalDistribution[_O]:
        """Return a discrete distribution over observations."""
