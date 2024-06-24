"""Hidden-parameter MDP."""

from __future__ import annotations

import abc
from typing import Generic, Set, TypeAlias, TypeVar

from robot_intake.envs.mdp import MDP, MDPAction, MDPState
from robot_intake.structs import CategoricalDistribution, HashableComparable, Image

HiddenParameter: TypeAlias = HashableComparable

_H = TypeVar("_H", bound=HiddenParameter)
_S = TypeVar("_S", bound=MDPState)
_A = TypeVar("_A", bound=MDPAction)


class HiPMDP(Generic[_H, _S, _A], MDP[_S, _A]):
    """An infinite-horizon hidden-parameter Markov Decision Process."""

    @property
    @abc.abstractmethod
    def _hidden_parameter(self) -> _H:
        """The current hidden parameter."""

    @property
    @abc.abstractmethod
    def hidden_parameter_space(self) -> Set[_H]:
        """Representation of the hidden parameter space."""

    @property
    @abc.abstractmethod
    def state_space(self) -> Set[_S]:
        """Representation of the MDP state set."""

    @property
    @abc.abstractmethod
    def action_space(self) -> Set[_A]:
        """Representation of the MDP action set."""

    @property
    def temporal_discount_factor(self) -> float:
        """Gamma, defaults to 1."""
        return 1.0

    def get_transition_distribution(
        self, state: _S, action: _A
    ) -> CategoricalDistribution[_S]:
        """Return a discrete distribution over next states."""
        return self._get_transition_distribution(self._hidden_parameter, state, action)

    @abc.abstractmethod
    def _get_transition_distribution(
        self, hidden_parameter: _H, state: _S, action: _A
    ) -> CategoricalDistribution[_S]:
        """Return a discrete distribution over next states."""

    def get_reward(self, state: _S, action: _A, next_state: _S) -> float:
        """Return (deterministic) reward for executing action in state."""
        return self._get_reward(self._hidden_parameter, state, action, next_state)

    @abc.abstractmethod
    def _get_reward(
        self, hidden_parameter: _H, state: _S, action: _A, next_state: _S
    ) -> float:
        """Return (deterministic) reward for executing action in state."""

    @abc.abstractmethod
    def render_state(self, state: _S) -> Image:
        """Optional rendering function for visualizations."""
