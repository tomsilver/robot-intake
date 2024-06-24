"""Hidden-parameter MDP."""

from __future__ import annotations

import abc
from typing import Generic, Set, TypeAlias, TypeVar

import numpy as np

from robot_intake.envs.mdp import MDP, MDPAction, MDPState
from robot_intake.structs import CategoricalDistribution, HashableComparable, Image

HiddenParameter: TypeAlias = HashableComparable

_H = TypeVar("_H", bound=HiddenParameter)
_S = TypeVar("_S", bound=MDPState)
_A = TypeVar("_A", bound=MDPAction)


class HiPMDP(Generic[_H, _S, _A]):
    """An infinite-horizon hidden-parameter Markov Decision Process."""

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

    @abc.abstractmethod
    def get_reward(
        self, hidden_parameter: _H, state: _S, action: _A, next_state: _S
    ) -> float:
        """Return (deterministic) reward for executing action in state."""

    @abc.abstractmethod
    def get_transition_distribution(
        self, hidden_parameter: _H, state: _S, action: _A
    ) -> CategoricalDistribution[_S]:
        """Return a discrete distribution over next states."""

    def sample_next_state(
        self, hidden_parameter: _H, state: _S, action: _A, rng: np.random.Generator
    ) -> _S:
        """Sample a next state from the transition distribution.

        This function may be overwritten by subclasses when the explicit
        distribution is too large to enumerate.
        """
        return self.get_transition_distribution(hidden_parameter, state, action).sample(
            rng
        )

    def get_transition_probability(
        self, hidden_parameter: _H, state: _S, action: _A, next_state: _S
    ) -> float:
        """Convenience method for some algorithms."""
        return self.get_transition_distribution(hidden_parameter, state, action)(
            next_state
        )

    @abc.abstractmethod
    def render_state(self, hidden_parameter: _H, state: _S) -> Image:
        """Optional rendering function for visualizations."""

    def to_mdp(self, hidden_parameter: _H) -> MDPFromHiPMDP[_S, _A]:
        """Create an MDP given a hidden parameter."""
        return MDPFromHiPMDP(self, hidden_parameter)


class MDPFromHiPMDP(MDP[_S, _A]):
    """Creates an MDP from a HiPMDP with a given hidden parameter."""

    def __init__(self, hip_mdp: HiPMDP[_H, _S, _A], hidden_parameter: _H) -> None:
        self._hip_mdp = hip_mdp
        self._hidden_parameter = hidden_parameter

    @property
    def state_space(self) -> Set[_S]:
        return self._hip_mdp.state_space

    @property
    def action_space(self) -> Set[_A]:
        return self._hip_mdp.action_space

    @property
    def temporal_discount_factor(self) -> float:
        return self._hip_mdp.temporal_discount_factor

    def get_reward(self, state: _S, action: _A, next_state: _S) -> float:
        return self._hip_mdp.get_reward(
            self._hidden_parameter, state, action, next_state
        )

    def get_transition_distribution(
        self, state: _S, action: _A
    ) -> CategoricalDistribution[_S]:
        return self._hip_mdp.get_transition_distribution(
            self._hidden_parameter, state, action
        )

    def render_state(self, state: _S) -> Image:
        return self._hip_mdp.render_state(self._hidden_parameter, state)
