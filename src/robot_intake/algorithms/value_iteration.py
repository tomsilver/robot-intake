"""Value iteration."""

from dataclasses import dataclass
from typing import Dict

import numpy as np

from robot_intake.envs.mdp import MDP, MDPAction, MDPPolicy, MDPState
from robot_intake.structs import Hyperparameters


@dataclass(frozen=True)
class ValueIterationHyperparameters(Hyperparameters):
    """Hyperparameters for value iteration."""

    max_num_iterations: int = 1000
    change_threshold: float = 1e-4
    print_every: int | None = None


def bellman_backup(s: MDPState, V: Dict[MDPState, float], mdp: MDP) -> float:
    """Look ahead one step and propose an update for the value of s."""
    vs = -float("inf")
    for a in mdp.action_space:
        qsa = 0.0
        for ns, p in mdp.get_transition_distribution(s, a).items():
            r = mdp.get_reward(s, a, ns)
            qsa += p * (r + mdp.temporal_discount_factor * V[ns])
        vs = max(qsa, vs)
    return vs


def value_function_to_greedy_policy(
    V: Dict[MDPState, float], mdp: MDP, rng: np.random.Generator
) -> MDPPolicy:
    """Create a greedy policy given a value function."""
    gamma = mdp.temporal_discount_factor
    P = mdp.get_transition_distribution
    R = mdp.get_reward

    # Note: do not call value_to_action_value_function() here because we can
    # avoid enumerating the state space.
    def Q(s: MDPState, a: MDPAction) -> float:
        """Shorthand for the action-value function."""
        return sum(P(s, a)(ns) * (R(s, a, ns) + gamma * V[ns]) for ns in P(s, a))

    def pi(s: MDPState) -> MDPAction:
        """The greedy policy."""
        # Break ties randomly on actions.
        return max(mdp.action_space, key=lambda a: (Q(s, a), rng.uniform()))

    return pi


def value_iteration(
    mdp: MDP,
    config: ValueIterationHyperparameters | None = None,
) -> Dict[MDPState, float]:
    """Run value iteration for a certain number of iterations or until the max
    change between iterations is below a threshold."""
    if config is None:
        config = ValueIterationHyperparameters()

    # Initialize V to all zeros.
    V = {s: 0.0 for s in mdp.state_space}

    for it in range(config.max_num_iterations):
        next_V = {}
        max_change = 0.0
        for s in mdp.state_space:
            next_V[s] = bellman_backup(s, V, mdp)
            max_change = max(abs(next_V[s] - V[s]), max_change)

        V = next_V

        # Check if we can terminate early.
        if config.print_every is not None and it % config.print_every == 0:
            print(f"VI max change after iteration {it} : {max_change}")

        if max_change < config.change_threshold:
            break

    return V
