"""Algorithms for policy evaluation."""

from typing import Callable, Dict

from sympy import Eq, Symbol, linsolve

from robot_intake.envs.mdp import MDP, MDPAction, MDPState


def evaluate_policy_linear_system(
    pi: Callable[[MDPState], MDPAction], mdp: MDP
) -> Dict[MDPState, float]:
    """Computes a value function by solving a system of linear equations."""
    # Get states, P, and R.
    S = mdp.state_space
    gamma = mdp.temporal_discount_factor
    P = mdp.get_transition_probability
    R = mdp.get_reward

    # Create symbolic variables for values.
    V = {s: Symbol(f"s{s}") for s in S}

    # Create equations.
    equations = []
    for s, v_s in V.items():
        rhs = sum(P(s, pi(s), ns) * (R(s, pi(s), ns) + gamma * V[ns]) for ns in S)
        equation = Eq(v_s, rhs)
        equations.append(equation)

    # Solve equations.
    ordered_S = sorted(S)
    solutions = linsolve(equations, [V[s] for s in ordered_S])
    solutions = list(solutions)
    assert len(solutions) == 1
    values = solutions[0]

    # Construct value function.
    return {s: float(v) for s, v in zip(ordered_S, values, strict=True)}
