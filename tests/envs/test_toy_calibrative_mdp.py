"""Tests for toy_calibrative_mdp.py."""

import numpy as np

from robot_intake.envs.toy_calibrative_mdp import (
    ToyCalibrativeMDP,
    _CoinQuestion,
    _RewardQuestion,
    _TaskQuestion,
    _ToyHiddenParameters,
)


def test_toy_calibrative_mdp():
    """Tests for toy_calibrative_mdp.py."""
    action_space = {"stay", "move"}
    task_space = {"task0", "task1"}
    robot_state_transitions = {
        0: {
            "stay": {
                0: 1.0,
                1: 0.0,
            },
            "move": {
                0: 0.0,
                1: 1.0,
            },
        },
        1: {
            "stay": {
                0: 0.0,
                1: 1.0,
            },
            "move": {
                0: 1.0,
                1: 0.0,
            },
        },
    }
    task_switch_prob = 0.1
    env = ToyCalibrativeMDP(
        action_space, task_space, robot_state_transitions, task_switch_prob
    )
    num_actions = len(env.action_space)
    num_states = len(robot_state_transitions)
    num_tasks = len(task_space)
    assert len(env.state_space) == num_states * num_tasks
    assert env.action_space == {"stay", "move"}
    assert (
        len(env.calibrative_action_space)
        == num_tasks**2 + num_tasks * (num_actions**2) + 1
    )
    assert env.observation_space == {True, False}
    task_probs = {
        "task0": 0.1,
        "task1": 0.9,
    }
    task_rewards = {
        "task0": {
            0: 1.0,
            1: -1.0,
        },
        "task1": {
            0: -1.0,
            1: 1.0,
        },
    }
    params = _ToyHiddenParameters(task_probs, task_rewards)
    dist = env.get_observation_distribution(params, _TaskQuestion("task0", "task1"))
    assert np.isclose(dist[False], 1.0)
    dist = env.get_observation_distribution(params, _TaskQuestion("task1", "task0"))
    assert np.isclose(dist[True], 1.0)
    dist = env.get_observation_distribution(params, _RewardQuestion("task0", 0, 1))
    assert np.isclose(dist[True], 1.0)
    dist = env.get_observation_distribution(params, _RewardQuestion("task0", 1, 0))
    assert np.isclose(dist[False], 1.0)
    dist = env.get_observation_distribution(params, _CoinQuestion())
    assert np.isclose(dist[True], 0.5)
    mdp = env.to_mdp(params)
    assert mdp.state_space == env.state_space
    assert mdp.action_space == env.action_space
    state = sorted(mdp.state_space)[0]
    action = sorted(mdp.action_space)[0]
    dist = mdp.get_transition_distribution(state, action)
    assert len(list(dist.items())) == 2
    rng = np.random.default_rng(123)
    next_state = mdp.sample_next_state(state, action, rng)
    r = mdp.get_reward(state, action, next_state)
    assert np.isclose(r, 1.0) or np.isclose(r, -1.0)
