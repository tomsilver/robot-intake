"""Tests for random_approach.py."""

import numpy as np

from robot_intake.approaches.random_approach import RandomCalibrativeApproach
from robot_intake.calibrators.toy_calibrator import ToyCalibrator
from robot_intake.envs.toy_calibrative_mdp import ToyCalibrativeMDP


def test_random_approach():
    """Tests for random_approach.py."""
    # Create the environment.
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
    env = ToyCalibrativeMDP(
        task_probs,
        task_rewards,
        action_space,
        task_space,
        robot_state_transitions,
    )
    # Create the calibrator.
    seed = 123
    calibrator = ToyCalibrator(action_space, task_space, robot_state_transitions, seed)
    # Create the approach.
    approach = RandomCalibrativeApproach(
        env.state_space,
        env.action_space,
        env.calibrative_action_space,
        env.observation_space,
        calibrator,
        seed,
    )

    # Calibration phase.
    rng = np.random.default_rng(seed)
    for _ in range(10):
        calibrative_action = approach.get_calibrative_action()
        obs = env.sample_observation(calibrative_action, rng)
        approach.observe_calibrative_response(obs)
    approach.calibrate()

    # Evaluation phase.
    state = env.sample_initial_state(rng)
    rew = 0.0
    for _ in range(10):
        action = approach.step(state)
        next_state = env.sample_next_state(state, action, rng)
        rew += env.get_reward(state, action, next_state)
        state = next_state
