"""Experiment showing performance vs calibration period length in toy MDP."""

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib
import numpy as np
import pandas as pd

from robot_intake.approaches.random_approach import RandomCalibrativeApproach
from robot_intake.calibrators.toy_calibrator import (
    ToyCalibrator,
    _ToyAction,
    _ToyRobotState,
    _ToyTask,
)
from robot_intake.envs.toy_calibrative_mdp import ToyCalibrativeMDP
from robot_intake.structs import CategoricalDistribution


def _main(
    start_seed: int,
    num_seeds: int,
    num_robot_states: int,
    num_tasks: int,
    num_actions: int,
    num_evaluation_episodes: int,
    evaluation_horizon: int,
    outdir: Path,
    load: bool,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    csv_file = outdir / "toy_random_experiment.csv"
    if load:
        assert csv_file.exists()
        df = pd.read_csv(csv_file)
        return _df_to_plot(df, outdir)
    columns = ["Seed", "Num Calibration Steps", "Evaluation Performance"]
    results: List[Tuple[int, int, float]] = []
    for num_calibration_steps in [0, 10, 100]:
        print(f"Starting {num_calibration_steps=}")
        for seed in range(start_seed, start_seed + num_seeds):
            print(f"Starting {seed=}")
            result = _run_single(
                seed,
                num_calibration_steps,
                num_robot_states,
                num_tasks,
                num_actions,
                num_evaluation_episodes,
                evaluation_horizon,
            )
            results.append((seed, num_calibration_steps, result))
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(csv_file)
    return _df_to_plot(df, outdir)


def _sample_task_probs(
    task_space: Set[_ToyTask], rng: np.random.Generator
) -> Dict[_ToyTask, float]:
    p = rng.dirichlet(np.ones(len(task_space)))
    ordered_tasks = sorted(task_space)
    return dict(zip(ordered_tasks, p, strict=True))


def _sample_task_rewards(
    task_space: Set[_ToyTask],
    robot_state_space: Set[_ToyRobotState],
    rng: np.random.Generator,
) -> Dict[_ToyTask, Dict[_ToyRobotState, float]]:
    task_rewards = {}
    ordered_robot_states = sorted(robot_state_space)
    for task in sorted(task_space):
        p = rng.normal(loc=0.0, scale=10.0, size=len(ordered_robot_states))
        task_rewards[task] = dict(zip(ordered_robot_states, p))
    return task_rewards


def _sample_robot_state_transitions(
    robot_state_space: Set[_ToyRobotState],
    action_space: Set[_ToyAction],
    rng: np.random.Generator,
) -> Dict[_ToyRobotState, Dict[_ToyAction, CategoricalDistribution[_ToyRobotState]]]:
    transitions: Dict[
        _ToyRobotState, Dict[_ToyAction, CategoricalDistribution[_ToyRobotState]]
    ] = {}
    ordered_robot_states = sorted(robot_state_space)
    ordered_actions = sorted(action_space)
    for state in ordered_robot_states:
        transitions[state] = {}
        for action in ordered_actions:
            p = rng.dirichlet(np.ones(len(ordered_robot_states)))
            d = dict(zip(ordered_robot_states, p, strict=True))
            transitions[state][action] = CategoricalDistribution(d)
    return transitions


def _run_single(
    seed: int,
    num_calibration_steps: int,
    num_robot_states: int,
    num_tasks: int,
    num_actions: int,
    num_evaluation_episodes: int,
    evaluation_horizon: int,
) -> float:
    rng = np.random.default_rng(seed)
    # Create the env.
    task_space = set(range(num_tasks))
    action_space = set(range(num_actions))
    robot_state_space = set(range(num_robot_states))
    task_probs = _sample_task_probs(task_space, rng)
    task_rewards = _sample_task_rewards(task_space, robot_state_space, rng)
    robot_state_transitions = _sample_robot_state_transitions(
        robot_state_space, action_space, rng
    )
    task_switch_prob = 0.1
    env = ToyCalibrativeMDP(
        task_probs,
        task_rewards,
        action_space,
        task_space,
        robot_state_transitions,
        task_switch_prob,
    )
    # Create the calibrator.
    calibrator = ToyCalibrator(
        action_space, task_space, robot_state_transitions, task_switch_prob, rng
    )
    # Create the approach.
    approach = RandomCalibrativeApproach(
        env.state_space,
        env.action_space,
        env.calibrative_action_space,
        env.observation_space,
        calibrator,
        rng,
    )

    # Calibration phase.
    print("Starting calibration phase...")
    rng = np.random.default_rng(rng)
    for _ in range(num_calibration_steps):
        calibrative_action = approach.get_calibrative_action()
        obs = env.sample_observation(calibrative_action, rng)
        approach.observe_calibrative_response(obs)
    print("Finishing calibration...")
    approach.finish_calibration()

    # Evaluation phase.
    print("Starting evaluation phase...")
    rng = np.random.default_rng(rng)
    rews = []
    for _ in range(num_evaluation_episodes):
        state = env.sample_initial_state(rng)
        rew = 0.0
        for _ in range(evaluation_horizon):
            action = approach.step(state)
            next_state = env.sample_next_state(state, action, rng)
            rew += env.get_reward(state, action, next_state)
            state = next_state
        rews.append(rew)
    mean_rew = float(np.mean(rews))
    print("Average rewards:", mean_rew)
    return mean_rew


def _df_to_plot(df: pd.DataFrame, outdir: Path) -> None:
    matplotlib.rcParams.update({"font.size": 20})
    fig_file = outdir / "toy_random_experiment.png"
    print(fig_file)
    print(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_seeds", default=5, type=int)
    parser.add_argument("--num_robot_states", default=5, type=int)
    parser.add_argument("--num_tasks", default=3, type=int)
    parser.add_argument("--num_actions", default=2, type=int)
    parser.add_argument("--num_evaluation_episodes", default=10, type=int)
    parser.add_argument("--evaluation_horizon", default=10, type=int)
    parser.add_argument("--outdir", default=Path("results"), type=Path)
    parser.add_argument("--load", action="store_true")
    parser_args = parser.parse_args()
    _main(
        parser_args.seed,
        parser_args.num_seeds,
        parser_args.num_robot_states,
        parser_args.num_tasks,
        parser_args.num_actions,
        parser_args.num_evaluation_episodes,
        parser_args.evaluation_horizon,
        parser_args.outdir,
        parser_args.load,
    )
