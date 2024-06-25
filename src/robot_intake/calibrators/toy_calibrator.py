"""Domain-specific calibrator for the ToyCalibrativeMDP."""

from collections import defaultdict
from typing import Callable, Dict, List, Set, Tuple

import numpy as np

from robot_intake.algorithms.value_iteration import (
    value_function_to_greedy_policy,
    value_iteration,
)
from robot_intake.calibrators.base_calibrator import Calibrator
from robot_intake.envs.calibrative_mdp import Observation
from robot_intake.envs.toy_calibrative_mdp import (
    ToyCalibrativeMDP,
    _RewardQuestion,
    _TaskQuestion,
    _ToyAction,
    _ToyCalibrativeAction,
    _ToyObservation,
    _ToyRobotState,
    _ToyState,
    _ToyTask,
)
from robot_intake.structs import CategoricalDistribution
from robot_intake.utils import topological_sort


class ToyCalibrator(
    Calibrator[_ToyState, _ToyAction, _ToyCalibrativeAction, _ToyObservation]
):
    """Domain-specific calibrator for the ToyCalibrativeMDP."""

    def __init__(
        self,
        action_space: Set[_ToyAction],
        task_space: Set[_ToyTask],
        robot_state_transitions: Dict[
            _ToyRobotState, Dict[_ToyAction, CategoricalDistribution[_ToyRobotState]]
        ],
        task_switch_prob: float,
        seed: int,
    ) -> None:
        self._action_space = action_space
        self._task_space = task_space
        self._robot_state_transitions = robot_state_transitions
        self._task_switch_prob = task_switch_prob
        self._seed = seed

    def calibrate(
        self, data: List[Tuple[_ToyCalibrativeAction, _ToyObservation]]
    ) -> Callable[[_ToyState], _ToyAction]:
        # Come up with a very coarse approximation of the hidden parameters
        # given the data and then solve the MDP with value iteration.
        task_data: List[Tuple[_TaskQuestion, Observation]] = []
        reward_data: List[Tuple[_RewardQuestion, Observation]] = []
        for calibrative_action, observation in data:
            if isinstance(calibrative_action, _TaskQuestion):
                task_data.append((calibrative_action, observation))
            elif isinstance(calibrative_action, _RewardQuestion):
                reward_data.append((calibrative_action, observation))
        task_probs = self._infer_task_probs(task_data)
        task_rewards = self._infer_task_rewards(reward_data)
        mdp = ToyCalibrativeMDP(
            task_probs,
            task_rewards,
            self._action_space,
            self._task_space,
            self._robot_state_transitions,
            self._task_switch_prob,
        )
        value_fn = value_iteration(mdp)
        tiebreak_rng = np.random.default_rng(self._seed)
        policy = value_function_to_greedy_policy(value_fn, mdp, tiebreak_rng)
        return policy  # type: ignore

    def _infer_task_probs(
        self, data: List[Tuple[_TaskQuestion, Observation]]
    ) -> Dict[_ToyTask, float]:
        pairwise_relations: List[Tuple[_ToyTask, _ToyTask]] = []
        for question, response in data:
            # Ignore nonsense.
            if question.task1 == question.task2:
                continue
            if response:
                pairwise_relations.append((question.task1, question.task2))
        ordered_tasks = topological_sort(self._task_space, pairwise_relations)
        # This is the approximation: linearly increasing probabilities.
        unnormed_probs = np.arange(1, len(ordered_tasks) + 1)
        z = sum(unnormed_probs)
        probs = [p / z for p in unnormed_probs]
        return dict(zip(ordered_tasks, probs))

    def _infer_task_rewards(
        self, data: List[Tuple[_RewardQuestion, Observation]]
    ) -> Dict[_ToyTask, Dict[_ToyRobotState, float]]:
        pairwise_relations: Dict[
            _ToyTask, List[Tuple[_ToyRobotState, _ToyRobotState]]
        ] = defaultdict(list)
        for question, response in data:
            # Ignore nonsense.
            if question.robot1 == question.robot2:
                continue
            if response:
                relation = (question.robot1, question.robot2)
                pairwise_relations[question.task].append(relation)
        task_rewards: Dict[_ToyTask, Dict[_ToyRobotState, float]] = {}
        states = sorted(self._robot_state_transitions)
        # Use knowledge of reward distribution.
        for task in self._task_space:
            ordered_states = topological_sort(states, pairwise_relations[task])
            bad_state, good_state = ordered_states[0], ordered_states[-1]
            d = {s: 0.0 for s in ordered_states}
            d[good_state] = 10.0
            d[bad_state] = -10.0
            task_rewards[task] = d
        return task_rewards
