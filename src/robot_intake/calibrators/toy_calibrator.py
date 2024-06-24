"""Domain-specific calibrator for the ToyCalibrativeMDP."""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np

from robot_intake.calibrators.base_calibrator import Calibrator
from robot_intake.envs.calibrative_mdp import CalibrativeAction, Observation
from robot_intake.envs.mdp import MDPPolicy
from robot_intake.envs.toy_calibrative_mdp import (
    ToyCalibrativeMDP,
    _RewardQuestion,
    _TaskQuestion,
    _ToyAction,
    _ToyRobotState,
    _ToyTask,
)
from robot_intake.structs import CategoricalDistribution
from robot_intake.utils import topological_sort


class ToyCalibrator(Calibrator):
    """Domain-specific calibrator for the ToyCalibrativeMDP."""

    def __init__(
        self,
        action_space: Set[_ToyAction],
        task_space: Set[_ToyTask],
        robot_state_transitions: Dict[
            _ToyRobotState, Dict[_ToyAction, CategoricalDistribution[_ToyRobotState]]
        ],
        task_switch_prob: float,
    ) -> None:
        self._action_space = action_space
        self._task_space = task_space
        self._robot_state_transitions = robot_state_transitions
        self._task_switch_prob = task_switch_prob

    def calibrate(self, data: List[Tuple[CalibrativeAction, Observation]]) -> MDPPolicy:
        # Come up with a very coarse approximation of the hidden parameters
        # given the data and then solve the MDP with value iteration.
        task_data: List[Tuple[_TaskQuestion, Observation]] = []
        reward_data: List[Tuple[_RewardQuestion, Observation]] = []
        for (calibrative_action, observation) in data:
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
        import ipdb

        ipdb.set_trace()

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
            else:
                # Assume no equality.
                pairwise_relations.append((question.task2, question.task1))
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
            else:
                # Assume no equality.
                relation = (question.robot2, question.robot1)
            pairwise_relations[question.task].append(relation)
        task_rewards: Dict[_ToyTask, Dict[_ToyRobotState, float]] = {}
        states = sorted(self._robot_state_transitions)
        for task in self._task_space:
            # This is the approximation: linearly increasing rewards.
            ordered_states = topological_sort(states, pairwise_relations[task])
            rewards = np.arange(1, len(ordered_states) + 1)
            rews = dict(zip(ordered_states, rewards))
            task_rewards[task] = rews
        return task_rewards
