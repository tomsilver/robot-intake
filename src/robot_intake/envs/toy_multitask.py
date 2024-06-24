"""A toy CalibrativeMDP with multiple tasks."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Set, TypeAlias

from robot_intake.envs.calibrative_mdp import CalibrativeMDP
from robot_intake.structs import CategoricalDistribution, Image

_ToyRobotState: TypeAlias = int
_ToyTask: TypeAlias = int
_ToyAction: TypeAlias = int


@dataclass(frozen=True, order=True)
class _ToyState:
    """The robot's state and the current task."""

    task: _ToyTask
    robot: _ToyRobotState


@dataclass(frozen=True, order=True)
class _ToyHiddenParameters:
    """Task distribution and within-task rewards."""

    task_probs: Dict[_ToyTask, float]
    task_rewards: Dict[_ToyTask, Dict[_ToyRobotState, float]]

    def __post_init__(self) -> None:
        assert set(self.task_probs) == set(self.task_rewards)


@dataclass(frozen=True, order=True)
class _TaskQuestion:
    """Is task1 more likely than task2 in the task distribution?"""

    task1: _ToyTask
    task2: _ToyTask


@dataclass(frozen=True, order=True)
class _RewardQuestion:
    """Is the reward for robot1 greater than that for robot2 in task?"""

    task: _ToyTask
    robot1: _ToyRobotState
    robot2: _ToyRobotState


@dataclass(frozen=True, order=True)
class _CoinQuestion:
    """Flip a coin and return the boolean resposne."""


_ToyCalibrativeAction: TypeAlias = _TaskQuestion | _RewardQuestion | _CoinQuestion
_ToyObservation: TypeAlias = bool


class ToyCalibrativeMDP(
    CalibrativeMDP[
        _ToyHiddenParameters,
        _ToyState,
        _ToyAction,
        _ToyCalibrativeAction,
        _ToyObservation,
    ]
):
    """A toy CalibrativeMDP with multiple tasks."""

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

    @property
    def calibrative_action_space(self) -> Set[_ToyCalibrativeAction]:
        actions: Set[_ToyCalibrativeAction] = set()

        # Add task questions.
        for task1 in self._task_space:
            for task2 in self._task_space:
                actions.add(_TaskQuestion(task1, task2))

        # Add reward questions.
        for task in self._task_space:
            for rob1 in self._robot_state_transitions:
                for rob2 in self._robot_state_transitions:
                    actions.add(_RewardQuestion(task, rob1, rob2))

        # Add coin flip question.
        actions.add(_CoinQuestion())

        return actions

    @property
    def observation_space(self) -> Set[_ToyObservation]:
        return {True, False}

    def get_observation_distribution(
        self,
        hidden_parameter: _ToyHiddenParameters,
        calibrative_action: _ToyCalibrativeAction,
    ) -> CategoricalDistribution[_ToyObservation]:
        if isinstance(calibrative_action, _TaskQuestion):
            task1, task2 = calibrative_action.task1, calibrative_action.task2
            task1_prob = hidden_parameter.task_probs[task1]
            task2_prob = hidden_parameter.task_probs[task2]
            answer = task1_prob > task2_prob
            return CategoricalDistribution({answer: 1.0, not answer: 0.0})

        if isinstance(calibrative_action, _RewardQuestion):
            task, robot1, robot2 = (
                calibrative_action.task,
                calibrative_action.robot1,
                calibrative_action.robot2,
            )
            robot1_rew = hidden_parameter.task_rewards[task][robot1]
            robot2_rew = hidden_parameter.task_rewards[task][robot2]
            answer = robot1_rew > robot2_rew
            return CategoricalDistribution({answer: 1.0, not answer: 0.0})

        assert isinstance(calibrative_action, _CoinQuestion)
        return CategoricalDistribution({True: 0.5, False: 0.5})

    @property
    def hidden_parameter_space(self) -> Set[_ToyHiddenParameters]:
        raise NotImplementedError("Hidden parameter space cannot be enumerated")

    @property
    def state_space(self) -> Set[_ToyState]:
        states = set()
        for task in self._task_space:
            for robot in self._robot_state_transitions:
                states.add(_ToyState(task, robot))
        return states

    @property
    def action_space(self) -> Set[_ToyAction]:
        return set(self._action_space)

    @property
    def temporal_discount_factor(self) -> float:
        return 1.0

    def get_reward(
        self,
        hidden_parameter: _ToyHiddenParameters,
        state: _ToyState,
        action: _ToyAction,
        next_state: _ToyState,
    ) -> float:
        task = next_state.task
        robot = next_state.robot
        return hidden_parameter.task_rewards[task][robot]

    def get_transition_distribution(
        self,
        hidden_parameter: _ToyHiddenParameters,
        state: _ToyState,
        action: _ToyAction,
    ) -> CategoricalDistribution[_ToyState]:
        robot_dist = self._robot_state_transitions[state.robot][action]
        # Case 1: task switches.
        state_dist: Dict[_ToyState, float] = defaultdict(float)
        for next_task, next_task_prob in hidden_parameter.task_probs.items():
            for next_robot, next_robot_prob in robot_dist.items():
                p = next_task_prob * next_robot_prob * self._task_switch_prob
                state_dist[_ToyState(next_task, next_robot)] += p
        # Case 2: task doesn't switch.
        for next_robot, next_robot_prob in robot_dist.items():
            p = next_robot_prob * (1 - self._task_switch_prob)
            state_dist[_ToyState(state.task, next_robot)] += p
        return CategoricalDistribution(state_dist, normalize=True)

    def render_state(
        self, hidden_parameter: _ToyHiddenParameters, state: _ToyState
    ) -> Image:
        raise NotImplementedError
