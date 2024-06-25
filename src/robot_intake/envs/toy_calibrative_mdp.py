"""A toy CalibrativeMDP with multiple tasks."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Set, TypeAlias

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


@dataclass(frozen=True)
class _TaskQuestion:
    """Is task1 more likely than task2 in the task distribution?"""

    task1: _ToyTask
    task2: _ToyTask

    def __lt__(self, other: Any) -> bool:
        return str(self) < str(other)


@dataclass(frozen=True)
class _RewardQuestion:
    """Is the reward for robot1 greater than that for robot2 in task?"""

    task: _ToyTask
    robot1: _ToyRobotState
    robot2: _ToyRobotState

    def __lt__(self, other: Any) -> bool:
        return str(self) < str(other)


@dataclass(frozen=True)
class _CoinQuestion:
    """Flip a coin and return the boolean resposne."""

    def __lt__(self, other: Any) -> bool:
        return str(self) < str(other)


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
        task_probs: Dict[_ToyTask, float],
        task_rewards: Dict[_ToyTask, Dict[_ToyRobotState, float]],
        action_space: Set[_ToyAction],
        task_space: Set[_ToyTask],
        robot_state_transitions: Dict[
            _ToyRobotState, Dict[_ToyAction, CategoricalDistribution[_ToyRobotState]]
        ],
        task_switch_prob: float,
    ) -> None:
        self._task_probs = task_probs
        self._task_rewards = task_rewards
        self._action_space = action_space
        self._task_space = task_space
        self._robot_state_transitions = robot_state_transitions
        self._task_switch_prob = task_switch_prob

    @property
    def _hidden_parameter(self) -> _ToyHiddenParameters:
        return _ToyHiddenParameters(self._task_probs, self._task_rewards)

    def get_initial_state_distribution(self) -> CategoricalDistribution[_ToyState]:
        # Always start out at the first robot state.
        dist: Dict[_ToyState, float] = {}
        init_robot_state = min(self._robot_state_transitions)
        for task, task_prob in self._hidden_parameter.task_probs.items():
            dist[_ToyState(task, init_robot_state)] = task_prob
        return CategoricalDistribution(dist)

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

    def _get_observation_distribution(
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
        return 0.99

    def _get_reward(
        self,
        hidden_parameter: _ToyHiddenParameters,
        state: _ToyState,
        action: _ToyAction,
        next_state: _ToyState,
    ) -> float:
        task = next_state.task
        robot = next_state.robot
        return hidden_parameter.task_rewards[task][robot]

    def _get_transition_distribution(
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

    def render_state(self, state: _ToyState) -> Image:
        raise NotImplementedError
