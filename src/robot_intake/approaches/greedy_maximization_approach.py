"""An approach that uses greedy maximization to order calibrative actions."""

from typing import Collection, Dict, List

from robot_intake.algorithms.policy_evaluation import evaluate_policy_linear_system
from robot_intake.approaches.base_approach import CalibrativeApproach
from robot_intake.envs.calibrative_mdp import CalibrativeAction, CalibrativeMDP


class GreedyMaximizationCalibrativeApproach(CalibrativeApproach):
    """Uses greedy maximization to order calibrative actions."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._calibrative_action_to_score: Dict[CalibrativeAction, float] = {}
        self._next_calibrative_action_idx = 0

    @property
    def _ordered_calibrative_actions(self) -> List[CalibrativeAction]:
        assert set(self._calibrative_action_space) == set(
            self._calibrative_action_to_score
        )
        key = lambda k: self._calibrative_action_to_score[k]
        return sorted(self._calibrative_action_space, key=key)

    def _train(self, training_envs: Collection[CalibrativeMDP]) -> None:
        for action in sorted(self._calibrative_action_space):
            total_score = 0.0
            for env in training_envs:
                total_score += self._score_action_in_env(action, env)
            self._calibrative_action_to_score[action] = total_score

    def _score_action_in_env(
        self, calibrative_action: CalibrativeAction, env: CalibrativeMDP
    ) -> float:
        # Run one-step calibration.
        obs = env.sample_observation(calibrative_action, self._rng)
        policy = self._calibrator.calibrate([(calibrative_action, obs)])
        # Evaluate the resulting policy.
        values = evaluate_policy_linear_system(policy, env)
        initial_state_dist = env.get_initial_state_distribution()
        value = float(sum(values[s] * p for s, p in initial_state_dist.items()))
        return value

    def _get_calibrative_action(self) -> CalibrativeAction:
        a = self._ordered_calibrative_actions[self._next_calibrative_action_idx]
        self._next_calibrative_action_idx += 1
        return a
