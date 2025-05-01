from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.policy.policy import Policy
import torch
import numpy as np

class NonLearningPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def get_weights(self):
        return {}

    def learn_on_batch(self, samples):
        return {}

    def set_weights(self, weights):
        pass

class DoNothingPolicy(NonLearningPolicy):
    def compute_actions(self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        return [-np.ones(1, dtype=int) for _ in obs_batch], [], {}

class GoStraight(NonLearningPolicy):
    def compute_actions(self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        return [4*np.ones(1, dtype=int) for _ in obs_batch], [], {}
        # return [4 for _ in obs_batch], [], {}

class NaiveRetrieve(NonLearningPolicy):
    def compute_actions(self,
        obs_batch,
        state_batches,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs
    ):
        actions = []
        for obs in obs_batch:
            if obs[15] == 1: # has flag
                bearing = obs[2] * 180 # Own home
            else: # doesn't have flag
                bearing = obs[0] * 180 # Enemy home
            # Round to nearest 45 degree angle
            bearing_rounded = 45.0 * np.round(bearing / 45.0)
            if bearing_rounded == 180.0:
                action = 0
            elif bearing_rounded == 135:
                action = 1
            elif bearing_rounded == 90:
                action = 2
            elif bearing_rounded == 45:
                action = 3
            elif bearing_rounded == 0:
                action = 4
            elif bearing_rounded == -45:
                action = 5
            elif bearing_rounded == -90:
                action = 6
            elif bearing_rounded == -135:
                action = 7
            elif bearing_rounded == -180:
                action = 0
            else:
                raise ValueError(f"oops: {bearing}, {bearing_rounded}")
            actions.append(action)
        # actions = [action*np.ones(1, dtype=int) for action in actions]
        return actions, [], {}
