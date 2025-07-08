# rl/custom_ppo.py

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.utils import obs_as_tensor

from stable_baselines3.common.buffers import RolloutBuffer
from gymnasium import spaces


class CustomPPO(PPO):
    """
    A custom PPO class that is able to handle a compound action space,
    where the action passed to the environment is a tuple.
    """

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the custom policy and store them into a ``RolloutBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)


                # The policy's `predict` method returns our custom tuple.
                actions_tuple, _states = self.policy.predict(
                    obs_tensor, deterministic=False
                )


                # Get allocations, values, and log_probs for the buffer.
                allocations, values, log_probs = self.policy(obs_tensor)

            # --- THE FIX IS HERE ---
            # The VecEnv wrapper expects a NumPy array where the first dimension is the number of environments.
            # It will misinterpret our action tuple. To prevent this, we wrap our tuple in a
            # NumPy object array of shape (num_envs,).
            actions_for_vec_env = np.empty(env.num_envs, dtype=object)
            actions_for_vec_env[0] = actions_tuple


            # Pass the protected action to the environment
            new_obs, rewards, dones, infos = env.step(actions_for_vec_env)


            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                allocations = allocations.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):

                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]

                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                allocations.cpu().numpy(),
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()


        return True

