# rl/attention_policy.py
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np
from typing import Tuple


class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that uses a Transformer Encoder to process a variable number of stocks.
    (This class remains the same and is correct)
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # The 'features_dim' passed here is the desired output dim for the final MLP heads,
        # but our transformer has its own internal dimension. We will use the one calculated.
        super().__init__(observation_space, features_dim)


        num_features = observation_space["features"].shape[-1]
        window_size = observation_space["features"].shape[-2]

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features, out_channels=64, kernel_size=3, padding=1
            ),

            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_features, window_size)
            cnn_output_dim = self.cnn(dummy_input).shape[1]


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_output_dim,
            nhead=4,
            dim_feedforward=features_dim,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # IMPORTANT: This exposes the true feature dimension of our transformer's output
        self._features_dim = cnn_output_dim

    def forward(self, observations):
        features = observations["features"]
        mask = observations["mask"]

        batch_size, num_assets, window_size, num_asset_features = features.shape
        features = features.view(
            batch_size * num_assets, window_size, num_asset_features
        ).permute(0, 2, 1)

        stock_features = self.cnn(features)
        stock_features = stock_features.view(batch_size, num_assets, -1)

        padding_mask = mask == 0
        transformer_output = self.transformer_encoder(
            stock_features, src_key_padding_mask=padding_mask
        )

        # We now return the mask as well, as it's needed in the policy's forward pass
        return transformer_output, mask


class AttentionPolicy(ActorCriticPolicy):
    """
    Custom policy using the AttentionFeaturesExtractor.
    This version lets the parent class build its default networks, then overwrites them.
    """

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        self.top_k_stocks = action_space.shape[0]


        # --- THE FIX ---
        # 1. Let the parent class run its full initialization.
        # It will create a default mlp_extractor, action_net, and value_net that we will ignore and overwrite.
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        # 2. Now, overwrite the networks with our custom architecture.
        # The input dimension depends on the output of our custom features_extractor.
        transformer_feature_dim = self.features_extractor.features_dim
        mlp_input_dim = self.top_k_stocks * transformer_feature_dim

        self.action_net = nn.Sequential(

            nn.Linear(mlp_input_dim, 128), nn.ReLU(), nn.Linear(128, self.top_k_stocks)
        )
        self.value_net = nn.Sequential(
            nn.Linear(mlp_input_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    # We DO NOT override _build_mlp_extractor(). We are overwriting the final networks directly.

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The latent_features are the output from our AttentionFeaturesExtractor
        latent_features, mask = self.features_extractor(obs)


        # SCREENING/RANKING
        scores = latent_features.mean(dim=-1)
        scores[mask == 0] = -torch.inf

        # SELECTION
        _, top_k_indices = torch.topk(scores, self.top_k_stocks, dim=1)


        # GATHER FEATURES
        top_k_features = torch.gather(
            latent_features,
            1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, latent_features.shape[-1]),
        )

        # ALLOCATION
        flat_top_k_features = top_k_features.flatten(start_dim=1)

        values = self.value_net(flat_top_k_features)
        mean_actions = self.action_net(flat_top_k_features)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)


        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob


    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: The observation dictionary.
        :param actions: The allocation actions taken in the rollout buffer.
        :return: The value of the state, the log probability of the action, and the entropy of the action distribution.
        """
        # --- THIS IS THE NEW METHOD ---
        # It follows the same logic as forward() and predict_values()
        latent_features, mask = self.features_extractor(obs)


        scores = latent_features.mean(dim=-1)
        scores[mask == 0] = -torch.inf
        _, top_k_indices = torch.topk(scores, self.top_k_stocks, dim=1)

        top_k_features = torch.gather(
            latent_features,
            1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, latent_features.shape[-1]),
        )
        flat_top_k_features = top_k_features.flatten(start_dim=1)

        mean_actions = self.action_net(flat_top_k_features)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)

        log_prob = distribution.log_prob(actions)
        values = self.value_net(flat_top_k_features)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Custom predict_values method to handle our specific data flow.
        It performs the same ranking and selection as the main forward pass.

        :param obs: The observation dictionary.
        :return: The estimated value of the observation.
        """
        # Run the feature extractor
        latent_features, mask = self.features_extractor(obs)

        # Perform the same ranking and selection logic
        scores = latent_features.mean(dim=-1)
        scores[mask == 0] = -torch.inf
        _, top_k_indices = torch.topk(scores, self.top_k_stocks, dim=1)
        top_k_features = torch.gather(

            latent_features,
            1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, latent_features.shape[-1]),
        )
        flat_top_k_features = top_k_features.flatten(start_dim=1)

        # Pass the correctly formatted tensor to the value network
        return self.value_net(flat_top_k_features)

    def predict(
        self,
        observation: np.ndarray,
        state: None = None,
        episode_start: None = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, None]:
        # (This method remains the same and is correct)
        if observation["features"].ndim == 3:
            observation = spaces.unflatten(self.observation_space, observation)
            observation = {
                key: val[np.newaxis, ...] for key, val in observation.items()
            }

        obs_tensor = {
            key: torch.as_tensor(val).to(self.device)
            for key, val in observation.items()
        }


        with torch.no_grad():
            latent_features, mask = self.features_extractor(obs_tensor)
            scores = latent_features.mean(dim=-1)
            scores[mask == 0] = -torch.inf
            top_k_indices = torch.topk(scores, self.top_k_stocks, dim=1)[1]

            top_k_features = torch.gather(
                latent_features,
                1,
                top_k_indices.unsqueeze(-1).expand(-1, -1, latent_features.shape[-1]),
            )
            flat_top_k_features = top_k_features.flatten(start_dim=1)
            mean_actions = self.action_net(flat_top_k_features)
            distribution = self.action_dist.proba_distribution(
                mean_actions, self.log_std
            )
            allocations = distribution.get_actions(deterministic=deterministic)

        compound_action = (top_k_indices.cpu().numpy()[0], allocations.cpu().numpy()[0])
        return compound_action, None

