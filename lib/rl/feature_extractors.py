import numpy as np
import torch
import torch.nn as nn
import gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from nip import nip

from lib.utils.layers import get_activation


@nip
class BasicGraphExtractor(BaseFeaturesExtractor):
    NAME = "basic_graph_extractor"

    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 256,
                 embedding_dim: int = 512,
                 n_ped_attention_heads: int = 8,
                 ped_only_attention: bool = False):
        super(BasicGraphExtractor, self).__init__(observation_space, features_dim)
        self._ped_only_attention = ped_only_attention

        n_max_peds, seq_len, state_dim = observation_space["peds_traj"].shape
        robot_state_dim = observation_space["robot_state"].shape[0]

        self._peds_traj_embedding = nn.Sequential(
            nn.Linear(seq_len * state_dim, embedding_dim),
            nn.Tanh()
        )
        self._ped_query_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._ped_key_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._ped_value_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._peds_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                     num_heads=n_ped_attention_heads,
                                                     batch_first=True)

        self._robot_embedding = nn.Sequential(
            nn.Linear(robot_state_dim, embedding_dim),
            nn.Tanh()
        )
        self._robot_query_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._robot_key_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._robot_value_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._robot_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                      num_heads=1,
                                                      batch_first=True)

        self._final_feature_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        peds_traj = observations["peds_traj"]  # (n_envs, n_max_peds, sequence_len, 2)
        peds_visibility = observations["peds_visibility"]  # (n_envs, n_max_peds)
        robot_state = observations["robot_state"]  # (n_envs, 4)

        # TODO: Substitute with learnable stub for cases where no pedestrians visible
        for i in range(peds_visibility.shape[0]):
            if peds_visibility[i].sum() == 0.:
                peds_visibility[i] = 1.
        key_padding_mask = torch.logical_not(peds_visibility > 0)

        feature = self._process_peds_traj(peds_traj, key_padding_mask)  # (n_envs, n_max_peds, emb_dim)
        if not self._ped_only_attention:
            feature = self._process_robot_state(robot_state, feature,
                                                key_padding_mask)  # (n_envs, emb_dim)
        else:
            feature = torch.mean(feature, dim=1)  # (n_envs, emb_dim)

        feature = self._final_feature_embedding.forward(feature)

        return feature

    def _process_peds_traj(self, peds_traj: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        n_envs, n_max_peds, seq_len, state_dim = peds_traj.shape
        peds_traj = torch.reshape(peds_traj, (n_envs, n_max_peds, seq_len * state_dim))

        peds_traj = self._peds_traj_embedding.forward(peds_traj)

        peds_traj_query = self._ped_query_embedding.forward(peds_traj)
        peds_traj_key = self._ped_key_embedding.forward(peds_traj)
        peds_traj_value = self._ped_value_embedding.forward(peds_traj)

        peds_traj, _ = self._peds_attention.forward(peds_traj_query, peds_traj_key, peds_traj_value,
                                                    key_padding_mask=padding_mask,
                                                    need_weights=False)

        return peds_traj

    def _process_robot_state(self, robot_state: torch.Tensor, peds: torch.Tensor, padding_mask: torch.Tensor) -> \
            torch.Tensor:
        robot_state = self._robot_embedding.forward(robot_state).unsqueeze(1)

        robot_query = self._robot_query_embedding.forward(robot_state)
        peds_key = self._robot_key_embedding.forward(peds)
        peds_value = self._robot_value_embedding.forward(peds)

        robot_state, _ = self._robot_attention.forward(robot_query, peds_key, peds_value,
                                                       key_padding_mask=padding_mask,
                                                       need_weights=False)
        robot_state = robot_state.squeeze(1)
        return robot_state


@nip
class SimpleGraphExtractor(BaseFeaturesExtractor):
    NAME = "simple_graph_extractor"

    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 256,
                 embedding_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_max_peds, seq_len, state_dim = observation_space["peds_traj"].shape
        robot_state_dim = observation_space["robot_state"].shape[0]

        self._peds_traj_embedding = nn.Sequential(
            nn.Linear(seq_len * state_dim, embedding_dim),
            nn.Tanh()
        )

        self._robot_embedding = nn.Sequential(
            nn.Linear(robot_state_dim, embedding_dim),
            nn.Tanh()
        )

        self._final_feature_embedding = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        peds_traj = observations["peds_traj"]  # (n_envs, n_max_peds, sequence_len, 2)
        peds_visibility = observations["peds_visibility"]  # (n_envs, n_max_peds)
        robot_state = observations["robot_state"]  # (n_envs, 4)
        n_envs, n_max_peds, seq_len, state_dim = peds_traj.shape

        peds_emb = torch.reshape(peds_traj, (n_envs, n_max_peds, seq_len * state_dim))
        peds_emb = self._peds_traj_embedding.forward(peds_emb)
        peds_emb = torch.mean(peds_emb, dim=1)

        robot_emb = self._robot_embedding.forward(robot_state)

        joint_feature = torch.cat((peds_emb, robot_emb), dim=1)

        final_feature = self._final_feature_embedding.forward(joint_feature)

        return final_feature


@nip
class PoolingGraphExtractor(BaseFeaturesExtractor):
    NAME = "basic_graph_extractor"

    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 256,
                 embedding_dim: int = 512,
                 n_ped_attention_heads: int = 8,
                 activation: str = "tanh"):
        super(PoolingGraphExtractor, self).__init__(observation_space, features_dim)
        activation = get_activation(activation)

        n_max_peds, seq_len, state_dim = observation_space["peds_traj"].shape
        robot_state_dim = observation_space["robot_state"].shape[0]

        self._peds_traj_embedding = nn.Sequential(
            nn.Linear(seq_len * state_dim, embedding_dim),
            activation()
        )
        self._ped_query_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._ped_key_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._ped_value_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._peds_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                     num_heads=n_ped_attention_heads,
                                                     batch_first=True)

        self._robot_embedding = nn.Sequential(
            nn.Linear(robot_state_dim, embedding_dim),
            activation()
        )

        self._final_feature_embedding = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            activation(),
            nn.Linear(embedding_dim, features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        peds_traj = observations["peds_traj"]  # (n_envs, n_max_peds, sequence_len, 2)
        peds_visibility = observations["peds_visibility"]  # (n_envs, n_max_peds)
        robot_state = observations["robot_state"]  # (n_envs, 4)

        # TODO: Substitute with learnable stub for cases where no pedestrians visible
        for i in range(peds_visibility.shape[0]):
            if peds_visibility[i].sum() == 0.:
                peds_visibility[i] = 1.
        key_padding_mask = torch.logical_not(peds_visibility > 0)

        peds_emb = self._process_peds_traj(peds_traj, key_padding_mask)  # (n_envs, n_max_peds, emb_dim)
        peds_emb = torch.mean(peds_emb, dim=1)  # (n_envs, emb_dim)

        robot_emb = self._robot_embedding.forward(robot_state)

        joint_feature = torch.cat((peds_emb, robot_emb), dim=1)

        final_feature = self._final_feature_embedding.forward(joint_feature)

        return final_feature

    def _process_peds_traj(self, peds_traj: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        n_envs, n_max_peds, seq_len, state_dim = peds_traj.shape
        peds_traj = torch.reshape(peds_traj, (n_envs, n_max_peds, seq_len * state_dim))

        peds_traj = self._peds_traj_embedding.forward(peds_traj)

        peds_traj_query = self._ped_query_embedding.forward(peds_traj)
        peds_traj_key = self._ped_key_embedding.forward(peds_traj)
        peds_traj_value = self._ped_value_embedding.forward(peds_traj)

        peds_traj, _ = self._peds_attention.forward(peds_traj_query, peds_traj_key, peds_traj_value,
                                                    key_padding_mask=padding_mask,
                                                    need_weights=False)

        return peds_traj


@nip
class DoubleAttentionExtractor(BaseFeaturesExtractor):
    NAME = "double_graph_extractor"

    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 256,
                 embedding_dim: int = 512,
                 n_ped_attention_heads: int = 8,
                 n_robot_peds_attention_heads: int = 1,
                 activation: str = "tanh",
                 embedding_activation: bool = True):
        super(DoubleAttentionExtractor, self).__init__(observation_space, features_dim)
        activation = get_activation(activation)
        embedding_activation = activation if embedding_activation else nn.Identity

        n_max_peds, seq_len, state_dim = observation_space["peds_traj"].shape
        robot_state_dim = observation_space["robot_state"].shape[0]

        self._peds_traj_embedding = nn.Sequential(
            nn.Linear(seq_len * state_dim, embedding_dim),
            embedding_activation()
        )
        self._peds_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                     num_heads=n_ped_attention_heads,
                                                     batch_first=True)

        self._robot_embedding = nn.Sequential(
            nn.Linear(robot_state_dim, embedding_dim),
            embedding_activation()
        )
        self._robot_peds_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                           num_heads=n_robot_peds_attention_heads,
                                                           batch_first=True)

        self._final_feature_embedding = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            activation(),
            nn.Linear(embedding_dim, features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        peds_traj_original = observations["peds_traj"]  # (n_envs, n_max_peds, sequence_len, 2)
        peds_visibility = observations["peds_visibility"]  # (n_envs, n_max_peds)
        robot_state = observations["robot_state"]  # (n_envs, 4)

        peds_traj = torch.clone(peds_traj_original)

        # TODO: Substitute with learnable stub for cases where no pedestrians visible
        for i in range(peds_visibility.shape[0]):
            if peds_visibility[i].sum() == 0.:
                peds_visibility[i] = 1.
                peds_traj[i, :, :, 0] = -10.
                peds_traj[i, :, :, 1] = -10.
        key_padding_mask = torch.logical_not(peds_visibility > 0)

        peds_emb = self._process_peds_traj(peds_traj, key_padding_mask)  # (n_envs, n_max_peds, emb_dim)

        robot_emb = self._robot_embedding.forward(robot_state)
        robot_peds_emb = self._robot_peds_emb(robot_emb.unsqueeze(1), peds_emb, key_padding_mask)[:, 0, :]

        joint_feature = torch.cat((robot_emb, robot_peds_emb), dim=1)
        final_feature = self._final_feature_embedding.forward(joint_feature)

        return final_feature

    def _process_peds_traj(self, peds_traj: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        n_envs, n_max_peds, seq_len, state_dim = peds_traj.shape
        peds_traj = torch.reshape(peds_traj, (n_envs, n_max_peds, seq_len * state_dim))
        peds_traj = self._peds_traj_embedding.forward(peds_traj)
        peds_traj, _ = self._peds_attention.forward(peds_traj, peds_traj, peds_traj,
                                                    key_padding_mask=padding_mask,
                                                    need_weights=False)
        return peds_traj

    def _robot_peds_emb(self, robot_emb: torch.Tensor, peds_emb: torch.Tensor,
                        key_padding_mask: torch.Tensor) -> torch.Tensor:
        robot_peds_emb, _ = self._robot_peds_attention.forward(query=robot_emb,
                                                               key=peds_emb,
                                                               value=peds_emb,
                                                               key_padding_mask=key_padding_mask,
                                                               need_weights=False)
        return robot_peds_emb


@nip
class WindowStackExtractor(BaseFeaturesExtractor):
    NAME = "window_stack_extractor"

    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 256,
                 activation: str = "tanh"):
        super(WindowStackExtractor, self).__init__(observation_space, features_dim)
        activation = get_activation(activation)

        n_history, n_max_peds, ped_state_dim = observation_space["peds_traj"].shape
        robot_state_dim = observation_space["robot_state"].shape[0]

        self._no_peds_stub = nn.Parameter(torch.zeros(ped_state_dim))

        self._peds_rnn = nn.GRU(input_size=n_max_peds * ped_state_dim,
                                hidden_size=features_dim,
                                batch_first=True)

        self._joint_mlp = nn.Sequential(
            nn.Linear(features_dim + robot_state_dim, features_dim),
            activation(),
            nn.Linear(features_dim, features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        obs_peds = observations["peds_traj"]  # (n_envs, n_stack, n_max_peds, 5)
        peds_visibility = observations["peds_visibility"]  # (n_envs, n_stack, n_max_peds)
        robot_state = observations["robot_state"]  # (n_envs, robot_emb_dim)

        obs_peds = self._transform_pedestrians(obs_peds, peds_visibility)

        _, feature = self._peds_rnn(obs_peds)
        feature = feature[0]
        feature = torch.cat((robot_state, feature), dim=1)
        feature = self._joint_mlp(feature)

        return feature

    def _transform_pedestrians(self, obs_peds: torch.Tensor, peds_visibility: torch.Tensor) -> torch.Tensor:
        n_envs, n_stack, n_max_peds, state_dim = obs_peds.shape
        obs_filtered = torch.ones_like(obs_peds, requires_grad=True) * obs_peds
        peds_visibility = peds_visibility.unsqueeze(-1)
        obs_filtered = obs_filtered * peds_visibility + torch.logical_not(peds_visibility) * self._no_peds_stub
        obs_filtered = obs_filtered.reshape((n_envs, n_stack, n_max_peds * state_dim))
        return obs_filtered


@nip
class DoubleNoPredictionAttentionExtractor(BaseFeaturesExtractor):
    NAME = "double_no_pred_graph_extractor"

    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 256,
                 embedding_dim: int = 128,
                 n_ped_attention_heads: int = 8,
                 n_robot_peds_attention_heads: int = 1,
                 activation: str = "tanh",
                 embedding_activation: bool = False):
        super(DoubleNoPredictionAttentionExtractor, self).__init__(observation_space, features_dim)
        activation = get_activation(activation)
        embedding_activation = activation if embedding_activation else nn.Identity

        n_max_peds, peds_state_dim = observation_space["peds"].shape
        robot_state_dim = observation_space["robot"].shape[0]

        self._peds_embedding = nn.Sequential(
            nn.Linear(peds_state_dim, embedding_dim),
            embedding_activation()
        )
        self._peds_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                     num_heads=n_ped_attention_heads,
                                                     batch_first=True)

        self._robot_embedding = nn.Sequential(
            nn.Linear(robot_state_dim, embedding_dim),
            embedding_activation()
        )
        self._robot_peds_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                           num_heads=n_robot_peds_attention_heads,
                                                           batch_first=True)

        self._final_feature_embedding = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            activation(),
            nn.Linear(embedding_dim, features_dim)
        )

    def forward(self, observations) -> torch.Tensor:
        peds = observations["peds"]  # (n_envs, n_max_peds, sequence_len, 2)
        peds_visibility = observations["peds_visibility"]  # (n_envs, n_max_peds)
        robot = observations["robot"]  # (n_envs, 4)

        # TODO: Substitute with learnable stub for cases where no pedestrians visible
        for i in range(peds_visibility.shape[0]):
            if peds_visibility[i].sum() == 0.:
                peds_visibility[i] = 1.
        key_padding_mask = torch.logical_not(peds_visibility > 0)

        peds = self._peds_embedding(peds)
        peds, _ = self._peds_attention(peds, peds, peds,
                                       key_padding_mask=key_padding_mask,
                                       need_weights=False)

        robot = self._robot_embedding.forward(robot)
        robot_peds, _ = self._robot_peds_attention(query=robot.unsqueeze(1),
                                                   key=peds,
                                                   value=peds,
                                                   key_padding_mask=key_padding_mask,
                                                   need_weights=False)
        robot_peds = robot_peds[:, 0, :]

        joint_feature = torch.cat((robot, robot_peds), dim=1)
        final_feature = self._final_feature_embedding.forward(joint_feature)

        return final_feature
