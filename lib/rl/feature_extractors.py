import torch
import torch.nn as nn
import gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BasicGraphExtractor(BaseFeaturesExtractor):
    NAME = "basic_graph_extractor"

    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 256,
                 embedding_dim: int = 512,
                 n_ped_attention_heads: int = 8):
        super(BasicGraphExtractor, self).__init__(observation_space, features_dim)

        n_max_peds, seq_len, state_dim = observation_space["peds_traj"].shape
        robot_state_dim = observation_space["robot_state"].shape[0]

        self._peds_traj_embedding = nn.Sequential(
            nn.Linear(seq_len * state_dim, embedding_dim),
            nn.ReLU()
        )
        self._ped_query_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._ped_key_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._ped_value_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._peds_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                     num_heads=n_ped_attention_heads,
                                                     batch_first=True)

        self._robot_embedding = nn.Sequential(
            nn.Linear(robot_state_dim, embedding_dim),
            nn.ReLU()
        )
        self._robot_query_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._robot_key_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._robot_value_embedding = nn.Linear(embedding_dim, embedding_dim)
        self._robot_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                      num_heads=1,
                                                      batch_first=True)

        self._final_feature_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
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

        peds_traj = self._process_peds_traj(peds_traj, key_padding_mask)  # (n_envs, n_max_peds, emb_dim)
        robot_state = self._process_robot_state(robot_state, peds_traj,
                                                key_padding_mask)  # (n_envs, emb_dim)

        feature = self._final_feature_embedding.forward(robot_state)

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
