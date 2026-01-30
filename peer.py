from abc import ABC
from typing import Type
import itertools as it

import numpy as np
import torch

from suggestionbuffer import SuggestionBuffer
from utils import make_env

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm


class PeerGroup:
    """ A group of peers who train together. """
    def __init__(self, peers, use_agent_values=False, init_agent_values=200.,
                 lr=0.95, switch_ratio=0, use_advantage=False,
                 max_peer_epochs=1_000_000_000):
        """
        :param peers: An iterable of peer agents
        :param lr: The learning rate for trust and agent values
        :param switch_ratio: switch_ratio == 0 means no switching
        :param use_advantage: use advantage instead of value for AV updates
        """
        self.peers = peers
        self.lr = lr
        self.switch_ratio = switch_ratio
        self.active_peer = None  # index of currently learning peer
        self.solo_epoch = False
        self.use_advantage = use_advantage
        self.max_peer_epochs = max_peer_epochs

        if use_agent_values:
            self.agent_values = np.full(len(peers), init_agent_values,
                                        dtype=np.float32)
            key = "agent_values"

        for peer in peers:
            peer.n_peers = len(peers)
            peer.group = self

            # setup agent values
            if use_agent_values:
                peer.peer_values[key] = self.agent_values  # noqa (Eq. 6)
                peer.peer_value_functions[key] = self._update_agent_values

    def _update_agent_values(self, batch_size=10):
        """ Updates the agent values with samples from the peers' buffers"""
        targets = np.zeros_like(self.peers, dtype=np.float32)
        counts = np.zeros_like(self.peers, dtype=np.float32)

        for peer in self.peers:
            bs = batch_size // len(self.peers)
            # reward, action, peer, new_obs, old_obs
            if peer.buffer is not None:
                batch = peer.buffer.sample(bs)
                if batch is None:  # buffer not sufficiently full
                    return

                obs = np.array([b[3] for b in batch]).reshape(bs, -1)
                v = peer.value(obs)

                if self.use_advantage:
                    # previous observations
                    prev_obs = np.array([b[4] for b in batch]).reshape(bs, -1)
                    prev_v = peer.value(prev_obs)
                else:
                    prev_v = np.zeros_like(v)  # no advantage (see Eq. 5)

                for i in range(len(batch)):  # Eq. 8
                    # Skip if peer index is None
                    if batch[i][2] is None:
                        continue
                    
                    # Ensure scalar values by extracting item from arrays
                    v_value = v[i].item() if hasattr(v[i], 'item') else float(v[i])
                    prev_v_value = prev_v[i].item() if hasattr(prev_v[i], 'item') else float(prev_v[i])
                    target = (batch[i][0] + peer.gamma * v_value) - prev_v_value
                    
                    # Ensure peer index is a scalar integer
                    peer_idx = int(batch[i][2]) if not isinstance(batch[i][2], int) else batch[i][2]
                    counts[peer_idx] += 1
                    targets[peer_idx] += target

        # ensure counts are >= 1, don't change these values
        targets[counts == 0] = self.agent_values[counts == 0]
        counts[counts == 0] = 1

        targets /= counts
        self.agent_values += self.lr * (targets - self.agent_values)  # Eq. 7

    def learn(self, n_epochs, max_epoch_len, callbacks, **kwargs):
        """ The outer peer learning routine. """
        assert len(callbacks) == len(self.peers)
        # more solo epochs
        boost_single = 0 < self.switch_ratio < 1
        if boost_single:
            self.switch_ratio = 1 / self.switch_ratio

        self.solo_epoch = False
        peer_epochs = 0
        for i in range(n_epochs):
            # don't do peer learning forever
            if peer_epochs < self.max_peer_epochs:
                # ratio of 0 never performs a solo episode
                if (i % (1 + self.switch_ratio) == 1) ^ boost_single:
                    self.solo_epoch = True
                else:
                    peer_epochs += 1
            else:  # budget spent
                self.solo_epoch = True

            for p, peer, callback in zip(it.count(), self.peers, callbacks):
                self.active_peer = p
                peer.learn(self.solo_epoch, total_timesteps=max_epoch_len,
                           callback=callback, tb_log_name=f"Peer{p}",
                           reset_num_timesteps=False,
                           log_interval=None, **kwargs)
                # update epoch for temperature decay
                peer.epoch += 1

        self.active_peer = None

    def __len__(self):
        return len(self.peers)


def make_peer_class(cls: Type[OffPolicyAlgorithm]):
    """ Creates a mixin with the corresponding algorithm class.
    :param cls: The learning algorithm (needs to have a callable critic).
    :return: The mixed in peer agent class.
    """

    class Peer(cls, ABC):
        """ Abstract Peer class
        needs to be mixed with a suitable algorithm. """
        def __init__(self, temperature, temp_decay, algo_args, env,
                     use_trust=False, use_critic=False, init_trust_values=200,
                     buffer_size=1000, follow_steps=10, seed=None,
                     use_trust_buffer=True, solo_training=False,
                     peers_sample_with_noise=False,
                     sample_random_actions=False, sample_from_suggestions=True,
                     epsilon=0.0, env_args=None, only_follow_peers=False):
            if env_args is None:
                env_args = {}
            super(Peer, self).__init__(**algo_args,
                                       env=make_env(env, **env_args),
                                       seed=seed)
            # create noise matrix on the correct device (only for algorithms with gSDE)
            if hasattr(self, "actor") and hasattr(self.actor, "reset_noise"):
                try:
                    self.actor.reset_noise(self.env.num_envs)
                except (AssertionError, AttributeError):
                    # reset_noise only works with gSDE, skip if not available
                    pass

            self.solo_training = solo_training
            self.init_values = dict()
            # store all peer values, e.g., trust and agent values in a dict
            self.peer_values = dict()
            # store corresponding functions as well
            self.peer_value_functions = dict()

            self.buffer = SuggestionBuffer(buffer_size)
            self.followed_peer = None
            self.__n_peers = None
            self.group = None
            self.epoch = 0

            if sample_random_actions:
                epsilon = 1.0

            if not solo_training:
                # all peers suggest without noise
                self.peers_sample_with_noise = peers_sample_with_noise
                # actions are sampled instead of taken greedily
                self.sample_actions = sample_from_suggestions
                self.epsilon = epsilon
                self.use_critic = use_critic

                if use_trust:
                    self.trust_values = np.array([])
                    self.init_values["trust"] = init_trust_values
                    self.peer_value_functions["trust"] = self._update_trust

                    self.use_buffer_for_trust = use_trust_buffer

                # sampling parameters
                self.temperature = temperature
                self.temp_decay = temp_decay

                self.follow_steps = follow_steps
                self.steps_followed = 0

                self.only_follow_peers = only_follow_peers

        @property
        def n_peers(self):
            return self.__n_peers

        @n_peers.setter
        def n_peers(self, n_peers):
            self.__n_peers = n_peers

            # Also reset the trust values
            if "trust" in self.init_values.keys():
                self.trust_values = np.full(self.__n_peers,
                                            self.init_values["trust"],
                                            dtype=np.float32)
                self.peer_values["trust"] = self.trust_values

        def critique(self, observations, actions) -> np.array:
            """ Evaluates the actions with the critic. """
            with torch.no_grad():
                a = torch.as_tensor(actions, device=self.device)
                o = torch.as_tensor(observations, device=self.device)

                # Handle different algorithm types
                if hasattr(self, 'critic'):
                    # SAC: has critic network
                    q_values = torch.cat(self.critic(o, a), dim=1)  # noqa
                    q_values, _ = torch.min(q_values, dim=1, keepdim=True)
                elif hasattr(self, 'q_net'):
                    # DQN: has q_net
                    q_values = self.q_net(o)
                    # Select Q-values for the given actions
                    if len(a.shape) == 1:
                        a = a.unsqueeze(-1)
                    q_values = q_values.gather(1, a.long())
                else:
                    raise AttributeError(f"Algorithm {type(self).__name__} has neither 'critic' nor 'q_net'")
                
                return q_values.cpu().numpy()

        def get_action(self, obs, deterministic=False):
            """ The core function of peer learning acquires the suggested
            actions of the peers and chooses one based on the settings. """
            # follow peer for defined number of steps
            followed_steps = self.steps_followed
            self.steps_followed += 1
            self.steps_followed %= self.follow_steps
            if 0 < followed_steps:
                peer = self.group.peers[self.followed_peer]
                det = (peer != self and not self.peers_sample_with_noise) or \
                    deterministic
                action, _ = peer.policy.predict(obs, deterministic=det)
                return action, None

            # get actions
            actions = []
            for peer in self.group.peers:
                # self always uses exploration, the suggestions of the other
                # peers only do if the critic method isn't used.
                det = (peer != self and not self.peers_sample_with_noise) or \
                      deterministic
                action, _ = peer.policy.predict(obs, deterministic=det)
                actions.append(action)
            actions = np.asarray(actions).squeeze(1)

            # critic (Eq. 3)
            if self.use_critic:
                observations = np.tile(obs, (self.n_peers, 1))
                q_values = self.critique(observations, actions).reshape(-1)
                self.peer_values['critic'] = q_values  # part of Eq. 9

            # calculate peer values, e.g., trust and agent values
            values = np.zeros(self.n_peers)
            for key in self.peer_values.keys():
                # part of Eq. 9 incl. Footnote 7
                values += self.__normalize(self.peer_values[key])

            if self.sample_actions:
                # sample action from probability distribution (Eq. 2)
                temp = self.temperature * np.exp(-self.temp_decay * self.epoch)
                p = np.exp(values / temp)
                p /= np.sum(p)
                self.followed_peer = np.random.choice(self.n_peers, p=p)
            elif self.only_follow_peers:
                p = np.full(self.n_peers, 1 / (self.n_peers - 1))
                p[self.group.peers.index(self)] = 0
                self.followed_peer = np.random.choice(self.n_peers, p=p)
            else:
                # act (epsilon) greedily
                if np.random.random(1) >= self.epsilon:
                    self.followed_peer = np.argmax(values)
                else:
                    self.followed_peer = np.random.choice(self.n_peers)

            action = actions[self.followed_peer]
            
            # Ensure action maintains array shape for vectorized environments
            # If action is scalar (after squeeze), reshape back to (1,)
            if np.isscalar(action) or (isinstance(action, np.ndarray) and action.ndim == 0):
                action = np.array([action])
            elif isinstance(action, np.ndarray) and action.ndim == 1 and len(action) > 1:
                # For continuous actions, reshape to (1, action_dim)
                action = action.reshape(1, -1)
            
            return action, None

        @staticmethod
        def __normalize(values):
            """ Normalize the values based on their absolute maximum. """
            return values / np.max(np.abs(values))

        def value(self, observations) -> np.ndarray:
            """ Calculates the value of the observations. """
            actions, _ = self.policy.predict(observations, False)
            return self.critique(observations, actions)

        def _update_trust(self, batch_size=10):
            """ Updates the trust values with samples from the buffer.
                (Eq. 5 and 8)
            """
            if self.use_buffer_for_trust:
                batch = self.buffer.sample(batch_size)
            else:
                batch = self.buffer.latest()
                batch_size = 1
            if batch is None:  # buffer not sufficiently full
                return

            # next observations
            obs = np.array([b[3] for b in batch]).reshape(batch_size, -1)
            v = self.value(obs)

            if self.group.use_advantage:
                # previous observations
                prev_obs = np.array([b[4] for b in batch]).reshape(batch_size,
                                                                   -1)
                prev_v = self.value(prev_obs)
            else:
                prev_v = np.zeros_like(v)  # no comparison to own act (Eq. 5)

            targets = np.zeros(self.n_peers)
            counts = np.zeros(self.n_peers)
            for i in range(batch_size):
                # Skip if peer index is None
                if batch[i][2] is None:
                    continue
                    
                # Ensure scalar values by extracting item from arrays
                v_value = v[i].item() if hasattr(v[i], 'item') else float(v[i])
                prev_v_value = prev_v[i].item() if hasattr(prev_v[i], 'item') else float(prev_v[i])
                target = (batch[i][0] + self.gamma * v_value) - prev_v_value  # Eq. 8
                
                # Ensure peer index is a scalar integer
                peer_idx = int(batch[i][2]) if not isinstance(batch[i][2], int) else batch[i][2]
                counts[peer_idx] += 1
                targets[peer_idx] += target

            # ensure counts are >= 1, don't change these values
            targets[counts == 0] = self.trust_values[counts == 0]
            counts[counts == 0] = 1

            targets /= counts
            # Eq. 4
            self.trust_values += self.group.lr * (targets - self.trust_values)
        
        def get_trust_data(self) -> dict:
            """
            Get trust data in unified format for analysis
            
            Returns:
                Dictionary with trust values compatible with TrustAnalyzer
            """
            # Check if trust is enabled for this peer
            if hasattr(self, 'trust_values') and self.trust_values is not None:
                # Convert numpy array to dict
                trust_dict = {i: float(self.trust_values[i]) for i in range(self.n_peers)}
            else:
                # No trust values - return uniform trust
                trust_dict = {i: 1.0 for i in range(self.n_peers)}
            
            return {
                'final_trust': trust_dict,
                'attention_trust': None,  # Baseline has no attention component
                'traditional_trust': trust_dict,
                'confidence': 0.0,
                'omega_t': 0.0
            }

        def _on_step(self):
            """ Adds updates of the peer values, e.g., trust or agent
            values. """
            super(Peer, self)._on_step()  # noqa

            if not self.group.solo_epoch:
                # update values, e.g., trust and agent values after ever step
                for key in self.peer_value_functions.keys():
                    self.peer_value_functions[key]()

        def _store_transition(self, replay_buffer, buffer_action, new_obs,
                              reward, dones, infos):
            """ Adds suggestion buffer handling. """

            # get previous observations
            old_obs = self._last_obs

            super(Peer, self)._store_transition(replay_buffer,  # noqa
                                                buffer_action, new_obs,
                                                reward, dones, infos)

            if not self.group.solo_epoch:
                # store transition in suggestion buffer as well (only if followed_peer is valid)
                if self.followed_peer is not None:
                    # Ensure reward is scalar
                    reward_scalar = float(reward[0]) if hasattr(reward, '__len__') else float(reward)
                    self.buffer.add(reward_scalar, buffer_action, self.followed_peer,
                                    new_obs, old_obs)

        def _predict_train(self, observation, state=None,
                           episode_start=None, deterministic=False):
            """ The action selection during training involves the peers. """
            if deterministic:
                return self.policy.predict(observation, state=state,
                                           episode_start=episode_start,
                                           deterministic=deterministic)
            else:
                return self.get_action(observation)

        def learn(self, solo_episode=False, **kwargs):
            """ Adds action selection with help of peers. """
            predict = self.predict  # safe for later

            # use peer suggestions only when wanted
            if not (self.solo_training or solo_episode):
                self.predict = self._predict_train
            else:
                self.followed_peer = self.group.peers.index(self)

            # Filter out parameters not supported by parent's learn method
            # eval_log_path is not a valid parameter in newer stable-baselines3
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['eval_log_path']}

            result = super(Peer, self).learn(**filtered_kwargs)

            self.predict = predict  # noqa
            return result

        def _excluded_save_params(self):
            """ Excludes attributes that are functions. Otherwise, the save
            method fails. """
            ex_list = super(Peer, self)._excluded_save_params()
            ex_list.extend(["peer_value_functions", "peer_values",
                            "group", "predict"])
            return ex_list

    return Peer
