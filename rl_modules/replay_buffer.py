import threading
import numpy as np
from sklearn.cluster import KMeans, MeanShift

"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T ####10^6 // 50 == 20000
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }

        # faher
        self.size_fgs = 150
        self.idx_fg = 0
        self.fgs = []
        self.cluster = KMeans(n_clusters=8)

        # thread lock
        self.lock = threading.Lock()
    
    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size
            # cher
            self.store_failed_goal(mb_ag, mb_g)

    def store_failed_goal(self, ag, g):
        ag = ag[:,-1,:]
        g = g[:,-1,:]
        failed = np.linalg.norm(ag - g, axis=-1) > 0.05
        for i in range(len(failed)):
            if failed[i]:
                if len(self.fgs) < self.size_fgs:
                    self.fgs.append([])
                self.fgs[self.idx_fg] = g[i,:].tolist()
                self.idx_fg = (self.idx_fg + 1) % self.size_fgs

                if self.idx_fg == self.size_fgs - 1:
                    self.cluster.fit(np.array(self.fgs))

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]

        # faher
        clustered_idx = []
        if len(self.fgs) == self.size_fgs:
            idx_c = self.cluster.predict(temp_buffers['ag'][:,-1,:])
            for i in range(idx_c.max()+1):
                clustered_idx.append(np.where(idx_c == i)[0].tolist())

        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size, clustered_idx)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx
