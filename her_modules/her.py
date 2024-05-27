import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k)) ### if k == 4 out is 0.8 
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions, clustered_idx=[]):
        T = episode_batch['actions'].shape[1] ### 50
        rollout_batch_size = episode_batch['actions'].shape[0] ### all size of buffer
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        # cher
        if len(clustered_idx) == 0:
            episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        else:
            episode_idxs = self.clustering_her_random(rollout_batch_size, batch_size, clustered_idx)

        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace goal with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions

    def clustering_her_random(self, rollout_batch_size, batch_size, clustered_idx):
        num_idx = len(clustered_idx)
        batch_size_cluster = int(batch_size/num_idx)
        batch_size_normal = batch_size - num_idx*batch_size_cluster
        episode_idxs = []
        for i in range(num_idx):
            num_in_cluster = len(clustered_idx[i])
            if num_in_cluster >= int(batch_size_cluster/4):
                idx = np.random.randint(0, num_in_cluster, batch_size_cluster)
                episode_idxs += np.array(clustered_idx[i])[idx].tolist()
            else:
                if num_in_cluster != 0:
                    idx = np.random.randint(0, num_in_cluster, num_in_cluster*4)
                    episode_idxs += np.array(clustered_idx[i])[idx].tolist()
                episode_idxs += np.random.randint(0, rollout_batch_size, batch_size_cluster-num_in_cluster*4).tolist()

        if batch_size_normal != 0:
            episode_idxs += np.random.randint(0, rollout_batch_size, batch_size_normal).tolist()

        episode_idxs = np.array(episode_idxs)
        return episode_idxs

