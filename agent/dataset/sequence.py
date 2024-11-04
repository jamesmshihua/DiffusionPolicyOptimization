import numpy as np
import tensorflow as tf
import logging
import pickle
import random
from tqdm import tqdm
from collections import namedtuple

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "actions conditions")
Transition = namedtuple("Transition", "actions conditions rewards dones")
TransitionWithReturn = namedtuple(
    "Transition", "actions conditions rewards dones reward_to_gos"
)

class StitchedSequenceDataset(tf.data.Dataset):
    """
    Load stitched trajectories of states/actions/images, and 1-D array of traj_lengths, from npz or pkl file.

    Use the first max_n_episodes episodes (instead of random sampling)

    Example:
        states: [----------traj 1----------][---------traj 2----------] ... [---------traj N----------]
        Episode IDs (determined based on traj_lengths):  [----------   1  ----------][----------   2  ---------] ... [----------   N  ---------]

    Each sample is a namedtuple of (1) chunked actions and (2) a list (obs timesteps) of dictionary with keys states and images.

    """

    def __init__(
        self,
        dataset_path,
        horizon_steps=64,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
        device="GPU",
    ):
        assert (
            img_cond_steps <= cond_steps
        ), "consider using more cond_steps than img_cond_steps"
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.use_img = use_img
        self.max_n_episodes = max_n_episodes
        self.dataset_path = dataset_path

        # Load dataset to device specified
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)  # only np arrays
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]  # 1-D array
        total_num_steps = np.sum(traj_lengths)

        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths, horizon_steps)

        # Extract states and actions up to max_n_episodes
        self.states = (
            tf.convert_to_tensor(dataset["states"][:total_num_steps], dtype=tf.float32)
        )  # (total_num_steps, obs_dim)
        self.actions = (
            tf.convert_to_tensor(dataset["actions"][:total_num_steps], dtype=tf.float32)
        )  # (total_num_steps, action_dim)
        log.info(f"Loaded dataset from {dataset_path}")
        log.info(f"Number of episodes: {min(max_n_episodes, len(traj_lengths))}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")
        if self.use_img:
            self.images = tf.convert_to_tensor(dataset["images"][:total_num_steps], dtype=tf.float32)  # (total_num_steps, C, H, W)
            log.info(f"Images shape/type: {self.images.shape, self.images.dtype}")

    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start):(start + 1)]
        actions = self.actions[start:end]
        states = tf.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        conditions = {"state": states}
        if self.use_img:
            images = self.images[(start - num_before_start):end]
            images = tf.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images
        batch = Batch(actions=actions, conditions=conditions)
        return batch

    def make_indices(self, traj_lengths, horizon_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint, also save the number of steps before it within the same trajectory
        """
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        return indices

    def set_train_val_split(self, train_split):
        """
        Not doing validation right now
        """
        num_train = int(len(self.indices) * train_split)
        train_indices = random.sample(self.indices, num_train)
        val_indices = [i for i in range(len(self.indices)) if i not in train_indices]
        self.indices = train_indices
        return val_indices

    def __len__(self):
        return len(self.indices)
    
    @property
    def element_spec(self):
        # Define the element specification based on dataset content
        spec = {
            "actions": tf.TensorSpec(shape=(self.horizon_steps, self.actions.shape[-1]), dtype=tf.float32),
            "conditions": {
                "state": tf.TensorSpec(shape=(self.cond_steps, self.states.shape[-1]), dtype=tf.float32),
            }
        }
        if self.use_img:
            spec["conditions"]["rgb"] = tf.TensorSpec(shape=(self.img_cond_steps, *self.images.shape[1:]), dtype=tf.float32)
        return spec
    
    def _inputs(self):
        return []



class StitchedSequenceQLearningDataset(StitchedSequenceDataset):
    def __init__(
        self,
        dataset_path,
        max_n_episodes=10000,
        discount_factor=1.0,
        device="GPU",
        get_mc_return=False,
        **kwargs,
    ):
        super().__init__()
        
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]
        total_num_steps = np.sum(traj_lengths)

        # Discount factor
        self.discount_factor = discount_factor

        # Rewards and dones (terminals)
        self.rewards = tf.convert_to_tensor(dataset["rewards"][:total_num_steps], dtype=tf.float32)
        log.info(f"Rewards shape/type: {self.rewards.shape}, {self.rewards.dtype}")
        self.dones = tf.convert_to_tensor(dataset["terminals"][:total_num_steps], dtype=tf.float32)
        log.info(f"Dones shape/type: {self.dones.shape}, {self.dones.dtype}")

        self.indices = self.make_indices(traj_lengths)
        self.get_mc_return = get_mc_return

        # Compute discounted reward-to-go for each trajectory
        if get_mc_return:
            self.reward_to_go = tf.zeros_like(self.rewards)
            cumulative_traj_length = np.cumsum(traj_lengths)
            prev_traj_length = 0
            for i, traj_length in tqdm(
                enumerate(cumulative_traj_length), desc="Computing reward-to-go"
            ):
                traj_rewards = self.rewards[prev_traj_length:traj_length]
                returns = tf.zeros_like(traj_rewards)
                prev_return = 0
                for t in range(len(traj_rewards)):
                    returns = (
                        traj_rewards[-t - 1] + self.discount_factor * prev_return
                    )
                    prev_return = returns
                self.reward_to_go[prev_traj_length:traj_length] = returns
                prev_traj_length = traj_length
            log.info(f"Computed reward-to-go for each trajectory.")

    def make_indices(self, traj_lengths):
        """
        Generates sampling indices from the dataset; each index maps to a datapoint.
        """
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length
            if not self.dones[cur_traj_index + traj_length - 1]:  # truncation
                max_start -= 1
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start)
            ]
            cur_traj_index += traj_length
        return indices
    
    def __getitem__(self, idx):
        start, _ = self.indices[idx]
        end = start + 1
        actions = self.actions[start:end]
        rewards = self.rewards[start:end]
        dones = self.dones[start:end]
        reward_to_gos = self.reward_to_go[start:end] if self.get_mc_return else tf.zeros_like(rewards)

        batch = {
            "states": self.states[start:end],
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "reward_to_go": reward_to_gos,
        }
        return batch

    @property
    def element_spec(self):
        # Define the element specification based on dataset content
        return {
            "states": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "actions": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "rewards": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "dones": tf.TensorSpec(shape=(None,), dtype=tf.float32),
            "reward_to_go": tf.TensorSpec(shape=(None,), dtype=tf.float32),
        }
        
    def _inputs(self):
        # Returns an empty list if no additional inputs are required
        return []