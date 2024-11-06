import logging
import pickle
import random
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "actions conditions")
Transition = namedtuple("Transition", "actions conditions rewards dones")
TransitionWithReturn = namedtuple(
    "Transition", "actions conditions rewards dones reward_to_gos"
)


class StitchedSequenceDataset(tf.keras.utils.Sequence):
    """
    Load stitched trajectories of states/actions/images, and 1-D array of traj_lengths, from npz or pkl file.

    Use the first max_n_episodes episodes (instead of random sampling)

    Example:
        states: [----------traj 1----------][---------traj 2----------] ... [---------traj N----------]
        Episode IDs (determined based on traj_lengths):  [----------   1  ----------][----------   2  ---------] ... [----------   N  ---------]

    Each sample is a namedtuple of (1) chunked actions and (2) a list (obs timesteps) of dictionary with keys states and images.

    """
    def __new__(
            self,
            dataset_path,
            horizon_steps=64,
            cond_steps=1,
            img_cond_steps=1,
            max_n_episodes=10000,
            use_img=False,
            device="GPU:0"
    ):
        assert img_cond_steps <= cond_steps, "consider using more cond_steps than img_cond_steps"
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
        # if "GPU" in device.upper():
        self.states = tf.convert_to_tensor(dataset["states"][:total_num_steps], dtype=tf.float32)
        self.actions = tf.convert_to_tensor(dataset["actions"][:total_num_steps], dtype=tf.float32)

        log.info(f"Loaded dataset from {dataset_path}")
        log.info(f"Number of episodes: {min(max_n_episodes, len(traj_lengths))}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")

        if self.use_img:
            self.images = tf.convert_to_tensor(dataset["images"][:total_num_steps], dtype=tf.float32)
            log.info(f"Images shape/type: {self.images.shape, self.images.dtype}")

        return tf.data.Dataset.from_generator(
            self._generator,
            output_signature=self.element_spec,
        )

    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start): (start + 1)]
        actions = self.actions[start:end]
        states = tf.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        conditions = {"state": states}
        if self.use_img:
            images = self.images[(start - num_before_start): end]
            images = tf.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images
        batch = Batch(actions, conditions)
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
            spec["conditions"]["rgb"] = tf.TensorSpec(shape=(self.img_cond_steps, *self.images.shape[1:]),
                                                      dtype=tf.float32)
        return spec

    def _generator(self):
        for idx in range(len(self)):
            yield {
                "actions": self[idx].actions,
                "conditions": {
                    "state": self[idx].conditions["state"]
                }
            }

    def _inputs(self):
        return []


class StitchedSequenceQLearningDataset(StitchedSequenceDataset):
    """
    Extends StitchedSequenceDataset to include rewards and dones for Q learning

    Do not load the last step of **truncated** episodes since we do not have the correct next state for the final step of each episode. Truncation can be determined by terminal=False but end of episode.
    """

    def __init__(
            self,
            dataset_path,
            max_n_episodes=10000,
            discount_factor=1.0,
            device="cuda:0",
            get_mc_return=False,
            **kwargs,
    ):
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]
        total_num_steps = np.sum(traj_lengths)

        # discount factor
        self.discount_factor = discount_factor

        # rewards and dones(terminals)
        self.rewards = (
            tf.convert_to_tensor(dataset["rewards"][:total_num_steps], dtype=tf.float32)
        )
        log.info(f"Rewards shape/type: {self.rewards.shape, self.rewards.dtype}")
        self.dones = (
            tf.convert_to_tensor(dataset["terminals"][:total_num_steps], dtype=tf.float32)
        )
        log.info(f"Dones shape/type: {self.dones.shape, self.dones.dtype}")

        super().__init__(
            dataset_path=dataset_path,
            max_n_episodes=max_n_episodes,
            **kwargs,
        )
        log.info(f"Total number of transitions using: {len(self)}")

        # compute discounted reward-to-go for each trajectory
        self.get_mc_return = get_mc_return
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
                    returns[-t - 1] = (
                            traj_rewards[-t - 1] + self.discount_factor * prev_return
                    )
                    prev_return = returns[-t - 1]
                self.reward_to_go[prev_traj_length:traj_length] = returns
                prev_traj_length = traj_length
            log.info(f"Computed reward-to-go for each trajectory.")

    def make_indices(self, traj_lengths, horizon_steps):
        """
        skip last step of truncated episodes
        """
        num_skip = 0
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            if not self.dones[cur_traj_index + traj_length - 1]:  # truncation
                max_start -= 1
                num_skip += 1
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        log.info(f"Number of transitions skipped due to truncation: {num_skip}")
        return indices

    def __getitem__(self, idx):
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start): (start + 1)]
        actions = self.actions[start:end]
        rewards = self.rewards[start: (start + 1)]
        dones = self.dones[start: (start + 1)]

        # Account for action horizon
        if idx < len(self.indices) - self.horizon_steps:
            next_states = self.states[
                (start - num_before_start + self.horizon_steps): start + 1 + self.horizon_steps
            ]
        else:
            # prevents indexing error, but ignored since done=True
            next_states = tf.zeros_like(states)

        # stack obs history
        states = tf.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        next_states = tf.stack(
            [
                next_states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        conditions = {"state": states, "next_state": next_states}
        if self.use_img:
            images = self.images[(start - num_before_start): end]
            images = tf.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images
        if self.get_mc_return:
            reward_to_gos = self.reward_to_go[start: (start + 1)]
            batch = TransitionWithReturn(
                actions,
                conditions,
                rewards,
                dones,
                reward_to_gos,
            )
        else:
            batch = Transition(
                actions,
                conditions,
                rewards,
                dones,
            )
        return batch
import logging
import pickle
import random
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "actions conditions")
Transition = namedtuple("Transition", "actions conditions rewards dones")
TransitionWithReturn = namedtuple(
    "Transition", "actions conditions rewards dones reward_to_gos"
)


class StitchedSequenceDataset(tf.keras.utils.Sequence):
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
            device="GPU:0"
    ):
        assert img_cond_steps <= cond_steps, "consider using more cond_steps than img_cond_steps"
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
        # if "GPU" in device.upper():
        self.states = tf.convert_to_tensor(dataset["states"][:total_num_steps], dtype=tf.float32)
        self.actions = tf.convert_to_tensor(dataset["actions"][:total_num_steps], dtype=tf.float32)

        log.info(f"Loaded dataset from {dataset_path}")
        log.info(f"Number of episodes: {min(max_n_episodes, len(traj_lengths))}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")

        if self.use_img:
            self.images = tf.convert_to_tensor(dataset["images"][:total_num_steps], dtype=tf.float32)
            log.info(f"Images shape/type: {self.images.shape, self.images.dtype}")

    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start): (start + 1)]
        actions = self.actions[start:end]
        states = tf.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        conditions = {"state": states}
        if self.use_img:
            images = self.images[(start - num_before_start): end]
            images = tf.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images
        batch = Batch(actions, conditions)
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
            spec["conditions"]["rgb"] = tf.TensorSpec(shape=(self.img_cond_steps, *self.images.shape[1:]),
                                                      dtype=tf.float32)
        return spec
        # actions_spec = tf.TensorSpec(shape=(self.horizon_steps, self.actions.shape[-1]), dtype=tf.float32)
        # conditions_spec =  {
        #     "state": tf.TensorSpec(shape=(self.cond_steps, self.states.shape[-1]), dtype=tf.float32),
        # }
        # if self.use_img:
        #     conditions_spec["rgb"] = tf.TensorSpec(shape=(self.img_cond_steps, *self.images.shape[1:]), dtype=tf.float32)
        # return (actions_spec, conditions_spec)
    
    def _inputs(self):
        return []


class StitchedSequenceQLearningDataset(StitchedSequenceDataset):
    """
    Extends StitchedSequenceDataset to include rewards and dones for Q learning

    Do not load the last step of **truncated** episodes since we do not have the correct next state for the final step of each episode. Truncation can be determined by terminal=False but end of episode.
    """

    def __init__(
            self,
            dataset_path,
            max_n_episodes=10000,
            discount_factor=1.0,
            device="cuda:0",
            get_mc_return=False,
            **kwargs,
    ):
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]
        total_num_steps = np.sum(traj_lengths)

        # discount factor
        self.discount_factor = discount_factor

        # rewards and dones(terminals)
        self.rewards = (
            tf.convert_to_tensor(dataset["rewards"][:total_num_steps], dtype=tf.float32)
        )
        log.info(f"Rewards shape/type: {self.rewards.shape, self.rewards.dtype}")
        self.dones = (
            tf.convert_to_tensor(dataset["terminals"][:total_num_steps], dtype=tf.float32)
        )
        log.info(f"Dones shape/type: {self.dones.shape, self.dones.dtype}")

        super().__init__(
            dataset_path=dataset_path,
            max_n_episodes=max_n_episodes,
            **kwargs,
        )
        log.info(f"Total number of transitions using: {len(self)}")

        # compute discounted reward-to-go for each trajectory
        self.get_mc_return = get_mc_return
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
                    returns[-t - 1] = (
                            traj_rewards[-t - 1] + self.discount_factor * prev_return
                    )
                    prev_return = returns[-t - 1]
                self.reward_to_go[prev_traj_length:traj_length] = returns
                prev_traj_length = traj_length
            log.info(f"Computed reward-to-go for each trajectory.")

    def make_indices(self, traj_lengths, horizon_steps):
        """
        skip last step of truncated episodes
        """
        num_skip = 0
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            if not self.dones[cur_traj_index + traj_length - 1]:  # truncation
                max_start -= 1
                num_skip += 1
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        log.info(f"Number of transitions skipped due to truncation: {num_skip}")
        return indices

    def __getitem__(self, idx):
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start): (start + 1)]
        actions = self.actions[start:end]
        rewards = self.rewards[start: (start + 1)]
        dones = self.dones[start: (start + 1)]

        # Account for action horizon
        if idx < len(self.indices) - self.horizon_steps:
            next_states = self.states[
                (start - num_before_start + self.horizon_steps): start + 1 + self.horizon_steps
            ]
        else:
            # prevents indexing error, but ignored since done=True
            next_states = tf.zeros_like(states)

        # stack obs history
        states = tf.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        next_states = tf.stack(
            [
                next_states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        conditions = {"state": states, "next_state": next_states}
        if self.use_img:
            images = self.images[(start - num_before_start): end]
            images = tf.stack(
                [
                    images[max(num_before_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images
        if self.get_mc_return:
            reward_to_gos = self.reward_to_go[start: (start + 1)]
            batch = TransitionWithReturn(
                actions,
                conditions,
                rewards,
                dones,
                reward_to_gos,
            )
        else:
            batch = Transition(
                actions,
                conditions,
                rewards,
                dones,
            )
        return batch
