import tensorflow as tf
import math


class CosineAnnealingWarmupRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            first_cycle_steps,
            cycle_mult=1.0,
            max_lr=0.1,
            min_lr=0.001,
            warmup_steps=0,
            gamma=1.0
    ):
        super().__init__()
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma

        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0

    def __call__(self, step):
        step_in_cycle = step - self._start_of_cycle(step)
        if step_in_cycle < self.warmup_steps:
            # Warmup phase: linearly increase from min_lr to max_lr
            lr = self.min_lr + (self.max_lr - self.min_lr) * (step_in_cycle / self.warmup_steps)
        else:
            # Cosine annealing phase
            progress = (step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2

        # Adjust max_lr for the current cycle
        adjusted_max_lr = self.max_lr * (self.gamma ** self.cycle)
        return tf.where(step >= self._start_of_cycle(step), lr, adjusted_max_lr)

    def _start_of_cycle(self, step):
        """
        Calculate the step at which the current cycle starts.
        """
        cycle_start = 0
        cycle_steps = self.first_cycle_steps
        while step >= cycle_start + cycle_steps:
            cycle_start += cycle_steps
            cycle_steps = int((cycle_steps - self.warmup_steps) * self.cycle_mult + self.warmup_steps)
            self.cycle += 1
        self.cur_cycle_steps = cycle_steps
        return cycle_start

    def get_config(self):
        return {
            "first_cycle_steps": self.first_cycle_steps,
            "cycle_mult": self.cycle_mult,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "gamma": self.gamma,
        }