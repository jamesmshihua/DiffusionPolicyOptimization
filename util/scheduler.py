import math

import tensorflow as tf


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


class CosineAnnealingWarmupRestarts2(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    optimizer (Optimizer): Wrapped optimizer.
    first_cycle_steps (int): First cycle step size.
    cycle_mult(float): Cycle steps magnification. Default: -1.
    max_lr(float): First cycle's max learning rate. Default: 0.1.
    min_lr(float): Min learning rate. Default: 0.001.
    warmup_steps(int): Linear warmup step size. Default: 0.
    gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
    last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
            self,
            initial_learning_rate: float,
            first_cycle_steps: int,
            cycle_mult: float = 1.0,
            max_lr: float = 0.1,
            min_lr: float = 0.001,
            warmup_steps: int = 0,
            gamma: float = 1.0,
            last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.initial_learning_rate = initial_learning_rate
        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super().__init__()

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.initial_learning_rate
        elif self.step_in_cycle < self.warmup_steps:
            return ((self.max_lr - self.initial_learning_rate)
                    * self.step_in_cycle / self.warmup_steps
                    + self.initial_learning_rate)
        else:
            return (self.initial_learning_rate
                    + (self.max_lr - self.initial_learning_rate)
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps)
                                    / (self.cur_cycle_steps - self.warmup_steps)))
                    / 2)

    def __call__(self, step):
        # if step is None:
        #     step = self.last_epoch + 1
        #     self.step_in_cycle = self.step_in_cycle + 1
        #     if self.step_in_cycle >= self.cur_cycle_steps:
        #         self.cycle += 1
        #         self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
        #         self.cur_cycle_steps = (
        #                 int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
        #                 + self.warmup_steps
        #         )
        # else:
        if step >= self.first_cycle_steps:
            if self.cycle_mult == 1.0:
                self.step_in_cycle = step % self.first_cycle_steps
                self.cycle = step // self.first_cycle_steps
            else:
                n = int(
                    math.log(
                        (
                                step / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                        ),
                        self.cycle_mult,
                    )
                )
                self.cycle = n
                self.step_in_cycle = step - int(
                    self.first_cycle_steps
                    * (self.cycle_mult ** n - 1)
                    / (self.cycle_mult - 1)
                )
                self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                    n
                )
        else:
            self.cur_cycle_steps = self.first_cycle_steps
            self.step_in_cycle = step

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(step)
        # for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
        #     param_group["lr"] = lr
        return self.get_lr()

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "first_cycle_steps": self.first_cycle_steps,
            "cycle_mult": self.cycle_mult,
            "base_max_lr": self.base_max_lr,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "gamma": self.gamma,
        }
